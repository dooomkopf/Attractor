"""attractor_n_ens/compute.py — Ensemble-n, Delay-Embedding, PCA, p-Zyklen"""

import warnings
import numpy as np
from scipy.ndimage import gaussian_filter1d


def compute_ensemble_n_signal(days_all, prices_all, window_sizes):
    """Build the raw ensemble-n observable from price windows."""
    n_matrix = np.full((len(window_sizes), len(days_all)), np.nan)
    for wi, ws in enumerate(window_sizes):
        half = ws // 2
        for i in range(half, len(days_all) - half):
            t1, t2 = float(days_all[i - half]), float(days_all[i + half])
            p1, p2 = prices_all[i - half], prices_all[i + half]
            if p1 > 0 and p2 > 0 and t2 > t1:
                denom = np.log(t2 / t1)
                if denom != 0:
                    n_matrix[wi, i] = np.log(p2 / p1) / denom

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        daily_n_all = np.nanmean(n_matrix, axis=0)

    half_max = max(window_sizes) // 2
    daily_n_all[:half_max] = np.nan
    daily_n_all[-half_max:] = np.nan
    return daily_n_all, half_max


def _apply_pca_conventions(pc, Vt, phase_offset=0.0):
    """Match the PC sign/rotation convention used across ensemble-n plots."""
    n_comp = Vt.shape[0]
    transform = np.eye(n_comp)
    if n_comp >= 2:
        transform[1, 1] = -1.0
        _cos, _sin = np.cos(phase_offset), np.sin(phase_offset)
        rot = np.array([[_cos, -_sin],
                        [_sin,  _cos]], dtype=float)
        transform[:2, :2] = rot @ transform[:2, :2]
    if n_comp >= 3:
        transform[2, 2] = -1.0
    return pc @ transform.T, transform @ Vt


def build_embedding_context_from_signal(signal, days_all, M, TAU, start_idx, half_max=0, phase_offset=0.0):
    """Build delay-embedding/PCA context in the same format expected by SSMLearn helpers."""
    mask_emb = (days_all >= start_idx - half_max) & np.isfinite(signal)
    sig = signal[mask_emb]
    days_emb = days_all[mask_emb]
    N = len(sig)
    W = (M - 1) * TAU

    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = sig[W - j * TAU: N - j * TAU]

    D_mean = D.mean(axis=0)
    D_c = D - D_mean
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc = D_c @ Vt.T
    pc, Vt = _apply_pca_conventions(pc, Vt, phase_offset=phase_offset)
    var = s**2 / (s**2).sum()

    return {
        'D': D,
        'D_mean': D_mean,
        'D_c': D_c,
        'Vt': Vt,
        'pc': pc,
        'var': var,
        'days_vecs': days_emb[W:].astype(float),
        'days_emb': days_emb,
        'mask_emb': mask_emb,
        'M': M,
        'TAU': TAU,
        'W': W,
        'N': N - W,
        'phase_offset': phase_offset,
    }


def _rosenstein_lyapunov(traj3d, dt=1.0, min_sep=100, k0=20, k1=150):
    """
    Rosenstein (1993) FTLE auf 3D-Phasenraumtrajektorie.
    Gibt λ₁ in yr⁻¹ zurück (dt in Tagen).
    """
    from scipy.spatial import cKDTree
    N     = len(traj3d)
    max_t = k1 + 5
    tree  = cKDTree(traj3d)
    k_q   = min(min_sep * 3, N - 1)
    _, idxs_all = tree.query(traj3d, k=k_q)

    nbrs = np.full(N, -1, dtype=int)
    for i in range(N):
        for idx in idxs_all[i]:
            if abs(int(idx) - i) >= min_sep:
                nbrs[i] = int(idx)
                break

    log_div = np.zeros(max_t)
    count   = np.zeros(max_t, dtype=float)
    for i in range(N):
        j = int(nbrs[i])
        if j < 0:
            continue
        steps = min(max_t, N - i, N - j)
        dists = np.linalg.norm(traj3d[i:i+steps] - traj3d[j:j+steps], axis=1)
        valid = dists > 1e-15
        log_div[:steps][valid] += np.log(dists[valid])
        count[:steps][valid]   += 1

    valid_k  = count > 0
    log_mean = np.where(valid_k, log_div / np.maximum(count, 1), np.nan)
    k_arr    = np.arange(k0, min(k1, max_t))
    mask     = np.isfinite(log_mean[k_arr])
    if mask.sum() < 5:
        return np.nan
    slope, _ = np.polyfit(k_arr[mask], log_mean[k_arr][mask], 1)
    return float(slope * 365.0 / dt)


def compute_all(days_all, prices_all, dates_all, params):
    """Berechnet alles aus Rohdaten. Gibt dict mit allen Arrays zurück."""
    END_CUT      = params['END_CUT']
    START_IDX    = params['START_IDX']
    TAU          = params['TAU']
    M            = params['M']
    SMOOTH_SIGMA = params['SMOOTH_SIGMA']
    WINDOW_SIZES = params['WINDOW_SIZES']
    PHASE_OFFSET = params['PHASE_OFFSET']
    HALVINGS     = params['HALVINGS']
    CYCLE_TOPS   = params['CYCLE_TOPS']
    CYCLE_BOTTOMS= params['CYCLE_BOTTOMS']

    _x_max_fixed = float(days_all[-1])
    if END_CUT > 0:
        days_all   = days_all[:-END_CUT]
        prices_all = prices_all[:-END_CUT]
        dates_all  = dates_all[:-END_CUT]

    # ── Ensemble n(t) ─────────────────────────────────────────────────────────
    daily_n_all, half_max = compute_ensemble_n_signal(days_all, prices_all, WINDOW_SIZES)

    days_n_all  = days_all.copy()
    dates_n_all = dates_all.copy()

    print(f"Ensemble windows: {WINDOW_SIZES}")
    print(f"Signal: {np.sum(np.isfinite(daily_n_all))} gültige Punkte von {len(daily_n_all)}")

    # ── Cycle-Tops / Bottoms / Halvings ───────────────────────────────────────
    def _find_days(event_list):
        result_days, labels, colors = [], [], []
        for dt, lbl, col in event_list:
            diffs = np.array([abs((d - dt).days) for d in dates_all])
            result_days.append(days_all[np.argmin(diffs)])
            labels.append(lbl)
            colors.append(col)
        return np.array(result_days), labels, colors

    used_peak_days,   peak_labels,   peak_colors   = _find_days(CYCLE_TOPS)
    used_bottom_days, bottom_labels, bottom_colors = _find_days(CYCLE_BOTTOMS)

    peak_n_vals = np.array([
        daily_n_all[np.argmin(np.abs(days_n_all - d))] for d in used_peak_days])
    bottom_n_vals = np.array([
        daily_n_all[np.argmin(np.abs(days_n_all - d))] for d in used_bottom_days])

    halving_days = []
    for hd in HALVINGS:
        diffs = np.array([abs((d - hd).days) for d in dates_all])
        halving_days.append(days_all[np.argmin(diffs)])
    halving_days = np.array(halving_days)
    _x_max_fixed = max(_x_max_fixed, float(halving_days[-1]) + 1460)

    # ── Zeitnormierung ────────────────────────────────────────────────────────
    d_min, d_max   = days_all[0], days_all[-1]
    t_norm_n_all   = (days_n_all - d_min) / (d_max - d_min)
    yr_labels      = [dates_all[np.argmin(np.abs(days_all - y))].strftime('%Y')
                      for y in np.linspace(d_min, d_max, 6)]

    # ── Delay Embedding + PCA ─────────────────────────────────────────────────
    ctx = build_embedding_context_from_signal(
        daily_n_all,
        days_n_all,
        M,
        TAU,
        START_IDX,
        half_max=half_max,
        phase_offset=PHASE_OFFSET,
    )
    D = ctx['D']
    D_c = ctx['D_c']
    Vt = ctx['Vt']
    pc = ctx['pc']
    var = ctx['var']
    days_emb = ctx['days_emb']
    mask_emb = ctx['mask_emb']
    dates_emb = dates_n_all[mask_emb]
    N = len(days_emb)
    W = ctx['W']

    pc_s = pc.copy()
    for j in range(pc_s.shape[1]):
        pc_s[:, j] = gaussian_filter1d(pc[:, j], sigma=SMOOTH_SIGMA)

    days_vecs  = ctx['days_vecs']
    dates_vecs = dates_emb[W:]
    t_norm_vec = (days_vecs - d_min) / (d_max - d_min)

    theta    = np.arctan2(pc[:, 1], pc[:, 0])
    r        = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
    theta_uw = np.unwrap(theta)


    # ── PCA-Cycle-Starts (anchored to H2) ────────────────────────────────────
    _h2_idx = int(np.argmin(np.abs(days_vecs - halving_days[1])))
    _pcyc_idxs, _pn = [_h2_idx], 1
    while True:
        _target = theta_uw[_h2_idx] + _pn * 2 * np.pi
        if _target > theta_uw[-1]:
            break
        _pcyc_idxs.append(np.argmin(np.abs(theta_uw - _target)))
        _pn += 1

    if _pcyc_idxs:
        _shift   = float(theta[_pcyc_idxs[0]])
        theta    = (theta    - _shift) % (2 * np.pi)
        theta_uw = theta_uw - _shift

    cum3 = np.cumsum(var)[2] * 100

    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_vecs - hday))
        if np.abs(days_vecs[hidx] - hday) < 200:
            print(f"Halving {i+1}: theta = {theta[hidx]:.4f} rad  ({np.degrees(theta[hidx]):.1f}°)")

    print(f"Embedding: {N-W} Vektoren  M={M} τ={TAU}d W={W}d")
    print(f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  PC3={var[2]*100:.1f}%  kum3={cum3:.1f}%")

    print("Berechne Lyapunov-Exponent (Rosenstein)...")
    lyapunov_ftle = _rosenstein_lyapunov(pc_s[:, :3], dt=1.0, min_sep=100, k0=20, k1=150)
    print(f"FTLE λ₁ = {lyapunov_ftle:+.4f} yr⁻¹")

    return dict(
        lyapunov_ftle=lyapunov_ftle,
        days_all=days_all, prices_all=prices_all, dates_all=dates_all,
        days_n_all=days_n_all, daily_n_all=daily_n_all, dates_n_all=dates_n_all,
        t_norm_n_all=t_norm_n_all, yr_labels=yr_labels,
        halving_days=halving_days, _x_max_fixed=_x_max_fixed,
        used_peak_days=used_peak_days, peak_labels=peak_labels, peak_colors=peak_colors,
        peak_n_vals=peak_n_vals,
        used_bottom_days=used_bottom_days, bottom_labels=bottom_labels,
        bottom_colors=bottom_colors, bottom_n_vals=bottom_n_vals,
        pc=pc, pc_s=pc_s, var=var, cum3=cum3,
        theta=theta, r=r, theta_uw=theta_uw,
        days_vecs=days_vecs, dates_vecs=dates_vecs, t_norm_vec=t_norm_vec,
        _pcyc_idxs=_pcyc_idxs,
    )
