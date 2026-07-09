"""SSM/res/surrogate_compute.py — IAAFT-Surrogate-Pipeline für SSM_res.

Engste Analogie zu surrogates/surrogates_n_IAAFT_compute.py, aber:
    * Datenpfad geht über SSM/res/data.py (load_data, --norm Option)
    * Embedding/PCA geht über SSM/res/embedding.py
    * Diskriminierende Statistik Q ist die SSM-Mannigfaltigkeit-Güte
      (R²_total von geometry.fit_geometry: slaved PCs ≈ Polynom(PC1, PC2))
    * Zusatzstatistik phase_pc3 = R²(PC3 ~ sin 2θ, cos 2θ) — direkte
      Vergleichbarkeit mit dem Vorlage-Test.

API:
    iaaft_surrogate(x, rng, n_iter, tol)
    compute_original_only(params)            → dict
    compute_surrogate_payload(orig, seed, p) → dict

Aufbau spiegelt 1:1 surrogates_n_IAAFT_compute.compute_original_only /
compute_surrogate_payload, damit der Entry-Point surrogate_ssm.py die
gleiche Struktur wie surrogates_theiler_test.py haben kann.
"""

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Modulpfad: SSM/res/ Geschwister einbinden
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from SSM_res_data       import load_data
from SSM_res_embedding  import build_embedding, pca, smooth_pcs
from SSM_res_geometry   import fit_geometry
from SSM_res_phase      import compute_phase_full


# ── IAAFT (Schreiber & Schmitz 1996) ──────────────────────────────────────────
def iaaft_surrogate(x, rng, n_iter=50, tol=1e-8):
    """IAAFT: erhält Amplitudenspektrum + Verteilung der Originalwerte.
    Identisch zu surrogates_n_IAAFT_compute.iaaft_surrogate.
    """
    n        = len(x)
    X_orig   = np.fft.rfft(x)
    amp      = np.abs(X_orig)
    x_sorted = np.sort(x)
    s        = rng.permutation(x)
    prev_err = np.inf
    for _ in range(n_iter):
        S     = np.fft.rfft(s)
        S_new = amp * np.exp(1j * np.angle(S))
        S_new[0] = X_orig[0]
        if n % 2 == 0:
            S_new[-1] = X_orig[-1]
        s_tmp = np.fft.irfft(S_new, n=n)
        ranks = np.argsort(np.argsort(s_tmp))
        s     = x_sorted[ranks]
        err   = np.sum((np.abs(np.fft.rfft(s)) - amp) ** 2) / (np.sum(amp ** 2) + 1e-30)
        if err < tol or abs(prev_err - err) < tol * 1e-3:
            break
        prev_err = err
    return s


# ── Diskriminierende Statistiken ──────────────────────────────────────────────
def _phase_pc3_corr(theta, pc3, intercept=False):
    """R² von PC3 ~ sin(2θ) + cos(2θ).  Identisch zum Vorlage-Test."""
    if intercept:
        X = np.column_stack([np.ones(len(theta)), np.sin(2 * theta), np.cos(2 * theta)])
    else:
        X = np.column_stack([np.sin(2 * theta), np.cos(2 * theta)])
    beta, _, _, _ = np.linalg.lstsq(X, pc3, rcond=None)
    pc3_hat = X @ beta
    ss_res = np.sum((pc3 - pc3_hat) ** 2)
    ss_tot = np.sum((pc3 - pc3.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _phase_frobenius_R2(theta, Y, intercept=True):
    """Multivariater Frobenius-R² der Regression Y = a + B [sin 2θ, cos 2θ]^T.

    Y ist eine (N, K)-Matrix von slaved-PC-Spalten. Das Modell ist eine
    EINZIGE multivariate lineare Regression mit dem regressor space
    span{1, sin 2θ, cos 2θ} (intercept default an).

    Definition (Codex 2026-04-07, Projektor-Form):
        P = X X⁺              (Hat-Matrix der Design-Matrix X)
        J = (1/N) 1 1ᵀ        (Spalten-Mean-Projektor)
        Q = 1 - ‖(I-P) Y‖²_F / ‖(I-J) Y‖²_F

    Eigenschaften:
      * Rotations-invariant in der durch die Spalten von Y aufgespannten
        Ebene: ‖(I-P)(YR)‖_F = ‖((I-P)Y)R‖_F = ‖(I-P)Y‖_F für orthogonales R.
      * Im Single-Spalt-Fall (Y eine Spalte) reduziert sich Q auf das
        normale univariate R² (Vorlagen-Konsistenz).
      * Range: [0, 1] wenn intercept aktiv.

    Args:
        theta:      (N,)  intrinsische Phase
        Y:          (N, K) slaved-PC-Spalten (K ≥ 1)
        intercept:  ob Intercept-Spalte mitfitten (default True; Codex
                    empfiehlt immer an).
    Returns:
        float Q ∈ [0, 1] (oder leicht außerhalb bei numerischen Effekten)
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if Y.shape[0] == 1 and Y.shape[1] != 1:
        Y = Y.T   # falls 1D als Zeile reinkam

    n = Y.shape[0]
    if intercept:
        X = np.column_stack([np.ones(n), np.sin(2 * theta), np.cos(2 * theta)])
    else:
        X = np.column_stack([np.sin(2 * theta), np.cos(2 * theta)])

    # Multi-Output Lstsq (gibt Y_hat = X @ B mit B per Moore-Penrose)
    B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    Y_hat = X @ B

    # Spaltenweise zentriertes Y für ss_tot (Codex C: colmean, nicht skalar)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    ss_res = float(np.sum((Y - Y_hat) ** 2))    # ‖(I-P)Y‖²_F
    ss_tot = float(np.sum(Y_centered ** 2))     # ‖(I-J)Y‖²_F
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _compute_Q_ssm(pc, master_idx=(0, 1), K=5, order=3,
                   phase_pcs=(4, 5), diag_pcs=(3, 4, 5)):
    """SSM-Q (eigene Basis):

    PRIMÄR (Variante C-frob, 2026-04-07 mit User & Codex abgestimmt):
        phase_R2_frob = Frobenius-R² der multivariaten Regression
                        Y = a + B [sin 2θ, cos 2θ]ᵀ
        mit Y = pc[:, [j-1 for j in phase_pcs]]  (z.B. Spalten PC4, PC5).

        Eigenschaft: rotations-invariant in der durch phase_pcs aufgespannten
        Ebene → der natürliche Test gegen das SVD-Rotations-Wackel-Artefakt
        bei nahezu degenerierten slaved-Eigenwerten. Im Single-PC-Fall
        identisch mit dem klassischen univariaten R² der Vorlage.

    DIAGNOSE:
        phase_R2_sum   = Σ_j R²(PC_j ~ sin 2θ + cos 2θ)  (achsenabhängig)
        phase_R2_per_pc[j] für j ∈ diag_pcs — single-PC-Variante
        R²_total + R²_per_pc aus dem Polynom-Fit (slaved ≈ poly(master))
    """
    fit = fit_geometry(pc, master_idx=master_idx, K=K, order=order, start_deg=2)
    u = pc[:, master_idx[0]]
    v = pc[:, master_idx[1]]
    theta = np.arctan2(v, u)

    # Per-PC R² (Diagnose)
    phase_per = {}
    for j in diag_pcs:
        if (j - 1) < pc.shape[1]:
            phase_per[j] = _phase_pc3_corr(theta, pc[:, j - 1])
        else:
            phase_per[j] = float('nan')

    # PRIMÄR: Frobenius-R² über Y = pc[:, phase_pcs-1]
    cols_used = [j - 1 for j in phase_pcs if (j - 1) < pc.shape[1]]
    if cols_used:
        Y_block = pc[:, cols_used]
        phase_R2_frob = _phase_frobenius_R2(theta, Y_block, intercept=True)
    else:
        phase_R2_frob = float('nan')

    # Sum (Diagnose, achsenabhängig)
    sum_used = [phase_per[j] for j in phase_pcs if j in phase_per and np.isfinite(phase_per[j])]
    phase_R2_sum = float(np.sum(sum_used)) if sum_used else float('nan')

    return {
        'phase_R2_frob':   phase_R2_frob,
        'phase_R2_sum':    phase_R2_sum,
        'phase_R2_per_pc': phase_per,
        'phase_pcs':       tuple(phase_pcs),
        'R2_total':        float(fit.R2_total) if np.isfinite(fit.R2_total) else float('nan'),
        'R2_per_pc':       fit.R2_per_pc.tolist(),
        'cond':            float(fit.cond) if np.isfinite(fit.cond) else float('inf'),
    }


def _compute_Q_ssm_origbasis(pc_orig_basis, master_idx=(0, 1), K=5, order=3,
                             phase_pcs=(4, 5), diag_pcs=(3, 4, 5)):
    """Wie _compute_Q_ssm, aber für die Projektion durch das Original-Vt.
    Diese Koordinaten sind für Surrogate nicht garantiert zentriert →
    phase_R2_frob hat per Konstruktion einen Intercept (Codex 2026-04-07,
    immer aktiv). R²_total bleibt informativ als Diagnostik, ist aber kein
    primärer Diskriminator (fit_geometry hat weder Re-Centering noch
    Konstantterm).
    """
    fit = fit_geometry(pc_orig_basis, master_idx=master_idx, K=K, order=order, start_deg=2)
    u = pc_orig_basis[:, master_idx[0]]
    v = pc_orig_basis[:, master_idx[1]]
    theta = np.arctan2(v, u)

    phase_per = {}
    for j in diag_pcs:
        if (j - 1) < pc_orig_basis.shape[1]:
            phase_per[j] = _phase_pc3_corr(theta, pc_orig_basis[:, j - 1], intercept=True)
        else:
            phase_per[j] = float('nan')

    cols_used = [j - 1 for j in phase_pcs if (j - 1) < pc_orig_basis.shape[1]]
    if cols_used:
        Y_block = pc_orig_basis[:, cols_used]
        phase_R2_frob = _phase_frobenius_R2(theta, Y_block, intercept=True)
    else:
        phase_R2_frob = float('nan')

    sum_used = [phase_per[j] for j in phase_pcs if j in phase_per and np.isfinite(phase_per[j])]
    phase_R2_sum = float(np.sum(sum_used)) if sum_used else float('nan')

    return {
        'phase_R2_frob':   phase_R2_frob,
        'phase_R2_sum':    phase_R2_sum,
        'phase_R2_per_pc': phase_per,
        'phase_pcs':       tuple(phase_pcs),
        'R2_total':        float(fit.R2_total) if np.isfinite(fit.R2_total) else float('nan'),
        'R2_per_pc':       fit.R2_per_pc.tolist(),
        'cond':            float(fit.cond) if np.isfinite(fit.cond) else float('inf'),
    }


# ── Original-Kontext (einmal) ─────────────────────────────────────────────────
def compute_original_only(params):
    """Lädt Daten, berechnet QR-Detrending (in data.load_data), Embedding, PCA,
    Phase, Geometrie-Fit. Gibt vollständigen Kontext zurück. Einmalig pro Lauf.
    """
    PERCENTILE   = params['PERCENTILE']
    START_IDX    = params['START_IDX']
    M            = params['M']
    TAU          = params['TAU']
    SMOOTH_SIGMA = params['SMOOTH_SIGMA']
    K            = params['K']
    ORDER        = params['ORDER']
    PHASE_PCS    = tuple(params.get('PHASE_PCS', (4, 5)))
    DIAG_PCS     = tuple(params.get('DIAG_PCS',  (3, 4, 5)))
    do_norm      = params['norm']
    filename     = params['filename']

    # 1) Daten laden + QuantReg + (optional) norm
    data = load_data(filename, START_IDX, percentile=PERCENTILE, norm=do_norm)
    signal       = data['signal']           # log-res ODER (p/QR-1)/(D_exp-1)
    days_emb     = data['days_emb']
    halving_days = data['halving_days']

    # 2) Embedding + PCA
    D, W = build_embedding(signal, M, TAU)
    D_mean = D.mean(axis=0)
    pca_res = pca(D)
    Vt   = pca_res.Vt
    pc   = pca_res.pc
    var  = pca_res.var
    pc_s = smooth_pcs(pc, SMOOTH_SIGMA)

    days_vecs = days_emb[W:]

    # 3) Phase + Cycle-Bounds
    phase_res = compute_phase_full(pc, days_vecs, halving_days,
                                   master_idx=(0, 1), anchor_idx=1)
    bounds = phase_res.bounds

    # 4) Geometrie-Fit (Original) + Q
    Q_orig = _compute_Q_ssm(pc, master_idx=(0, 1), K=K, order=ORDER,
                            phase_pcs=PHASE_PCS, diag_pcs=DIAG_PCS)
    # In der eigenen Basis = Original-Basis, also "_orig" identisch:
    Q_orig['phase_R2_frob_orig']   = Q_orig['phase_R2_frob']
    Q_orig['phase_R2_sum_orig']    = Q_orig['phase_R2_sum']
    Q_orig['phase_R2_per_pc_orig'] = dict(Q_orig['phase_R2_per_pc'])
    Q_orig['R2_total_orig']        = Q_orig['R2_total']

    return {
        'data':         data,
        'signal':       signal,
        'days_emb':     days_emb,
        'days_vecs':    days_vecs,
        'halving_days': halving_days,
        'D_mean':       D_mean,
        'Vt':           Vt,
        'pc':           pc,
        'pc_s':         pc_s,
        'var':          var,
        'phase':        phase_res,
        'bounds':       bounds,
        'theta_orig':   phase_res.theta,
        'Q_orig':       Q_orig,
        'W':  W,  'M':  M,  'TAU':  TAU,
        'K':  K,  'ORDER': ORDER, 'SMOOTH_SIGMA': SMOOTH_SIGMA,
        'PHASE_PCS': PHASE_PCS, 'DIAG_PCS': DIAG_PCS,
    }


# ── Surrogat-Payload (pro Seed) ───────────────────────────────────────────────
def compute_surrogate_payload(orig, seed, params):
    """Ein IAAFT-Lauf (loc oder glob) → Embedding → BEIDE PCA-Basen → Q.

    Liefert sowohl 'eigene' (frische SVD) als auch 'orig' (Projektion durch
    das Vt der Originaldaten) — analog zur Vorlage.
    """
    M            = orig['M']
    W            = orig['W']
    TAU          = orig['TAU']
    K            = orig['K']
    ORDER        = orig['ORDER']
    SMOOTH_SIGMA = orig['SMOOTH_SIGMA']
    PHASE_PCS    = orig['PHASE_PCS']
    DIAG_PCS     = orig['DIAG_PCS']
    N_ITER       = params['N_ITER']
    do_loc       = params['loc']

    signal       = orig['signal']
    days_emb     = orig['days_emb']
    halving_days = orig['halving_days']
    D_mean       = orig['D_mean']
    Vt           = orig['Vt']
    days_vecs    = orig['days_vecs']

    rng = np.random.default_rng(seed)

    # 1) IAAFT — global oder segmentweise (loc: an Halvings ausgerichtet)
    #    Hinweis (Codex 2026-04-07): Der peak-match-Shift aus der Vorlage
    #    (sl += max(seg) - max(sl)) ist bei IAAFT ein No-op, weil IAAFT
    #    die Verteilung exakt erhält → Segmentmaximum bleibt identisch.
    #    Wir lassen ihn deshalb hier weg.
    if do_loc:
        halv_bounds = [days_emb[0] - 1] + list(halving_days) + [days_emb[-1] + 1]
        sl = np.empty_like(signal)
        for k in range(len(halv_bounds) - 1):
            d0, d1 = halv_bounds[k], halv_bounds[k + 1]
            mask   = (days_emb > d0) & (days_emb <= d1)
            seg    = signal[mask]
            if len(seg) < 20:
                sl[mask] = seg
                continue
            sl[mask] = iaaft_surrogate(seg, rng, n_iter=N_ITER)
    else:
        sl = iaaft_surrogate(signal, rng, n_iter=N_ITER)

    # 2) Embedding der Surrogat-Reihe
    Ns = len(sl)
    Ds = np.empty((Ns - W, M))
    for j in range(M):
        Ds[:, j] = sl[W - j * TAU : Ns - j * TAU]

    # 3a) Eigene Basis: frische SVD
    Ds_c = Ds - Ds.mean(axis=0)
    _, ss, Vt_s = np.linalg.svd(Ds_c, full_matrices=False)
    pcs_own  = Ds_c @ Vt_s.T
    var_own  = ss ** 2 / (ss ** 2).sum()

    # 3b) Original-Basis: dieselbe Vt anwenden
    pcs_orig = (Ds - D_mean) @ Vt.T

    # 4) Glättung (für Plot)
    pcs_own_s  = smooth_pcs(pcs_own,  SMOOTH_SIGMA)
    pcs_orig_s = smooth_pcs(pcs_orig, SMOOTH_SIGMA)

    # 5) Q in beiden Basen
    Q_own  = _compute_Q_ssm(pcs_own,  master_idx=(0, 1), K=K, order=ORDER,
                            phase_pcs=PHASE_PCS, diag_pcs=DIAG_PCS)
    Q_o    = _compute_Q_ssm_origbasis(pcs_orig, master_idx=(0, 1), K=K, order=ORDER,
                                      phase_pcs=PHASE_PCS, diag_pcs=DIAG_PCS)

    Q_surr = {
        # ── PRIMÄR (Frobenius-R², rotations-invariant in PC4/PC5-Ebene) ──
        'phase_R2_frob':      Q_own['phase_R2_frob'],
        'phase_R2_frob_orig': Q_o['phase_R2_frob'],
        # ── DIAGNOSE: sum-R² (achsenabhängig) ──
        'phase_R2_sum':      Q_own['phase_R2_sum'],
        'phase_R2_sum_orig': Q_o['phase_R2_sum'],
        # ── DIAGNOSE: per-PC phase R² ──
        'phase_R2_per_pc':      dict(Q_own['phase_R2_per_pc']),
        'phase_R2_per_pc_orig': dict(Q_o['phase_R2_per_pc']),
        # ── DIAGNOSE: Polynom-Manifold ──
        'R2_total':       Q_own['R2_total'],
        'R2_total_orig':  Q_o['R2_total'],
        'R2_per_pc':      Q_own['R2_per_pc'],
        'R2_per_pc_orig': Q_o['R2_per_pc'],
        'cond':           Q_own['cond'],
        'cond_orig':      Q_o['cond'],
    }

    # 6) Phase aus eigener Basis (für Plot Fig 2) — geankert + intrinsic bounds
    #    Wichtig: gleiche Pipeline wie für das Original (compute_phase_full),
    #    sonst sind die Surrogat-Panels gegen das Original-Panel verschoben.
    phase_surr = compute_phase_full(
        pcs_own, days_vecs, halving_days,
        master_idx=(0, 1), anchor_idx=1)

    return {
        'sl':         sl,
        'pcs_own':    pcs_own,
        'pcs_own_s':  pcs_own_s,
        'pcs_orig':   pcs_orig,
        'pcs_orig_s': pcs_orig_s,
        'var_own':    var_own,
        'theta':      phase_surr.theta,
        'pc3':        pcs_own[:, 2],
        'phase':      phase_surr,
        'bounds':     phase_surr.bounds,
        'Q_surr':     Q_surr,
    }
