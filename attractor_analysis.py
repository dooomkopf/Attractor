#!/usr/bin/env python3
"""
Lyapunov-Exponent des BTC PCA-Attraktors.

Workflow:
  1. Daten + Delay-Embedding identisch zu attractor_cy.py laden
  2. Rosenstein-Methode → globaler λ₁ (+ lokale FTLEs nach Zyklus)
  3. Attraktor eingefärbt nach FTLE (PC1/PC2 + 3D)

Parameter: TAU, M, START_IDX wie in attractor_cy.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")

# ── PARAMETER (identisch zu attractor_cy.py) ──────────────────────────────────
PERCENTILE = 0.01
START_IDX  = 1164
TAU        = 30
M          = 50
W          = (M - 1) * TAU   # 1470

# ── Rosenstein-Einstellungen ───────────────────────────────────────────────────
LY_DT          = 1.0          # 1 Tag pro Zeitschritt
LY_THEILER     = W            # = 1470
LY_MAX_HORIZON = 120
LY_FIT_RANGE   = (0, 120)
FTLE_SMOOTH_WINDOW = 180

CYCLES_JSON  = '/home/hz/Data/gold/cycles.json'

# ── Darstellung ───────────────────────────────────────────────────────────────
try:
    plt.style.use('hz.mplstyle')
except Exception:
    plt.rcParams.update({
        'figure.facecolor':  'black',
        'axes.facecolor':    '#1A1A1A',
        'axes.edgecolor':    '#CCCCCC',
        'axes.labelcolor':   '#CCCCCC',
        'text.color':        '#CCCCCC',
        'xtick.color':       '#CCCCCC',
        'ytick.color':       '#CCCCCC',
        'grid.color':        '#666666',
        'legend.facecolor':  '#1A1A1A',
        'legend.edgecolor':  '#CCCCCC',
        'savefig.facecolor': 'black',
        'font.size':         11,
    })


# ══════════════════════════════════════════════════════════════════════════════
# Hilfsfunktionen
# ══════════════════════════════════════════════════════════════════════════════

def read_btc_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                data.append((int(parts[0]), float(parts[1]),
                             datetime.strptime(parts[2], "%d.%m.%Y")))
    days   = np.array([d[0] for d in data])
    prices = np.array([d[1] for d in data])
    dates  = np.array([d[2] for d in data])
    return days, prices, dates


def _zscore_columns(x):
    x  = np.asarray(x, dtype=float)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (x - mu) / sd


# ══════════════════════════════════════════════════════════════════════════════
# Lyapunov (Rosenstein)
# ══════════════════════════════════════════════════════════════════════════════

def _nearest_valid_neighbor(tree, x, i, theiler, last_start, k0=8):
    n = len(x)
    k = min(max(k0, 2), n)
    while True:
        dists, idxs = tree.query(x[i], k=k)
        dists = np.atleast_1d(dists)
        idxs  = np.atleast_1d(idxs)
        for d, j in zip(dists[1:], idxs[1:]):
            j = int(j)
            if not np.isfinite(d) or d <= 0.0:
                continue
            if j >= last_start:
                continue
            if abs(j - i) <= theiler:
                continue
            return j, float(d)
        if k >= n:
            return None, None
        k = min(n, 2 * k)


def estimate_max_lyapunov_rosenstein(
    pc,
    dt=1.0,
    theiler=1470,
    max_horizon=120,
    fit_range=(10, 50),
    standardize=True,
):
    """
    Rosenstein-Schätzer für den größten Lyapunov-Exponenten.

    Rückgabe: dict mit lambda_global_per_day, lambda_global_per_year,
              r2, pairs_used, t, mean_log_divergence, local_ftle_per_day
    """
    x = np.asarray(pc, dtype=float)
    if standardize:
        x = _zscore_columns(x)

    n          = len(x)
    last_start = n - max_horizon
    tree       = cKDTree(x)
    pairs      = []

    print(f"  Rosenstein: N={n}, theiler={theiler}, max_horizon={max_horizon}")
    print("  Suche Nachbarpaare …", flush=True)
    for i in range(last_start):
        j, d0 = _nearest_valid_neighbor(tree, x, i, theiler, last_start)
        if j is not None:
            pairs.append((i, j, d0))

    print(f"  {len(pairs)} Nachbarpaare gefunden.")

    eps    = np.finfo(float).eps
    curves = []
    for i, j, _ in pairs:
        d = np.linalg.norm(x[i:i + max_horizon] - x[j:j + max_horizon], axis=1)
        curves.append(np.log(np.maximum(d, eps)))
    curves = np.asarray(curves)

    mean_log_div = np.nanmean(curves, axis=0)
    t            = np.arange(max_horizon, dtype=float) * dt

    k0, k1 = fit_range
    slope, intercept = np.polyfit(t[k0:k1], mean_log_div[k0:k1], 1)
    fit    = slope * t[k0:k1] + intercept

    ss_res = np.sum((mean_log_div[k0:k1] - fit) ** 2)
    ss_tot = np.sum((mean_log_div[k0:k1] - mean_log_div[k0:k1].mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    local_ftle = [np.polyfit(t[k0:k1], row[k0:k1], 1)[0] for row in curves]
    local_ftle = np.asarray(local_ftle)

    return {
        "lambda_global_per_day":  slope,
        "lambda_global_per_year": slope * 365.25,
        "r2":                     r2,
        "pairs_used":             len(pairs),
        "t":                      t,
        "mean_log_divergence":    mean_log_div,
        "fit_range":              fit_range,
        "local_ftle_per_day":     local_ftle,
        "intercept":              intercept,
        "pair_starts":            np.array([i for i, j, _ in pairs], dtype=int),
        "curves":                 curves,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════════

def _rolling_mean(y, window):
    y = np.asarray(y, dtype=float)
    if len(y) == 0 or window <= 1:
        return y.copy()

    w = min(int(window), len(y))
    kernel = np.ones(w, dtype=float)
    numer  = np.convolve(y, kernel, mode='same')
    denom  = np.convolve(np.ones_like(y), kernel, mode='same')
    return numer / denom


def plot_all(
    lyap,
    periods=None,
    cycle_curves=None,
    pc3=None,
    ftle_dates=None,
    ftle_days=None,
    pca_boundaries=None,
):
    from matplotlib.gridspec import GridSpec
    import matplotlib.dates as mdates
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor('black')
    CL = '#00AAFF'
    # Oben: zwei kleine flache Plots — eng zusammen, wenig Rand
    gs_top = GridSpec(1, 2, figure=fig, left=0.06, right=0.97,
                      top=0.97, bottom=0.68, wspace=0.28)
    # Unten: 3D zentriert und groß
    gs_bot = GridSpec(1, 1, figure=fig, left=0.08, right=0.92,
                      top=0.62, bottom=0.03)
    ax0 = fig.add_subplot(gs_top[0, 0])
    ax1 = fig.add_subplot(gs_top[0, 1])
    ax2 = fig.add_subplot(gs_bot[0, 0], projection='3d')

    # ── Oben links: Mean log divergence ───────────────────────────────────────
    ax = ax0
    ax.set_facecolor('#1A1A1A')
    t   = lyap['t']
    mld = lyap['mean_log_divergence']
    k0, k1 = lyap['fit_range']
    fit = lyap['lambda_global_per_day'] * t[k0:k1] + lyap['intercept']
    if cycle_curves is not None:
        for name, cc in cycle_curves.items():
            ax.plot(t, cc['mean'], color=cc['color'], linewidth=1.2,
                    alpha=0.75, label=f"{name}  λ₁={cc['slope']*365.25:.3f}/yr")
    ax.plot(t, mld, color='white', linewidth=2.0, alpha=0.9, label='All cycles mean')
    ax.plot(t[k0:k1], fit, color='#FF4444', linewidth=2.0, linestyle='--',
            label=f"Global λ₁ = {lyap['lambda_global_per_year']:.3f}/yr  (R²={lyap['r2']:.3f})")
    ax.set_xlabel('Δt  [days]', color='#CCCCCC')
    ax.set_ylabel('⟨ln d(t)⟩', color='#CCCCCC')
    ax.set_title('Rosenstein: Mean log divergence', color='#E0E0E0')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(colors='#CCCCCC')

    # ── Oben rechts: Lokale FTLE-Zeitreihe ───────────────────────────────────
    ax = ax1
    ax.set_facecolor('#1A1A1A')
    ftle = lyap['local_ftle_per_day'] * 365.25
    ftle_bin_colors = np.where(ftle >= 2.0, '#FF4444', '#4488FF')
    if ftle_dates is not None:
        ax.scatter(ftle_dates[ftle < 2.0], ftle[ftle < 2.0], s=8,
                   color='#4488FF', alpha=0.55, linewidths=0, label='FTLE < 2/yr')
        ax.scatter(ftle_dates[ftle >= 2.0], ftle[ftle >= 2.0], s=8,
                   color='#FF4444', alpha=0.55, linewidths=0, label='FTLE ≥ 2/yr')
    else:
        ax.scatter(np.arange(len(ftle)), ftle, c=ftle_bin_colors, s=8,
                   alpha=0.55, linewidths=0)

    if pca_boundaries is not None:
        for _i, _bd in enumerate(pca_boundaries):
            ax.axvline(_bd, color='white', linewidth=1.0, alpha=0.35,
                       linestyle='--', label='PCA cycle' if _i == 0 else None)
    ax.axhline(0.0, color='#BBBBBB', linewidth=1.0, linestyle=':',
               label='FTLE = 0')
    ax.axhline(lyap['lambda_global_per_year'], color='#FF4444', linewidth=1.5,
               linestyle='--', label=f"Global λ₁ = {lyap['lambda_global_per_year']:.3f}/yr")
    if ftle_dates is not None:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlabel('Start date', color='#CCCCCC')
    ax.set_ylabel('FTLE  [1/year]', color='#CCCCCC')
    ax.set_title('Local FTLE over time',
                 color='#E0E0E0')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(colors='#CCCCCC')
    ax.tick_params(axis='x', labelrotation=30)

    # ── Unten (groß): 3D Attraktor nach FTLE ─────────────────────────────────
    if pc3 is not None:
        starts = lyap['pair_starts']
        ftle_v = _rolling_mean(lyap['local_ftle_per_day'] * 365.25, FTLE_SMOOTH_WINDOW)
        vmax   = np.percentile(np.abs(ftle_v), 95)
        ax = ax2
        bin_colors_3d = np.where(ftle_v >= 2.0, '#FF4444', '#4488FF')
        ax.scatter(pc3[starts[ftle_v < 2.0], 0], pc3[starts[ftle_v < 2.0], 1],
                   pc3[starts[ftle_v < 2.0], 2],
                   color='#4488FF', s=6, alpha=0.6, linewidths=0, label='FTLE < 2/yr')
        ax.scatter(pc3[starts[ftle_v >= 2.0], 0], pc3[starts[ftle_v >= 2.0], 1],
                   pc3[starts[ftle_v >= 2.0], 2],
                   color='#FF4444', s=6, alpha=0.6, linewidths=0, label='FTLE ≥ 2/yr')
        ax.legend(fontsize=9, loc='upper left')
        ax.set_xlabel('PC1', color='#CCCCCC', labelpad=4)
        ax.set_ylabel('PC2', color='#CCCCCC', labelpad=4)
        ax.set_zlabel('PC3', color='#CCCCCC', labelpad=4)
        ax.set_title('')
        ax.tick_params(colors='#CCCCCC')
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#444444')

    fig.tight_layout()


def plot_residuals_ftle(dates_all, log_res_all, ftle_dates, ftle_vals, pca_boundaries=None):
    """Zweites Fenster: BTC-Residuen eingefärbt nach lokalem FTLE."""
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('#1A1A1A')

    # Hintergrund: vollständige Residuen (grau)
    ax.plot(dates_all, log_res_all, color='#555555', linewidth=0.6, alpha=0.7)

    # Residuenwerte an den FTLE-Tagen interpolieren
    ftle_res = np.interp(
        [d.toordinal() for d in ftle_dates],
        [d.toordinal() for d in dates_all],
        log_res_all,
    )
    mask_lo = ftle_vals < 2.0
    mask_hi = ftle_vals >= 2.0
    ax.scatter(ftle_dates[mask_lo], ftle_res[mask_lo], color='#4488FF',
               s=6, alpha=0.7, linewidths=0, zorder=3, label='FTLE < 2/yr')
    ax.scatter(ftle_dates[mask_hi], ftle_res[mask_hi], color='#FF4444',
               s=6, alpha=0.7, linewidths=0, zorder=3, label='FTLE ≥ 2/yr')

    if pca_boundaries is not None:
        for _i, _bd in enumerate(pca_boundaries):
            ax.axvline(_bd, color='white', linewidth=1.0, alpha=0.35,
                       linestyle='--', label='PCA cycle' if _i == 0 else None)

    ax.axhline(0.0, color='#888888', linewidth=0.8, linestyle=':')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(colors='#CCCCCC', axis='x', labelrotation=30)
    ax.tick_params(colors='#CCCCCC', axis='y')
    ax.set_xlabel('Date', color='#CCCCCC')
    ax.set_ylabel('log residual  x(t)', color='#CCCCCC')
    ax.set_title('BTC Residuals — colored by local FTLE  (red=expansion, blue=contraction)',
                 color='#E0E0E0')
    ax.legend(fontsize=9)
    fig.tight_layout()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Daten + Embedding (identisch zu attractor_cy.py) ──────────────────────
    days_all, prices_all, dates_all = read_btc_data('ziel.csv')
    log_days_all = np.log(days_all)
    log_btc_all  = np.log(prices_all)
    X_all        = sm.add_constant(log_days_all)
    qr           = QuantReg(log_btc_all, X_all).fit(q=PERCENTILE)
    log_fit_all  = qr.predict(X_all)
    log_res_all  = np.log(prices_all / np.exp(log_fit_all))

    mask_emb = days_all >= START_IDX
    log_res  = log_res_all[mask_emb]
    days_emb = days_all[mask_emb]
    dates_emb = dates_all[mask_emb]
    N        = len(log_res)

    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = log_res[W - j * TAU : N - j * TAU]

    D_c      = D - D.mean(axis=0)
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc       = D_c @ Vt.T
    var      = s**2 / (s**2).sum()

    pc3 = pc[:, :3]

    print(f"Embedding: {N - W} Vektoren, M={M}, τ={TAU}d, W={W}d")
    print(f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  PC3={var[2]*100:.1f}%\n")

    # ── Lyapunov ──────────────────────────────────────────────────────────────
    print("=== Lyapunov (Rosenstein) ===")
    lyap = estimate_max_lyapunov_rosenstein(
        pc3,
        dt=LY_DT,
        theiler=LY_THEILER,
        max_horizon=LY_MAX_HORIZON,
        fit_range=LY_FIT_RANGE,
        standardize=True,
    )
    print(f"  λ₁ = {lyap['lambda_global_per_day']:.5f} / Tag")
    print(f"  λ₁ = {lyap['lambda_global_per_year']:.3f} / Jahr")
    print(f"  R²  = {lyap['r2']:.3f}")
    print(f"  Paare = {lyap['pairs_used']}")
    print(f"  Median lokaler FTLE = {np.median(lyap['local_ftle_per_day'])*365.25:.3f} / Jahr\n")

    # ── Zyklusfarben für FTLEs ────────────────────────────────────────────────
    with open(CYCLES_JSON) as _f:
        _cy = json.load(_f)
    _cycle_labels = ['', "'13 Cycle", "'17 Cycle", "'21 Cycle", "'25 Cycle"]
    _periods = []
    for _i, _p in enumerate(_cy['halving_periods']):
        _col = 'grey' if _i == 0 else (_p['peaks'][0]['color'] if _p['peaks'] else 'orange')
        _periods.append((_p['start_index'], _p['end_index'], _col, _cycle_labels[_i]))

    def _cy_color(day_idx):
        for _s, _e, _c, _ in _periods:
            if _s <= day_idx < _e:
                return _c
        return 'orange'

    days_vecs   = days_emb[W:]
    dates_vecs  = dates_emb[W:]
    ftle_days   = days_vecs[lyap['pair_starts']]
    ftle_dates  = dates_vecs[lyap['pair_starts']]
    ftle_colors = np.array([_cy_color(day_idx) for day_idx in ftle_days])

    # Per-Zyklus mittlere Divergenzkurven
    t = lyap['t']
    k0, k1 = lyap['fit_range']
    cycle_curves = {}
    for _s, _e, _c, _n in _periods:
        if not _n:
            continue
        mask = ftle_colors == _c
        if mask.sum() < 5:
            continue
        mean_c = np.nanmean(lyap['curves'][mask], axis=0)
        slope_c = np.polyfit(t[k0:k1], mean_c[k0:k1], 1)[0]
        cycle_curves[_n] = {'color': _c, 'mean': mean_c, 'slope': slope_c}

    # ── PCA-inhärente Zyklusgrenzen (2π-Kreuzungen von theta_unwrapped) ────────
    theta_uw = np.unwrap(np.arctan2(pc[:, 1], pc[:, 0]))
    pca_boundary_dates = []
    for k in range(1, 20):
        level = k * 2 * np.pi
        cross = np.where((theta_uw[:-1] < level) & (theta_uw[1:] >= level))[0]
        if len(cross) > 0:
            pca_boundary_dates.append(dates_vecs[cross[0]])

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_all(lyap, periods=_periods, cycle_curves=cycle_curves, pc3=pc3,
             ftle_dates=ftle_dates, ftle_days=ftle_days,
             pca_boundaries=pca_boundary_dates)
    ftle_vals = lyap['local_ftle_per_day'] * 365.25
    plot_residuals_ftle(dates_all, log_res_all, ftle_dates, ftle_vals,
                        pca_boundaries=pca_boundary_dates)
    plot_poincare(pc3, dates_vecs, periods=_periods)
    plt.show()


def plot_poincare(pc3, dates_vecs, periods=None):
    """
    Poincaré-Schnitt bei theta = k·2π (jede vollständige Umdrehung).
    Links:  (r_n, PC3_n) — Schnittgeometrie
    Rechts: Rückkehrabbildung r_n → r_{n+1}
    """
    import matplotlib.dates as mdates

    theta_uw = np.unwrap(np.arctan2(pc3[:, 1], pc3[:, 0]))
    r_all    = np.sqrt(pc3[:, 0]**2 + pc3[:, 1]**2)

    # Kreuzungen: wo theta_uw eine neue 2π-Stufe überschreitet
    cross_r, cross_pc3, cross_dates = [], [], []
    for k in range(1, 30):
        level = k * 2 * np.pi
        idx = np.where((theta_uw[:-1] < level) & (theta_uw[1:] >= level))[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        # Lineare Interpolation für genaue Position
        frac = (level - theta_uw[i]) / (theta_uw[i+1] - theta_uw[i])
        cross_r.append(r_all[i]    + frac * (r_all[i+1]    - r_all[i]))
        cross_pc3.append(pc3[i, 2] + frac * (pc3[i+1, 2]   - pc3[i, 2]))
        cross_dates.append(dates_vecs[i])

    cross_r    = np.array(cross_r)
    cross_pc3  = np.array(cross_pc3)
    cross_dates = np.array(cross_dates)
    n = len(cross_r)

    if n < 3:
        print("Zu wenige Poincaré-Kreuzungen für einen Plot.")
        return

    # Zyklusfarben für die Kreuzungspunkte
    def _get_color(date):
        if periods is None:
            return '#00AAFF'
        for _s, _e, _c, _n in periods:
            d = (date - dates_vecs[0]).days + (dates_vecs[0] - dates_vecs[0]).days
            # Vergleich über day-index: Datum → Tag-Index annähern
            return _c   # vereinfacht: erste passende Periode
        return 'orange'

    # Farben: nach Zeit (Index)
    colors = plt.cm.plasma(np.linspace(0, 1, n))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('black')

    # ── Links: Poincaré-Schnitt (r, PC3) ─────────────────────────────────────
    ax1.set_facecolor('#1A1A1A')
    sc = ax1.scatter(cross_r, cross_pc3, c=np.arange(n), cmap='plasma',
                     s=80, zorder=3)
    # Pfeile: n → n+1
    for i in range(n - 1):
        ax1.annotate('', xy=(cross_r[i+1], cross_pc3[i+1]),
                     xytext=(cross_r[i], cross_pc3[i]),
                     arrowprops=dict(arrowstyle='->', color='white',
                                     lw=0.8, alpha=0.4))
    # Jahreszahlen an den Punkten
    for i, d in enumerate(cross_dates):
        ax1.annotate(d.strftime('%y'), (cross_r[i], cross_pc3[i]),
                     textcoords='offset points', xytext=(5, 4),
                     color='white', fontsize=8)
    plt.colorbar(sc, ax=ax1, label='Umlauf-Index')
    ax1.set_xlabel('r = √(PC1² + PC2²)  [Amplitude]', color='#CCCCCC')
    ax1.set_ylabel('PC3', color='#CCCCCC')
    ax1.set_title('Poincaré-Schnitt  θ = k·2π\n'
                  'Punkt → Grenzzyklus | Kurve → Torus | Wolke → Chaos',
                  color='#E0E0E0')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(colors='#CCCCCC')

    # ── Rechts: Rückkehrabbildung r_n → r_{n+1} ──────────────────────────────
    ax2.set_facecolor('#1A1A1A')
    ax2.scatter(cross_r[:-1], cross_r[1:], c=np.arange(n-1), cmap='plasma',
                s=80, zorder=3)
    for i in range(n - 2):
        ax2.annotate('', xy=(cross_r[i+1], cross_r[i+2]),
                     xytext=(cross_r[i], cross_r[i+1]),
                     arrowprops=dict(arrowstyle='->', color='white',
                                     lw=0.8, alpha=0.4))
    for i, d in enumerate(cross_dates[:-1]):
        ax2.annotate(d.strftime('%y'), (cross_r[i], cross_r[i+1]),
                     textcoords='offset points', xytext=(5, 4),
                     color='white', fontsize=8)
    # Diagonale (Fixpunkt-Referenz)
    lim = np.array([cross_r.min()*0.9, cross_r.max()*1.1])
    ax2.plot(lim, lim, color='#666666', linewidth=1.0, linestyle='--',
             alpha=0.6, label='r_n = r_{n+1}')
    ax2.set_xlabel('r_n', color='#CCCCCC')
    ax2.set_ylabel('r_{n+1}', color='#CCCCCC')
    ax2.set_title('Rückkehrabbildung  r_n → r_{n+1}\n'
                  'Diagonal → stabiler Zyklus | Kurve → Torus | Streuung → Chaos',
                  color='#E0E0E0')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(colors='#CCCCCC')
    ax2.legend(fontsize=9)

    fig.tight_layout()


if __name__ == '__main__':
    main()
