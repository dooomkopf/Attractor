#!/usr/bin/env python3
"""
Scan: Fixed embedding window W, vary number of dimensions M.
      W_FIXED ~ 3.77y, TAU = W / (M-1) adjusts automatically.
      Metrics: equidistance error σ and PC2/PC1 ratio.
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from datetime import datetime
from scipy.interpolate import make_interp_spline
import warnings
warnings.filterwarnings("ignore")

# ── PARAMETERS ────────────────────────────────────────────────────────────────
PERCENTILE = 0.01
START_IDX  = 1164
W_YEARS    = 3.77        # fixed embedding window
M_MIN      = 5
M_MAX      = 60
HALVINGS   = [datetime(2012, 11, 28), datetime(2016, 7, 9),
              datetime(2020, 5, 11),  datetime(2024, 4, 20)]
# ──────────────────────────────────────────────────────────────────────────────

try:
    plt.style.use('hz.mplstyle')
except Exception:
    plt.rcParams.update({
        'figure.facecolor': 'black', 'axes.facecolor': '#1A1A1A',
        'axes.edgecolor': '#CCCCCC', 'axes.labelcolor': '#CCCCCC',
        'text.color': '#CCCCCC', 'xtick.color': '#CCCCCC',
        'ytick.color': '#CCCCCC', 'grid.color': '#666666',
        'font.size': 11,
    })


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


def equidist_error(angles_rad):
    """Std of gaps between consecutive halvings (rad). 0 = perfectly equidistant."""
    a = np.array(angles_rad) % (2 * np.pi)
    a = np.sort(a)
    all_gaps = np.diff(np.append(a, a[0] + 2 * np.pi))
    inner_gaps = np.delete(all_gaps, np.argmax(all_gaps))
    return float(np.std(inner_gaps))


def main():
    days_all, prices_all, dates_all = read_btc_data('ziel.csv')

    # halving day indices
    halving_days = []
    for hd in HALVINGS:
        diffs = np.array([abs((d - hd).days) for d in dates_all])
        halving_days.append(days_all[np.argmin(diffs)])
    halving_days = np.array(halving_days)

    # quantile regression – once
    X           = sm.add_constant(np.log(days_all))
    qr          = QuantReg(np.log(prices_all), X).fit(q=PERCENTILE)
    log_res_all = np.log(prices_all / np.exp(qr.predict(X)))

    mask_emb = days_all >= START_IDX
    log_res  = log_res_all[mask_emb]
    days_emb = days_all[mask_emb]
    N        = len(log_res)

    W_base = round(W_YEARS * 365)   # nominal window
    results = []

    for M in range(M_MIN, M_MAX + 1):
        TAU = max(1, round(W_base / (M - 1)))
        W   = TAU * (M - 1)        # effective window
        if W >= N:
            pass
            continue

        D = np.empty((N - W, M))
        for j in range(M):
            D[:, j] = log_res[W - j * TAU : N - j * TAU]

        D_c      = D - D.mean(axis=0)
        _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
        pc       = D_c @ Vt.T
        var      = s**2 / (s**2).sum()

        days_vecs = days_emb[W:]
        theta     = np.arctan2(pc[:, 1], pc[:, 0])
        r         = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)

        halv_thetas = []
        halv_r      = []
        halv_idx    = []
        for hi, hday in enumerate(halving_days):
            hidx = np.argmin(np.abs(days_vecs - hday))
            if np.abs(days_vecs[hidx] - hday) < 200:
                halv_thetas.append(float(theta[hidx]))
                halv_r.append(float(r[hidx]))
                halv_idx.append(hi)

        if len(halv_thetas) < 3:
            pass
            continue

        eqdist  = equidist_error(halv_thetas)
        balance = float(var[1] / var[0]) if var[0] > 0 else 0

        results.append({
            'M': M, 'W': W, 'TAU': TAU,
            'balance': balance,
            'eqdist': eqdist,
            'halv_thetas': halv_thetas,
            'halv_r': halv_r,
            'halv_idx': halv_idx,
        })
        pass

    if not results:
        pass
        return

    mm      = np.array([r['M']       for r in results])
    eq_arr  = np.array([r['eqdist']  for r in results])
    bal_arr = np.array([r['balance'] for r in results])

    best_eq_idx  = np.argmin(eq_arr)
    best_bal_idx = np.argmin(np.abs(bal_arr - 1.0))

    # Savitzky-Golay smoothing
    from scipy.signal import savgol_filter
    eq_deg     = np.degrees(eq_arr)
    window     = min(51, len(mm) - (1 if len(mm) % 2 == 0 else 0))
    if window % 2 == 0:
        window -= 1
    eq_smooth  = savgol_filter(eq_deg, window, polyorder=2)
    bal_smooth = savgol_filter(bal_arr, window, polyorder=2)
    mm_fine    = np.linspace(mm[0], mm[-1], 500)
    spl_eq     = make_interp_spline(mm, eq_smooth, k=3)
    eq_fine    = spl_eq(mm_fine)
    spl_bal    = make_interp_spline(mm, bal_smooth, k=3)
    bal_fine   = spl_bal(mm_fine)

    best_eq_fine_idx  = np.argmin(eq_fine)
    best_bal_fine_idx = np.argmin(np.abs(bal_fine - 1.0))
    best_m_eq  = mm_fine[best_eq_fine_idx]
    best_m_bal = mm_fine[best_bal_fine_idx]

    # SG sweep: nur GÜLTIGE ungerade Fenster bis max(len(mm))
    _max_w = len(mm) - (1 if len(mm) % 2 == 0 else 0)
    if _max_w % 2 == 0:
        _max_w -= 1
    _sg_best_eq  = []
    _sg_best_bal = []
    _sg_curves_eq  = []
    _sg_curves_bal = []
    _seen_w = set()
    for _w in range(17, _max_w + 1, 1):
        _ww = _w if _w % 2 == 1 else _w - 1
        if _ww < 5 or _ww in _seen_w:
            continue
        _seen_w.add(_ww)
        _eq_s  = savgol_filter(eq_deg, _ww, polyorder=2)
        _bal_s = savgol_filter(bal_arr, _ww, polyorder=2)
        _spl_e = make_interp_spline(mm, _eq_s, k=3)
        _spl_b = make_interp_spline(mm, _bal_s, k=3)
        _ef = _spl_e(mm_fine)
        _bf = _spl_b(mm_fine)
        _sg_curves_eq.append(_ef)
        _sg_curves_bal.append(_bf)
        _sg_best_eq.append((mm_fine[np.argmin(_ef)], np.min(_ef)))
        _sg_best_bal.append((mm_fine[np.argmin(np.abs(_bf - 1.0))], _bf[np.argmin(np.abs(_bf - 1.0))]))
    _sg_best_eq  = np.array(_sg_best_eq)
    _sg_best_bal = np.array(_sg_best_bal)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(13, 4))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('#1A1A1A')
    ax2 = ax.twinx()

    _all_optima = np.concatenate([np.round(_sg_best_eq[:, 0]),
                                   np.round(_sg_best_bal[:, 0])])
    span_min = float(np.min(_all_optima))
    span_max = float(np.max(_all_optima))
    ax.axvspan(span_min, span_max, color='#888888', alpha=0.12, zorder=0,
               label=f'Optimal M [{span_min:.0f}--{span_max:.0f}]')

    ax.scatter(mm, np.degrees(eq_arr), color='#666666', s=30, zorder=5)
    # SG sweep Kurven
    for _c in _sg_curves_eq:
        ax.plot(mm_fine, _c, color='white', linewidth=0.8, alpha=0.25, zorder=3)
    ax.plot(mm_fine, eq_fine, color='white', linewidth=2.0, label=r'Equidistance error $\sigma$ [$^\circ$]')
    # SG sweep Optima — ein Punkt pro M-Bucket, Größe ~ Häufigkeit
    if len(_sg_best_eq) > 0:
        _mx_eq = np.round(_sg_best_eq[:, 0]).astype(int)
        _uniq_eq, _counts_eq = np.unique(_mx_eq, return_counts=True)
        _y_eq = np.array([np.mean(_sg_best_eq[_mx_eq == m, 1]) for m in _uniq_eq])
        ax.scatter(_uniq_eq, _y_eq,
                   color='#DDDD66', s=_counts_eq * 30, zorder=9,
                   label='SG sweep optima')
    ax.set_ylabel(r'$\sigma$ [$^\circ$]', color='white')
    ax.tick_params(axis='y', colors='white')

    ax2.scatter(mm, bal_arr, color='#334466', s=30, zorder=5)
    for _c in _sg_curves_bal:
        ax2.plot(mm_fine, _c, color='#66AAFF', linewidth=0.8, alpha=0.25, zorder=3)
    ax2.plot(mm_fine, bal_fine, color='#66AAFF', linewidth=2.0, label='PC2/PC1  (1.0 = circle)')
    if len(_sg_best_bal) > 0:
        _mx_bal = np.round(_sg_best_bal[:, 0]).astype(int)
        _uniq_bal, _counts_bal = np.unique(_mx_bal, return_counts=True)
        _y_bal = np.array([np.mean(_sg_best_bal[_mx_bal == m, 1]) for m in _uniq_bal])
        ax2.scatter(_uniq_bal, _y_bal,
                    color='#DDDD66', s=_counts_bal * 30, zorder=11)
    # Padding: 20% oben, 10% unten
    eq_range = np.max(eq_fine) - np.min(eq_fine)
    ax.set_ylim(max(0, np.min(eq_fine) - eq_range * 0.1), np.max(eq_fine) + eq_range * 0.2)
    bal_range = np.max(bal_fine) - np.min(bal_fine)
    ax2.set_ylim(max(0, np.min(bal_fine) - bal_range * 0.1), np.max(bal_fine) + bal_range * 0.2)
    ax2.set_ylabel('PC2 / PC1', color='#66AAFF')
    ax2.tick_params(axis='y', colors='#66AAFF')

    # M=35 Referenz (Surrogate) — grüner Punkt
    _m_ref = 35
    _ref_idx = np.argmin(np.abs(mm_fine - _m_ref))
    ax.scatter([_m_ref], [eq_fine[_ref_idx]], color='#44FF44', s=80, zorder=12,
               label='M=35 (surrogate ref.)')
    ax2.scatter([_m_ref], [bal_fine[_ref_idx]], color='#44FF44', s=80, zorder=12)

    ax.set_xlim(M_MIN - 1, M_MAX + 1)
    ax.set_xlabel('Dimensions M  (W fixed)', color='#E0E0E0')
    ax.set_xticks(np.arange(M_MIN, M_MAX + 1, 5))
    ax.tick_params(axis='x', colors='#CCCCCC')
    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(f'M scan  (W={W_base}d \u2248 {W_YEARS}y fixed,  \u03c4 = W/(M\u22121))',
                     color='#CCCCCC', fontsize=13, y=0.955,
                     fontname='Comfortaa', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
              facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#E0E0E0',
              loc='center right', markerscale=0.3)
    plt.subplots_adjust(top=0.88, left=0.07, right=0.93)
    plt.show()


if __name__ == '__main__':
    main()
