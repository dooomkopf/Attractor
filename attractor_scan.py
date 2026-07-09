#!/usr/bin/env python3
"""
Scan: Optimize embedding window W so that all halving events
      are maximally equidistant in the polar plot (minimum arc spread).
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
M_LIST     = list(range(21, 66, 2))  # 21, 23, 25, ..., 65 (enthält 37)
STEP_Y     = 0.5         # 6 months
Y_MIN      = 2.5
Y_MAX      = 4.5
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
    days  = np.array([d[0] for d in data])
    prices = np.array([d[1] for d in data])
    dates  = np.array([d[2] for d in data])
    return days, prices, dates


def min_arc(angles_rad):
    """Minimum arc (rad) enclosing all angles."""
    a = np.array(angles_rad) % (2 * np.pi)
    a = np.sort(a)
    gaps = np.diff(np.append(a, a[0] + 2 * np.pi))
    return float(2 * np.pi - np.max(gaps))


def equidist_error(angles_rad):
    """Std of gaps between consecutive halvings (rad). 0 = perfectly equidistant."""
    a = np.array(angles_rad) % (2 * np.pi)
    a = np.sort(a)
    all_gaps = np.diff(np.append(a, a[0] + 2 * np.pi))
    # largest gap = empty arc → discard
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
    X        = sm.add_constant(np.log(days_all))
    qr       = QuantReg(np.log(prices_all), X).fit(q=PERCENTILE)
    log_res_all = np.log(prices_all / np.exp(qr.predict(X)))

    mask_emb = days_all >= START_IDX
    log_res  = log_res_all[mask_emb]
    days_emb = days_all[mask_emb]
    N        = len(log_res)

    # ── Scan für jedes M ────────────────────────────────────────────────────────
    years_range = np.arange(Y_MIN, Y_MAX + STEP_Y / 2, STEP_Y)

    def _scan_one_m(M_val):
        """W-Scan für ein M. Gibt (results, yy_fine, eq_fine, bal_fine, best_*) zurück."""
        results = []
        for y in years_range:
            W   = round(y * 365)
            TAU = max(1, round(W / (M_val - 1)))
            W   = TAU * (M_val - 1)
            if W >= N:
                continue
            D = np.empty((N - W, M_val))
            for j in range(M_val):
                D[:, j] = log_res[W - j * TAU : N - j * TAU]
            D_c      = D - D.mean(axis=0)
            _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
            pc       = D_c @ Vt.T
            var      = s**2 / (s**2).sum()
            days_vecs = days_emb[W:]
            theta     = np.arctan2(pc[:, 1], pc[:, 0])
            r         = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
            halv_thetas, halv_r, halv_idx = [], [], []
            for hi, hday in enumerate(halving_days):
                hidx = np.argmin(np.abs(days_vecs - hday))
                if np.abs(days_vecs[hidx] - hday) < 200:
                    halv_thetas.append(float(theta[hidx]))
                    halv_r.append(float(r[hidx]))
                    halv_idx.append(hi)
            if len(halv_thetas) < 3:
                continue
            results.append({
                'y': y, 'W': W, 'TAU': TAU,
                'balance': float(var[1] / var[0]) if var[0] > 0 else 0,
                'eqdist': equidist_error(halv_thetas),
                'arc': min_arc(halv_thetas),
                'halv_thetas': halv_thetas, 'halv_r': halv_r, 'halv_idx': halv_idx,
                'theta': theta, 'r': r, 'days_vecs': days_vecs,
            })
        if len(results) < 3:
            return None
        yy      = np.array([r['y']      for r in results])
        eq_arr  = np.array([r['eqdist'] for r in results])
        bal_arr = np.array([r['balance'] for r in results])
        yy_fine = np.linspace(yy[0], yy[-1], 500)
        spl_eq  = make_interp_spline(yy, np.degrees(eq_arr), k=3)
        eq_fine = spl_eq(yy_fine)
        spl_bal = make_interp_spline(yy, bal_arr, k=3)
        bal_fine = spl_bal(yy_fine)
        best_eq_i  = np.argmin(eq_fine)
        best_bal_i = np.argmin(np.abs(bal_fine - 1.0))
        best = results[np.argmin(eq_arr)]
        return {
            'results': results, 'yy': yy, 'eq_arr': eq_arr, 'bal_arr': bal_arr,
            'yy_fine': yy_fine, 'eq_fine': eq_fine, 'bal_fine': bal_fine,
            'best_eq_y': yy_fine[best_eq_i], 'best_eq_val': eq_fine[best_eq_i],
            'best_bal_y': yy_fine[best_bal_i], 'best_bal_val': bal_fine[best_bal_i],
            'best': best,
        }

    all_scans = []
    for mi, M in enumerate(M_LIST):
        pass
        s = _scan_one_m(M)
        if s:
            all_scans.append((M, mi, s))

    if not all_scans:
        print("No results.")
        return

    # Polar-Plot: vom ersten M (Referenz)
    best = all_scans[0][2]['best']

    # ── Scan plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(13, 4))
    ax2 = ax.twinx()

    # Span über alle M
    _all_best_y = []
    for M, mi, s in all_scans:
        _all_best_y.extend([s['best_eq_y'], s['best_bal_y']])
    period_min = min(_all_best_y)
    period_max = max(_all_best_y)
    ax.axvspan(period_min, period_max, color='#888888', alpha=0.12, zorder=0,
               label=f'optimal window [{period_min:.2f}y--{period_max:.2f}y]')

    _first_eq = True
    _first_bal = True
    _M_REF = 35  # Surrogate-Referenz
    # Zeichne M_REF zuletzt (oben)
    _sorted_scans = sorted(all_scans, key=lambda x: x[0] == _M_REF)
    for M, mi, s in _sorted_scans:
        _is_ref = (M == _M_REF)
        _c_eq   = '#44FF44' if _is_ref else 'white'
        _c_bal  = '#44FF44' if _is_ref else '#66AAFF'
        _c_dot  = '#44FF44' if _is_ref else 'yellow'
        _lw     = 2.5 if _is_ref else 1.5
        _alpha  = 0.9 if _is_ref else 0.5
        _s_dot  = 80 if _is_ref else 50
        # Datenpunkte
        ax.scatter(s['yy'], np.degrees(s['eq_arr']), color='#666666', s=15, zorder=5)
        ax2.scatter(s['yy'], s['bal_arr'], color='#334466', s=15, zorder=5)
        # Fit-Kurven
        _lbl_eq  = (r'Equidistance error $\sigma$' if _first_eq else
                     (f'M={_M_REF} (surrogate ref.)' if _is_ref else None))
        _lbl_bal = ('PC2/PC1' if _first_bal else None)
        ax.plot(s['yy_fine'], s['eq_fine'], color=_c_eq, linewidth=_lw, alpha=_alpha,
                label=_lbl_eq)
        ax2.plot(s['yy_fine'], s['bal_fine'], color=_c_bal, linewidth=_lw, alpha=_alpha,
                 linestyle='--', label=_lbl_bal)
        _first_eq = _first_bal = False
        # Optimum-Punkte
        ax.scatter([s['best_eq_y']], [s['best_eq_val']],
                   color=_c_dot, s=_s_dot, alpha=_alpha, zorder=10)
        ax2.scatter([s['best_bal_y']], [s['best_bal_val']],
                    color=_c_dot, s=_s_dot, alpha=_alpha, zorder=10)

    ax.set_ylabel(r'$\sigma$ [$^\circ$]', color='white')
    ax.tick_params(axis='y', colors='white')
    ax2.set_ylabel('PC2 / PC1', color='#66AAFF')
    ax2.tick_params(axis='y', colors='#66AAFF')
    ax.set_xlabel('Embedding window W [years]')
    ax.set_xticks(np.arange(Y_MIN, Y_MAX + 0.01, 0.5))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # ylim: auto mit margins
    ax.margins(y=0.10)
    ax2.margins(y=0.10)

    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(f'Halving Phase Scan  M = (20, 65, inc5)',
                     color='#CCCCCC', fontsize=13, y=0.955,
                     fontname='Comfortaa', fontweight='bold')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
              facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#E0E0E0',
              loc='center left')
    plt.subplots_adjust(top=0.88, left=0.07, right=0.93)

    # ── Polar-Plot: Halving-Positionen für alle M ──────────────────────────────
    COLORS_H = ['#CCDDFF', '#88BBFF', '#5599FF', '#2277FF']
    YEARS_H  = ['2012', '2016', '2020', '2024']

    fig2 = plt.figure(figsize=(7, 7))
    ax_pol = fig2.add_subplot(111, projection='polar')

    _seen_hi = set()
    for M, mi, s in all_scans:
        res = s['best']
        _is_ref = (M == _M_REF)
        for th, rv, hi_ in zip(res['halv_thetas'], res['halv_r'], res['halv_idx']):
            _mk = '^' if hi_ == 2 else 'o'  # 2020 = Dreieck
            if _is_ref:
                ax_pol.scatter(th, rv, color='#44FF44', s=100, alpha=0.9,
                               marker=_mk, edgecolors='white', linewidths=1.0, zorder=11)
            else:
                ax_pol.scatter(th, rv, color=COLORS_H[hi_], s=40, alpha=0.5,
                               marker=_mk, edgecolors='none', zorder=5)
            _seen_hi.add(hi_)

    from matplotlib.lines import Line2D
    _legend_handles = []
    for hi in sorted(_seen_hi):
        _mk = '^' if hi == 2 else 'o'
        _legend_handles.append(Line2D([0], [0], marker=_mk, color='w', markerfacecolor=COLORS_H[hi],
                                       markersize=8, linestyle='None', label=YEARS_H[hi]))
    _legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#44FF44',
                                   markeredgecolor='white', markersize=8, linestyle='None',
                                   label=f'M={_M_REF} (surrogate ref.)'))

    ax_pol.set_yticklabels([])
    ax_pol.tick_params(colors='#CCCCCC')
    ax_pol.legend(handles=_legend_handles, fontsize=10, facecolor='#1A1A1A',
                  edgecolor='#808080', labelcolor='#E0E0E0', loc='upper right',
                  title='Halving', title_fontsize=10)
    with plt.rc_context({'text.usetex': False}):
        fig2.suptitle('Halving Equidistance Optimization',
                      color='#CCCCCC', fontsize=13, y=0.975,
                      fontname='Comfortaa', fontweight='bold')
        fig2.text(0.5, 0.93, f'M={", ".join(str(m) for m in M_LIST)}, best W per M',
                  ha='center', color='#888888', fontsize=9)
    plt.subplots_adjust(top=0.90, bottom=0.06)

    plt.show()


if __name__ == '__main__':
    main()
