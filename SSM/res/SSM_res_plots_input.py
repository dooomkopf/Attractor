"""SSM/res/plots_input.py — Sanity-Check-Plot der Eingangsdaten.

Fig 0: BTC-Residuen (price/exp(QuantReg-fit)) auf log-y, Halving-Marker,
       Cycle-Top-Marker, Jahr-Colorbar. Layout angelehnt an
       attractor.py:200-279 (Fenster 1, oberer Plot).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _make_segments(x, y):
    """Aus attractor.py:77-79 — Segmente für LineCollection."""
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def plot_residuals(data, percentile=0.01, fig_num=0, shade_range=None):
    """Fig 0 — Eingangs-Sanity-Plot.

    Zeigt das Signal das tatsächlich ins Embedding geht:
        norm=False:  rohe BTC-Residuen p/QR (log-y, Trend bei 1.0)
        norm=True:   normalisiertes Signal (p/QR - 1) / (D_exp - 1) (linear-y)

    Halving-Marker (lime), Cycle-Top-Marker (rot), farbcodiert nach Jahr.
    """
    days_all   = data['days_all']
    dates_all  = data['dates_all']
    halving_days     = data['halving_days']
    cycle_top_days   = data['cycle_top_days']
    cycle_top_labels = data['cycle_top_labels']
    days_emb_start   = data['days_emb'][0]
    is_norm = bool(data.get('norm', False))

    # Was wird gezeigt?
    if is_norm:
        signal_all = data['signal_all']      # (p/QR - 1) / (D_exp - 1)
        y_label    = r'$(p/\mathrm{QR}-1)/(D_\mathrm{exp}-1)$  (normalized)'
        y_log      = False
        suptitle   = ('Residual SSM (free) — input sanity: '
                      'NORMALIZED residuals (cycle amplitude removed)')
        ref_line_y = 0.0
        ref_label  = 'baseline 0'
    else:
        signal_all = data['signal_all']    # p/QR  (log-Skala sinnvoll)
        y_label    = r'residual $= p / \exp(\mathrm{QuantReg})$  (log)'
        y_log      = True
        suptitle   = 'Residual SSM (free) — input sanity: BTC residuals after QuantReg detrend'
        ref_line_y = 1.0
        ref_label  = f'QuantReg $q={percentile:.2f}$ trend'

    # Zeit-Normalisierung 0..1 für Colorbar
    d_min, d_max = float(days_all[0]), float(days_all[-1])
    t_norm_all = (days_all.astype(float) - d_min) / (d_max - d_min)

    fig, ax = plt.subplots(1, 1, figsize=(13, 5), num=fig_num)
    ax.set_facecolor('#1A1A1A')

    # Hauptlinie als LineCollection mit Coolwarm-Cmap (Farbe = Jahr)
    segs = _make_segments(days_all, signal_all)
    lc = LineCollection(segs, cmap='coolwarm', linewidth=1.0, alpha=0.9)
    lc.set_array(t_norm_all[:-1])
    ax.add_collection(lc)

    if y_log:
        ax.set_yscale('log')
        ax.set_ylim(signal_all.min() * 0.85, signal_all.max() * 1.15)
    else:
        ax.set_yscale('linear')
        s_lo, s_hi = float(np.nanmin(signal_all)), float(np.nanmax(signal_all))
        margin = 0.05 * (s_hi - s_lo)
        ax.set_ylim(s_lo - margin, s_hi + margin)
    ax.set_xlim(0, d_max * 1.02)
    ax.axhline(y=ref_line_y, color='#808080', linestyle='--', alpha=0.7,
               linewidth=1.2, label=ref_label)

    # start_idx Marker
    ax.axvline(days_emb_start, color='#44FF88', linestyle=':', linewidth=1.2,
               alpha=0.7, label=f'start_idx = {days_emb_start}')

    # Schattierung des verarbeiteten Visualisierungs-Range
    if shade_range is not None:
        rlo, rhi = int(shade_range[0]), int(shade_range[1])
        ax.axvspan(rlo, rhi, color='#FFAA33', alpha=0.13,
                   label=f'processed range [{rlo}..{rhi}]')

    # Halving-Marker
    for i, hday in enumerate(halving_days):
        hidx = int(np.argmin(np.abs(days_all - hday)))
        ax.scatter(days_all[hidx], signal_all[hidx], color='lime', s=70,
                   zorder=10, edgecolors='black', linewidth=0.6,
                   label='Halving' if i == 0 else '_nolegend_')
        ax.annotate(str(i + 1), (days_all[hidx], signal_all[hidx]),
                    textcoords='offset points', xytext=(6, 6),
                    color='lime', fontsize=10, fontweight='bold')

    # Cycle-Tops
    for cd, lbl in zip(cycle_top_days, cycle_top_labels):
        cidx = int(np.argmin(np.abs(days_all - cd)))
        ax.scatter(days_all[cidx], signal_all[cidx], color='red', s=55,
                   zorder=11, alpha=0.85)
        ax.annotate(lbl, (days_all[cidx], signal_all[cidx]),
                    textcoords='offset points', xytext=(6, 6),
                    color='white', fontsize=9, fontweight='bold')

    # Latest
    ax.scatter(days_all[-1], signal_all[-1], color='white', s=50,
               zorder=12, edgecolors='black', linewidth=0.8, label='Latest')

    ax.set_xlabel(r'day index', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=9, facecolor='#1A1A1A',
              edgecolor='#808080', labelcolor='#E0E0E0')

    # Colorbar (Ticks = Jahre, kein Label)
    cbar = fig.colorbar(lc, ax=ax, pad=0.01, fraction=0.046)
    n_ticks = 6
    tick_pos = np.linspace(0, 1, n_ticks)
    tick_dates = []
    for t in tick_pos:
        idx = int(np.argmin(np.abs(t_norm_all - t)))
        tick_dates.append(dates_all[idx].strftime('%Y'))
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels(tick_dates)

    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(suptitle,
                     color='#CCCCCC', fontsize=13, y=0.985,
                     fontname='Comfortaa', fontweight='bold')
    plt.subplots_adjust(top=0.92, left=0.07, right=0.97, bottom=0.10)
    return fig
