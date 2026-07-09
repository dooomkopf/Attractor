#!/usr/bin/env python3
"""
attractor_n_ens.py — Ensemble-n(t) Attraktor.

Signal:  n_ens(t) = mean über ws ∈ [90,100,...,180]:
           n(t, ws) = log(p[t+ws/2] / p[t-ws/2]) / log(day[t+ws/2] / day[t-ws/2])
         Am Stück, kein Zyklen-Split, kein Smoothing.

Gleiche Struktur wie attractor_n.py: TAU=30, M=50, PCA → 3D + Slider.

Module:
    attractor_n_ens/attractor_n_ens_utils.py    — I/O, Segmente
    attractor_n_ens/attractor_n_ens_compute.py  — Ensemble-n, PCA, p-Zyklen
    attractor_n_ens/attractor_n_ens_plot1.py    — Figure 1: Ensemble-n + 3D
    attractor_n_ens/attractor_n_ens_plot2.py    — Figure 2: Polar + PC3, Figure 4
    attractor_n_ens/attractor_n_ens_controls.py — Buttons, Slider
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from attractor_n_ens.attractor_n_ens_utils   import read_btc_data
from attractor_n_ens.attractor_n_ens_compute import compute_all
from attractor_n_ens.attractor_n_ens_plot1   import plot_fig1
from attractor_n_ens.attractor_n_ens_plot2   import plot_fig2, plot_fig3

# ── PARAMETER ─────────────────────────────────────────────────────────────────
END_CUT      = 0            # Tage am Ende abschneiden (0 = kein Cut)
START_IDX    = 1164
TAU          = 30
M            = 50
SMOOTH_SIGMA = 60
LABEL_WINDOW = 30
SHOW_FIG3    = True
CMAP         = plt.cm.coolwarm
WINDOW_SIZES = list(range(90, 181, 10))   # [90,100,...,180]
PHASE_OFFSET = 0.0              # Phasen-Offset in Rad
HALVINGS     = [datetime(2012, 11, 28), datetime(2016, 7, 9),
                datetime(2020, 5, 11),  datetime(2024, 4, 20)]
CYCLE_TOPS   = [
    (datetime(2013, 12,  4), "T1", '#0000FF'),
    (datetime(2017, 12, 16), "T2", '#90EE90'),
    (datetime(2021, 11,  9), "T3", '#FF69B4'),
    (datetime(2025, 10,  7), "T4", 'orange'),
]
CYCLE_BOTTOMS = [
    (datetime(2015,  1, 14), "B1", '#0000FF'),
    (datetime(2018, 12, 14), "B2", '#90EE90'),
    (datetime(2022, 11, 21), "B3", '#FF69B4'),
    (datetime(2026,  2,  6), "B4", 'orange'),   # vorläufig
]
# ──────────────────────────────────────────────────────────────────────────────

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


def main():
    params = dict(
        END_CUT=END_CUT, START_IDX=START_IDX, TAU=TAU, M=M,
        SMOOTH_SIGMA=SMOOTH_SIGMA, LABEL_WINDOW=LABEL_WINDOW,
        CMAP=CMAP, WINDOW_SIZES=WINDOW_SIZES, PHASE_OFFSET=PHASE_OFFSET,
        HALVINGS=HALVINGS, CYCLE_TOPS=CYCLE_TOPS, CYCLE_BOTTOMS=CYCLE_BOTTOMS,
        SHOW_FIG3=SHOW_FIG3,
    )

    days_all, prices_all, dates_all = read_btc_data(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ziel.csv'))
    data = compute_all(days_all, prices_all, dates_all, params)

    plot_fig1(data, params)
    plot_fig2(data, params)
    if SHOW_FIG3:
        plot_fig3(data, params)

    plt.show()


if __name__ == '__main__':
    main()
