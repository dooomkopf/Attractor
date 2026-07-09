#!/usr/bin/env python3
"""Verteilung der log-Residuen, getrennt nach Halving-Phase (P1/P2/P3).

Phasen-Definition analog delta-n-dist-lin.py / delta-n-dist-log.py:
  P1 (Hype):        100  <= days_since_halving <  550
  P2 (Suppression): 550  <= days_since_halving <  950
  P3 (Relaxation):  days_since_halving < 100  oder  >= 950

Ein Tag = ein Sample. Nur Darstellung, keine Fits.
"""

import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

ATTRACTOR_DIR = '/home/hz/Data/Attractor'
if ATTRACTOR_DIR not in sys.path:
    sys.path.insert(0, ATTRACTOR_DIR)
from ssmlearn_res import read_btc_data  # noqa: E402

plt.style.use(os.path.join(ATTRACTOR_DIR, 'hz.mplstyle'))

PERCENTILE = 0.01
START_IDX = 1164
FILENAME = os.path.join(ATTRACTOR_DIR, 'ziel.csv')

HALVINGS = [1425, 2744, 4146, 5586, 7044]

CYCLES = [("'13", 1425, 2744), ("'17", 2744, 4146),
          ("'21", 4146, 5586), ("'25", 5586, 7044)]

PHASE_COLORS = {'P1': 'green', 'P2': 'red', 'P3': 'blue'}
PHASE_TITLE = {
    'P1': r'P1 (Hype)        $100 \leq d < 550$',
    'P2': r'P2 (Suppression) $550 \leq d < 950$',
    'P3': r'P3 (Relaxation)  $d < 100 \;\cup\; d \geq 950$',
}


def get_phase(day):
    halving_day = None
    for h in HALVINGS:
        if day >= h:
            halving_day = h
        else:
            break
    if halving_day is None:
        return None
    d = day - halving_day
    if 100 <= d < 550:
        return 'P1'
    if 550 <= d < 950:
        return 'P2'
    return 'P3'


def main():
    days, prices, _ = read_btc_data(FILENAME)
    log_d = np.log(days); log_p = np.log(prices)
    X = sm.add_constant(log_d)
    fit = QuantReg(log_p, X).fit(q=PERCENTILE).predict(X)
    res_all = np.log(prices / np.exp(fit))

    mask = days >= START_IDX
    r = res_all[mask]
    d = days[mask]

    phases = np.array([get_phase(int(t)) for t in d])
    groups = {p: r[phases == p] for p in ('P1', 'P2', 'P3')}

    r_min, r_max = float(r.min()), float(r.max())
    bins = np.linspace(r_min, r_max, 80)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    for ax, phase in zip(axes, ('P1', 'P2', 'P3')):
        rp = groups[phase]
        ax.hist(rp, bins=bins, color=PHASE_COLORS[phase],
                edgecolor='k', linewidth=0.3, alpha=0.85)
        ax.set_yscale('log')
        ax.set_xlabel('log-Residuum r')
        ax.set_ylabel('count (log)')
        ax.set_title(f'{PHASE_TITLE[phase]}   N={len(rp)}',
                     color='#CCCCCC', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        print(f'{phase}: N={len(rp):4d}  mean={rp.mean():+.4f}  '
              f'median={np.median(rp):+.4f}  std={rp.std():.4f}')

    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(
            f"BTC log-Residuen nach Halving-Phase   "
            f"(N_total={len(r)}, ab start_idx={START_IDX})",
            color='#CCCCCC', fontsize=13, y=0.985,
            fontname='Comfortaa', fontweight='bold')
    fig.subplots_adjust(top=0.90, wspace=0.18, left=0.05, right=0.98)

    fig2, axes2 = plt.subplots(len(CYCLES), 3, figsize=(16, 4 * len(CYCLES)),
                                sharex=True, sharey='row')
    for row, (cyc_label, h_start, h_end) in enumerate(CYCLES):
        cyc_mask = (d >= h_start) & (d < h_end)
        for col, phase in enumerate(('P1', 'P2', 'P3')):
            ax = axes2[row, col]
            rp = r[cyc_mask & (phases == phase)]
            ax.hist(rp, bins=bins, color=PHASE_COLORS[phase],
                    edgecolor='k', linewidth=0.3, alpha=0.85)
            ax.set_yscale('log')
            if len(rp):
                med_rp = float(np.median(rp))
                ax.plot([med_rp], [0], marker='o', markersize=9,
                        color='#FFD700', markeredgecolor='k', markeredgewidth=0.8,
                        transform=ax.get_xaxis_transform(),
                        clip_on=False, zorder=10)
            ax.grid(True, alpha=0.3, which='both')
            if row == 0:
                ax.set_title(PHASE_TITLE[phase], color='#CCCCCC', fontsize=11)
            if row == len(CYCLES) - 1:
                ax.set_xlabel('log-Residuum r')
            if col == 0:
                ax.set_ylabel(f"Cycle {cyc_label}\n(d {h_start}-{h_end})\ncount",
                              fontsize=10)
            mu = rp.mean() if len(rp) else float('nan')
            md = float(np.median(rp)) if len(rp) else float('nan')
            sd = rp.std() if len(rp) else float('nan')
            ax.text(0.98, 0.97,
                    f"N={len(rp)}\nmean={mu:+.2f}\nmed={md:+.2f}\nstd={sd:.2f}",
                    transform=ax.transAxes, fontsize=9,
                    color='#E0E0E0', va='top', ha='right',
                    bbox=dict(facecolor='#1A1A1A', edgecolor='#808080', alpha=0.85))
            print(f"{cyc_label} {phase}: N={len(rp):4d}  mean={mu:+.3f}  "
                  f"median={md:+.3f}  std={sd:.3f}")

    with plt.rc_context({'text.usetex': False}):
        fig2.suptitle("BTC log-Residuen   Zyklus x Phase",
                      color='#CCCCCC', fontsize=13, y=0.985,
                      fontname='Comfortaa', fontweight='bold')
    fig2.subplots_adjust(top=0.94, wspace=0.18, hspace=0.22, left=0.07, right=0.98)

    plt.show()


if __name__ == '__main__':
    main()
