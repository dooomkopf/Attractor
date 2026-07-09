#!/usr/bin/env python3
"""14: 3D phase-space (PC1, PC2, PC3) of BTC ensemble-n SSM.

Trajektorie der reduzierten SSM-Koordinaten in 3D mit Time-Coloring und
Halving-Events als markierte 3D-Punkte. Plus 2D-Projektionen rechts:
PC1-PC2, PC1-PC3, PC2-PC3.

CLI:
  ./14_cli_phase3d.py --time_mode linear --ssm_dim 3 --windows 180
  ./14_cli_phase3d.py --time_mode log    --ssm_dim 3 --windows 180
"""

import argparse
import os
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

HZ_MPLSTYLE = '/home/hz/Data/hz.mplstyle'
if os.path.exists(HZ_MPLSTYLE):
    plt.style.use(HZ_MPLSTYLE)
    mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']

from analyze_n_ens.constants import (  # noqa: E402
    DEFAULT_FILENAME, DEFAULT_M, DEFAULT_YEARS, START_IDX,
)
from analyze_n_ens.data import build_n_ens_context  # noqa: E402

HALVING_DAYS = [1425, 2744, 4146, 5586]
HALVING_LABELS = ['H1', 'H2', 'H3', 'H4']
COLOR_HALV = 'grey'  # per attractor_n.py convention
COLOR_TRAJ = '#666666'  # connecting line, dimmer than halvings
LEGEND_KW = dict(loc='upper left', fontsize=10, facecolor='#1A1A1A',
                 edgecolor='#808080', labelcolor='#E0E0E0', framealpha=0.85)


def main():
    ap = argparse.ArgumentParser(
        description='3D phase-space (PC1, PC2, PC3) of BTC Ensemble-n SSM')
    ap.add_argument('--filename', default=DEFAULT_FILENAME)
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--start_idx', type=int, default=START_IDX)
    ap.add_argument('--ssm_dim', type=int, default=3,
                    help='Min 3 (PC1+PC2+PC3 needed). Higher dim still uses first 3 PCs.')
    ap.add_argument('--time_mode', choices=['linear', 'log'], default='linear')
    ap.add_argument('--windows', type=int, nargs='+', default=[180],
                    help='Window sizes (days) for n_ens; default = [180].')
    ap.add_argument('--elev', type=float, default=22.0,
                    help='3D elevation angle (deg)')
    ap.add_argument('--azim', type=float, default=-60.0,
                    help='3D azimuth angle (deg)')
    ap.add_argument('--no-show', action='store_true')
    ap.add_argument('--save', type=str, default=None)
    args = ap.parse_args()

    payload = build_n_ens_context(args.filename, args.M, args.years, args.start_idx,
                                   time_mode=args.time_mode, window_sizes=args.windows)
    ctx = payload['ctx']
    pc1 = ctx['pc'][:, 0]
    pc2 = ctx['pc'][:, 1]
    pc3 = ctx['pc'][:, 2]
    days = ctx['days_vecs'].astype(float)
    var = ctx.get('var')

    halv_idx = []
    halv_labels = []
    for h_day, h_lab in zip(HALVING_DAYS, HALVING_LABELS):
        if days[0] <= h_day <= days[-1]:
            halv_idx.append(int(np.argmin(np.abs(days - h_day))))
            halv_labels.append(h_lab)

    color_t = np.log10(days)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.6, 1.6, 1.0],
                          height_ratios=[1.0, 1.0, 1.0])
    ax_3d = fig.add_subplot(gs[:, 0:2], projection='3d')
    ax_p12 = fig.add_subplot(gs[0, 2])
    ax_p13 = fig.add_subplot(gs[1, 2])
    ax_p23 = fig.add_subplot(gs[2, 2])

    # Remove grey panes (per attractor_n.py convention)
    for pane in [ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#444444')
    ax_3d.tick_params(colors='#CCCCCC')

    ax_3d.plot(pc1, pc2, pc3, lw=0.5, color=COLOR_TRAJ, alpha=0.5, zorder=1)
    sc3 = ax_3d.scatter(pc1, pc2, pc3, c=color_t, cmap='plasma', s=2,
                        alpha=0.7, zorder=2)
    if halv_idx:
        ax_3d.scatter(pc1[halv_idx], pc2[halv_idx], pc3[halv_idx],
                      s=80, c=COLOR_HALV, edgecolors='white', linewidth=1.0,
                      zorder=10, label='Halving')
        for idx, lab in zip(halv_idx, halv_labels):
            ax_3d.text(pc1[idx], pc2[idx], pc3[idx], '  ' + lab,
                       color='white', fontsize=11, fontweight='bold')
    if var is not None:
        var_str = (rf'PC1  ({var[0]*100:.1f}\%)',
                   rf'PC2  ({var[1]*100:.1f}\%)',
                   rf'PC3  ({var[2]*100:.1f}\%)')
    else:
        var_str = ('PC1', 'PC2', 'PC3')
    ax_3d.set_xlabel(var_str[0], color='#CCCCCC', labelpad=8)
    ax_3d.set_ylabel(var_str[1], color='#CCCCCC', labelpad=8)
    ax_3d.set_zlabel(var_str[2], color='#CCCCCC', labelpad=8)
    ax_3d.set_title(f'3D phase-space ({args.time_mode}-clock SSM)', fontsize=13)
    ax_3d.view_init(elev=args.elev, azim=args.azim)
    ax_3d.legend(**LEGEND_KW)

    cb = plt.colorbar(sc3, ax=ax_3d, fraction=0.03, pad=0.08)
    cb.set_label(r'$\log_{10}(\mathrm{day})$', color='#CCCCCC', fontsize=12)
    cb.ax.tick_params(labelsize=11)

    def _proj(ax, x, y, x_label, y_label):
        ax.plot(x, y, lw=0.4, color=COLOR_TRAJ, alpha=0.4)
        ax.scatter(x, y, c=color_t, cmap='plasma', s=1.5, alpha=0.7)
        if halv_idx:
            ax.scatter(x[halv_idx], y[halv_idx], s=60, c=COLOR_HALV,
                       edgecolors='white', linewidth=1.0, zorder=10)
            for idx, lab in zip(halv_idx, halv_labels):
                ax.annotate(lab, (x[idx], y[idx]), xytext=(5, 5),
                            textcoords='offset points', color='white',
                            fontsize=9, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)

    _proj(ax_p12, pc1, pc2, 'PC1', 'PC2')
    ax_p12.set_title('PC1–PC2', fontsize=11)
    _proj(ax_p13, pc1, pc3, 'PC1', 'PC3')
    ax_p13.set_title('PC1–PC3', fontsize=11)
    _proj(ax_p23, pc2, pc3, 'PC2', 'PC3')
    ax_p23.set_title('PC2–PC3', fontsize=11)

    win_str = '-'.join(str(w) for w in args.windows)
    with plt.rc_context({'text.usetex': False}):
        plt.suptitle(f'BTC Ensemble n: 3D phase-space  '
                     f'[ssm_dim={args.ssm_dim}, time={args.time_mode}, windows={win_str}]',
                     color='#CCCCCC', fontsize=13, y=0.985,
                     fontname='Comfortaa', fontweight='bold')

    plt.subplots_adjust(top=0.93, bottom=0.06, left=0.04, right=0.97,
                        hspace=0.40, wspace=0.30)

    if args.save is None:
        args.save = os.path.join(
            HERE,
            f'14_phase3d_dim{args.ssm_dim}_{args.time_mode}_w{win_str}.png',
        )
    fig.savefig(args.save, dpi=200, facecolor='#0a0a0a')
    print(f'plot saved to {args.save}')
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
