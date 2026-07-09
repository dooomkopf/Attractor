#!/usr/bin/env python3
"""CLI: SSMLearn slave test on BTC residuals with plots."""

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_residuals.constants import DEFAULT_FILENAME, DEFAULT_M, DEFAULT_YEARS, START_IDX
from analyze_residuals.common import analysis_time_vector
from analyze_residuals.data import build_residual_context
from analyze_residuals.ssm_learn import run_slave_test, format_slave_test

try:
    plt.style.use('/home/hz/Data/Attractor/hz.mplstyle')
except Exception:
    plt.rcParams.update({
        'figure.facecolor': '#0a0a0a', 'axes.facecolor': '#1a1a1a',
        'text.color': '#CCCCCC', 'axes.labelcolor': '#CCCCCC',
        'xtick.color': '#CCCCCC', 'ytick.color': '#CCCCCC',
    })


def _plot_results(ctx, result):
    pc = ctx['pc']
    pc_pred = result['pc_pred']
    ssm_dim = result['ssm_dim']
    slave_results = result['slave_results']
    fit_time = np.asarray(ctx.get('fit_time_vecs', ctx['days_vecs']), dtype=float)
    if result['time_mode'] == 'linear':
        t_color = (fit_time - fit_time[0]) / 365.25
    else:
        t_color = fit_time - fit_time[0]
    Vt = ctx['Vt']
    pred_obs = result['res']['pred_obs']
    var = ctx['var']

    n_slaves = len(slave_results)
    n_top = 2 + min(2, n_slaves)
    n_bot = max(0, n_slaves - 2)
    n_cols = max(n_top, n_bot) if n_bot > 0 else n_top
    fig = plt.figure(figsize=(4.5 * n_cols, 9), constrained_layout=True)
    n_rows = 2 if n_bot > 0 else 1

    ax = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    D_c = ctx['D_c']
    dec = pred_obs.T
    pc3_true = D_c @ Vt[:3].T
    pc3_dec = dec @ Vt[:3].T
    ax.plot(pc3_true[:, 0], pc3_true[:, 1], pc3_true[:, 2],
            lw=0.3, alpha=0.5, color='cyan', label='trajectory')
    ax.plot(pc3_dec[:, 0], pc3_dec[:, 1], pc3_dec[:, 2],
            lw=0.3, alpha=0.5, color='red', label='decoder')
    ax.set_xlabel(f'PC1 ({var[0]*100:.0f}\\%%)')
    ax.set_ylabel(f'PC2 ({var[1]*100:.0f}\\%%)')
    ax.set_zlabel(f'PC3 ({var[2]*100:.0f}\\%%)')
    cum3 = sum(var[:3]) * 100
    ax.set_title(f'PC1-3 ({cum3:.0f}\\%%): traj vs decoder')
    ax.legend(fontsize=7, loc='upper left')

    for i, sr in enumerate(slave_results):
        k = sr['pc']
        if i < 2:
            ax = fig.add_subplot(n_rows, n_cols, 2 + i)
        else:
            ax = fig.add_subplot(n_rows, n_cols, n_cols + 1 + (i - 2))
        ax.scatter(pc[:, k], pc_pred[:, k], s=0.5, alpha=0.3, c='lime')
        lims = [min(pc[:, k].min(), pc_pred[:, k].min()),
                max(pc[:, k].max(), pc_pred[:, k].max())]
        ax.plot(lims, lims, 'w--', lw=0.8, alpha=0.5)
        ax.set_xlabel(f'PC{k+1} true')
        ax.set_ylabel(f'PC{k+1} pred')
        ax.set_title(f'PC{k+1}: $R^2$={sr["r2"]:.3f} {sr["verdict"]}')
        ax.set_aspect('equal')

    ax = fig.add_subplot(n_rows, n_cols, n_top)
    ax.scatter(pc[:, 0], pc[:, 1], s=0.3, alpha=0.3, c=t_color, cmap='coolwarm')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Phase portrait (color=time)')

    fig.suptitle(
        f'BTC Residuals SSMLearn  ssm_dim={ssm_dim}  poly={result["poly_degree"]}  time={result["time_mode"]}',
        fontsize=11,
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description='SSMLearn slave test on BTC residuals')
    ap.add_argument('--filename', type=str, default=DEFAULT_FILENAME)
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--start_idx', type=int, default=START_IDX)
    ap.add_argument('--ssm_dim', type=int, default=2)
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2)
    ap.add_argument('--time_mode', '--time-mode', dest='time_mode',
                    choices=['linear', 'log'], default='linear')
    ap.add_argument('--max_slave_pc', type=int, default=8)
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    payload = build_residual_context(args.filename, args.M, args.years, args.start_idx, time_mode=args.time_mode)
    ctx = payload['ctx']
    time_vec = analysis_time_vector(ctx)

    result = run_slave_test(ctx, args.ssm_dim, args.poly_degree, args.max_slave_pc, time_vec=time_vec, time_mode=args.time_mode)

    print("=" * 84)
    for line in format_slave_test(result, ctx):
        print(line)
    print("=" * 84)

    if args.show:
        fig = _plot_results(ctx, result)
        plt.show()


if __name__ == '__main__':
    main()
