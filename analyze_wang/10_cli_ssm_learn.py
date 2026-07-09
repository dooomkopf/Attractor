#!/usr/bin/env python3
"""CLI: data-driven SSM fit on Wang trajectory with plots."""

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_wang.constants import (
    DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D,
    DEFAULT_IC, DEFAULT_TRANSIENT_FRAC,
)
from analyze_wang.simulate import simulate_trajectory
from analyze_wang.ssm_learn import (
    prepare_wang_data, fit_ssm_wang, slave_test, format_learn_results,
)

try:
    plt.style.use('/home/hz/Data/Attractor/hz.mplstyle')
except Exception:
    plt.rcParams.update({
        'figure.facecolor': '#0a0a0a', 'axes.facecolor': '#1a1a1a',
        'text.color': '#CCCCCC', 'axes.labelcolor': '#CCCCCC',
        'xtick.color': '#CCCCCC', 'ytick.color': '#CCCCCC',
    })


def _parse_ic(text):
    parts = [float(x.strip()) for x in text.split(',')]
    if len(parts) != 3:
        raise ValueError('IC must have exactly 3 comma-separated floats')
    return tuple(parts)


def _plot_results(ctx, fit_result, slave_results):
    pc = ctx['pc']
    t = ctx['t'] - ctx['t'][0]
    ssm_dim = fit_result['ssm_dim']
    pred_obs = fit_result['pred_obs']
    Vt = ctx['Vt']
    pc_pred = pred_obs.T @ Vt.T

    fig = plt.figure(figsize=(18, 5), constrained_layout=True)

    # (1) 3D trajectory + decoded
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    traj = ctx['traj']
    dec = pred_obs.T + ctx['traj'].mean(axis=0)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.3, alpha=0.5, color='cyan', label='trajectory')
    ax.plot(dec[:, 0], dec[:, 1], dec[:, 2], lw=0.3, alpha=0.5, color='red', label='SSM decoder')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'3D: trajectory vs decoder (dim={ssm_dim}, poly={fit_result["poly_degree"]})')
    ax.legend(fontsize=7, loc='upper left')

    # (2) Slave test: PC3 true vs predicted
    ax = fig.add_subplot(1, 3, 2)
    if slave_results:
        sr = slave_results[0]
        k = sr['pc_index']
        ax.scatter(pc[:, k], pc_pred[:, k], s=0.5, alpha=0.3, c='lime')
        lims = [min(pc[:, k].min(), pc_pred[:, k].min()),
                max(pc[:, k].max(), pc_pred[:, k].max())]
        ax.plot(lims, lims, 'w--', lw=0.8, alpha=0.5)
        ax.set_xlabel(f'PC{k+1} true')
        ax.set_ylabel(f'PC{k+1} predicted')
        ax.set_title(f'Slave test PC{k+1}: $R^2$={sr["r2"]:.4f}')
        ax.set_aspect('equal')
    else:
        ax.text(0.5, 0.5, f'no slave PCs\n(ssm_dim={ssm_dim} = obs_dim={ctx["obs_dim"]})',
                ha='center', va='center', fontsize=11, transform=ax.transAxes)
        ax.set_title('Slave test: n/a')

    # (3) Reduced coordinates phase portrait
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(pc[:, 0], pc[:, 1], s=0.3, alpha=0.3, c=t, cmap='coolwarm')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Reduced coordinates (PC1 vs PC2)')

    fig.suptitle(f'SSMLearn on Wang (x,y,z)  ssm_dim={ssm_dim}  poly={fit_result["poly_degree"]}',
                 fontsize=11)
    return fig


def main():
    ap = argparse.ArgumentParser(description='Data-driven SSM fit on Wang trajectory')
    ap.add_argument('--a', type=float, default=DEFAULT_A)
    ap.add_argument('--b', type=float, default=DEFAULT_B)
    ap.add_argument('--c', type=float, default=DEFAULT_C)
    ap.add_argument('--d', type=float, default=DEFAULT_D)
    ap.add_argument('--ic', type=str, default=','.join(str(v) for v in DEFAULT_IC))
    ap.add_argument('--t_final', type=float, default=150.0)
    ap.add_argument('--n_eval', type=int, default=75000)
    ap.add_argument('--transient_frac', type=float, default=DEFAULT_TRANSIENT_FRAC)
    ap.add_argument('--ssm_dim', type=int, default=2)
    ap.add_argument('--poly', type=int, default=2, dest='poly_degree')
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    sim = simulate_trajectory(
        a=args.a, b=args.b, c=args.c, d=args.d,
        ic=_parse_ic(args.ic),
        t_final=args.t_final, n_eval=args.n_eval,
        transient_frac=args.transient_frac,
    )
    ctx = prepare_wang_data(sim)
    fit_result = fit_ssm_wang(ctx, args.ssm_dim, args.poly_degree)
    slave_results = slave_test(ctx, fit_result)

    print("=" * 84)
    for line in format_learn_results(ctx, fit_result, slave_results):
        print(line)
    print("=" * 84)

    if args.show:
        fig = _plot_results(ctx, fit_result, slave_results)
        plt.show()


if __name__ == '__main__':
    main()
