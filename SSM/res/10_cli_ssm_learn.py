#!/usr/bin/env python3
"""10: SSMLearn-style decoder/slave diagnostics on LPPL simulation.

2-row plot layout:
  top row:  [3D traj vs decoder] [PC3 scatter] [PC4 scatter] [phase portrait]
  bot row:  remaining slave PCs (PC5, PC6, ...)
"""

import argparse
import os
import sys
import logging
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
warnings.filterwarnings("ignore")
logging.getLogger('ridge_regression').setLevel(logging.WARNING)
logging.getLogger('SSMLearn').setLevel(logging.WARNING)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ATTRACTOR = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _ATTRACTOR, os.path.join(_ATTRACTOR, 'SSMLearnPy')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt
import numpy as np
from lppl_system import build_params
from SSM_res_lppl_data import (
    build_lppl_context, DEFAULT_M, DEFAULT_YEARS, DEFAULT_T_FINAL,
    DEFAULT_N_EVAL, DAYS_PER_YEAR,
)
from ssmlearn_res import fit_ssm

try:
    plt.style.use(os.path.join(_ATTRACTOR, 'hz.mplstyle'))
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description='10: SSMLearn slave diagnostics on LPPL simulation')
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--t_final', type=float, default=DEFAULT_T_FINAL)
    ap.add_argument('--n_eval', type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--ssm_dim', type=int, default=2, metavar='INT')
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2, metavar='POLY')
    ap.add_argument('--max_slave_pc', type=int, default=8,
                    help='highest PC index tested as slave')
    ap.add_argument('--wang-off', action='store_true',
                    help='quasi-2D: Z_A=0, Z_B=0, Z_MIX=8e-5')
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    p = build_params(wang_off=args.wang_off)
    mode_tag = "QUASI-2D" if args.wang_off else "FULL 3D"

    ctx, tau, pca_res, t, traj = build_lppl_context(
        p, args.M, args.years, args.t_final, args.n_eval)

    res = fit_ssm(ctx, args.ssm_dim, poly_degree=args.poly_degree, compute_prediction=False)
    pred_obs = res['pred_obs']
    pc_pred = pred_obs.T @ pca_res.Vt.T

    D_obs = ctx['D_c'].T
    diff = D_obs - pred_obs
    fit_err = float(np.linalg.norm(diff) / (np.linalg.norm(D_obs) + 1e-30))

    n_pcs = min(args.max_slave_pc, pca_res.pc.shape[1])
    slave_results = []
    for k in range(args.ssm_dim, n_pcs):
        true_k = pca_res.pc[:, k]
        pred_k = pc_pred[:, k]
        ss_res = np.sum((true_k - pred_k) ** 2)
        ss_tot = np.sum((true_k - true_k.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-30)
        verdict = 'SLAVED' if r2 >= 0.70 else ('PARTIAL' if r2 >= 0.30 else 'INDEPENDENT')
        slave_results.append({
            'pc': k, 'r2': float(r2), 'verdict': verdict,
            'rms_true': float(np.sqrt(np.mean(true_k ** 2))),
            'rms_err': float(np.sqrt(np.mean((true_k - pred_k) ** 2))),
        })

    # ── Text report ──────────────────────────────────────────────────────────
    print("=" * 84)
    print(f"LPPL SIMULATION SSMLEARN READOUT -- {mode_tag}")
    print("=" * 84)
    print("DATA")
    print(f"  M                 : {args.M}")
    print(f"  years             : {args.years:.2f}")
    print(f"  tau               : {tau}d")
    print(f"  kept vectors      : {ctx['N']}")
    print()
    print("MODEL CHOICE")
    print(f"  ssm_dim           : {args.ssm_dim}")
    print(f"  poly              : {args.poly_degree}")
    print(f"  fit_err           : {fit_err:.4e}")
    print(f"  cum var           : {np.cumsum(pca_res.var)[args.ssm_dim - 1] * 100:.2f}%")
    print()
    print("SLAVE TEST")
    print()
    for sr in slave_results:
        bar = '#' * int(min(30, max(0, sr['r2']) * 30))
        print(f"  PC{sr['pc'] + 1:2d}: R2={sr['r2']:+.4f}  "
              f"rms_true={sr['rms_true']:.3e}  rms_err={sr['rms_err']:.3e}  "
              f"{sr['verdict']:11s}  {bar}")
    slaved = sum(1 for s in slave_results if s['verdict'] == 'SLAVED')
    partial = sum(1 for s in slave_results if s['verdict'] == 'PARTIAL')
    indep = sum(1 for s in slave_results if s['verdict'] == 'INDEPENDENT')
    print(f"\n  SUMMARY: {slaved} slaved, {partial} partial, {indep} independent")
    print("=" * 84)

    # ── Plots ────────────────────────────────────────────────────────────────
    if not args.show:
        return

    t_days = ctx['days_vecs'].astype(float)
    t_years = (t_days - t_days[0]) / DAYS_PER_YEAR

    # Top row: 4 fixed panels
    #   [0] 3D traj vs decoder
    #   [1] PC3 scatter
    #   [2] PC4 scatter
    #   [3] phase portrait
    # Bottom row: remaining slave PCs (PC5, PC6, ...) = slave_results[2:]

    # Determine which slave results go in top row vs bottom
    # PC3 = slave_results[0] (pc index = ssm_dim), PC4 = slave_results[1] (pc index = ssm_dim+1)
    top_slaves = slave_results[:2]  # PC3, PC4 (may be fewer if max_slave_pc is small)
    bot_slaves = slave_results[2:]  # remaining

    n_top_cols = 4  # 3D + PC3 + PC4 + phase portrait
    n_bot_cols = max(len(bot_slaves), 1)
    n_cols = max(n_top_cols, n_bot_cols)
    n_rows = 2 if bot_slaves else 1

    fig = plt.figure(figsize=(4.5 * n_cols, 4.5 * n_rows), constrained_layout=True)

    # ── Top row, panel 0: 3D trajectory vs decoder ───────────────────────
    ax = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
    pc3_true = ctx['D_c'] @ pca_res.Vt[:3].T
    pc3_dec = pred_obs.T @ pca_res.Vt[:3].T
    ax.plot(pc3_true[:, 0], pc3_true[:, 1], pc3_true[:, 2],
            lw=0.3, alpha=0.5, color='cyan', label='traj')
    ax.plot(pc3_dec[:, 0], pc3_dec[:, 1], pc3_dec[:, 2],
            lw=0.3, alpha=0.5, color='red', label='decoder')
    cum3 = sum(pca_res.var[:3]) * 100
    ax.set_title(f'PC1-3 ({cum3:.0f}%)')
    ax.legend(fontsize=7)

    # ── Top row, panels 1-2: PC3 / PC4 scatter ──────────────────────────
    for i, sr in enumerate(top_slaves):
        k = sr['pc']
        ax = fig.add_subplot(n_rows, n_cols, 2 + i)
        ax.scatter(pca_res.pc[:, k], pc_pred[:, k], s=0.5, alpha=0.3, c='lime')
        lims = [min(pca_res.pc[:, k].min(), pc_pred[:, k].min()),
                max(pca_res.pc[:, k].max(), pc_pred[:, k].max())]
        ax.plot(lims, lims, 'w--', lw=0.8, alpha=0.5)
        ax.set_xlabel(f'$\\mathrm{{PC{k + 1}}}$ true')
        ax.set_ylabel(f'$\\mathrm{{PC{k + 1}}}$ pred')
        ax.set_title(f'PC{k + 1}: R2={sr["r2"]:.3f} {sr["verdict"]}')
        ax.set_aspect('equal')

    # ── Top row, panel 3: phase portrait ─────────────────────────────────
    phase_col = 2 + len(top_slaves)
    ax = fig.add_subplot(n_rows, n_cols, phase_col + 1)
    ax.scatter(pca_res.pc[:, 0], pca_res.pc[:, 1], s=0.3, alpha=0.3,
               c=t_years, cmap='coolwarm')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Phase portrait')

    # ── Bottom row: remaining slave PCs ──────────────────────────────────
    for i, sr in enumerate(bot_slaves):
        k = sr['pc']
        ax = fig.add_subplot(n_rows, n_cols, n_cols + 1 + i)
        ax.scatter(pca_res.pc[:, k], pc_pred[:, k], s=0.5, alpha=0.3, c='lime')
        lims = [min(pca_res.pc[:, k].min(), pc_pred[:, k].min()),
                max(pca_res.pc[:, k].max(), pc_pred[:, k].max())]
        ax.plot(lims, lims, 'w--', lw=0.8, alpha=0.5)
        ax.set_xlabel(f'$\\mathrm{{PC{k + 1}}}$ true')
        ax.set_ylabel(f'$\\mathrm{{PC{k + 1}}}$ pred')
        ax.set_title(f'PC{k + 1}: R2={sr["r2"]:.3f} {sr["verdict"]}')
        ax.set_aspect('equal')

    fig.suptitle(
        f'LPPL SSM/res Slave Test  dim={args.ssm_dim}  poly={args.poly_degree} -- {mode_tag}',
        fontsize=11,
    )
    plt.show()


if __name__ == '__main__':
    main()
