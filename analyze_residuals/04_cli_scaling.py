#!/usr/bin/env python3
"""CLI: amplitude scaling test — does |z_sub| scale as |z_main|^2?
Analog to Wang 04_cli_scaling — with scatter plot + fit."""

import argparse
import os
import sys
import logging
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
warnings.filterwarnings("ignore")
logging.getLogger('ridge_regression').setLevel(logging.WARNING)
logging.getLogger('SSMLearn').setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if '/home/hz/Data/Attractor' not in sys.path:
    sys.path.insert(0, '/home/hz/Data/Attractor')

from analyze_residuals.constants import DEFAULT_FILENAME, DEFAULT_M, DEFAULT_YEARS, START_IDX, DAYS_PER_YEAR
from analyze_residuals.data import build_residual_context
from analyze_residuals.common import analysis_time_vector, identify_modes, smooth_real_series
from ssmlearn_res import fit_ssm

try:
    plt.style.use('/home/hz/Data/Attractor/hz.mplstyle')
except Exception:
    plt.rcParams.update({
        'figure.facecolor': '#0a0a0a', 'axes.facecolor': '#1a1a1a',
        'text.color': '#CCCCCC', 'axes.labelcolor': '#CCCCCC',
        'xtick.color': '#CCCCCC', 'ytick.color': '#CCCCCC',
    })


def main():
    ap = argparse.ArgumentParser(description='Amplitude scaling test: |z_sub| vs |z_main|^2')
    ap.add_argument('--filename', type=str, default=DEFAULT_FILENAME)
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--start_idx', type=int, default=START_IDX)
    ap.add_argument('--ssm_dim', type=int, default=2)
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2)
    ap.add_argument('--time_mode', choices=['linear', 'log'], default='linear')
    ap.add_argument('--clip_pct', type=float, default=20.0)
    ap.add_argument('--smooth_days', type=float, default=180.0)
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    payload = build_residual_context(args.filename, args.M, args.years, args.start_idx, time_mode=args.time_mode)
    ctx = payload['ctx']
    time_vec = analysis_time_vector(ctx)
    res = fit_ssm(ctx, args.ssm_dim, poly_degree=args.poly_degree, compute_prediction=False, time_vec=time_vec)

    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :args.ssm_dim]
    eigvals, eigvecs = np.linalg.eig(linear_part)
    try:
        idx_main, idx_sub = identify_modes(eigvals)
    except RuntimeError as exc:
        print(
            f"{exc}. Current default (ssm_dim={args.ssm_dim}, poly={args.poly_degree}) "
            "is a single-mode control. For harmonic scaling diagnostics use "
            "--ssm_dim 4 --poly 1."
        )
        return
    period_unit = "y" if args.time_mode == "linear" else "fit"
    period_scale = DAYS_PER_YEAR if args.time_mode == "linear" else 1.0
    T_main = (2.0 * np.pi / abs(eigvals[idx_main].imag)) / period_scale
    T_sub = (2.0 * np.pi / abs(eigvals[idx_sub].imag)) / period_scale

    pc = ctx['pc'][:, :args.ssm_dim].T
    V_inv = np.linalg.inv(eigvecs)
    Z = V_inv @ pc
    amp_main, amp_sub = np.abs(Z[idx_main, :]), np.abs(Z[idx_sub, :])

    threshold = np.percentile(amp_main, args.clip_pct)
    mask = amp_main >= threshold
    am, asub = amp_main[mask], amp_sub[mask]
    am2 = am ** 2
    slope = float(np.sum(am2 * asub) / (np.sum(am2 ** 2) + 1e-30))
    pred = slope * am2
    corr = float(np.corrcoef(am2, asub)[0, 1])
    ss_res = np.sum((asub - pred) ** 2)
    r2_origin = 1.0 - ss_res / (np.sum(asub ** 2) + 1e-30)
    cv_main = float(np.std(am) / (np.mean(am) + 1e-30))

    fit_time = np.asarray(time_vec, dtype=float)
    if args.time_mode == 'linear':
        t_plot = (fit_time - fit_time[0]) / DAYS_PER_YEAR
        x_unit = 'years'
    else:
        t_plot = fit_time - fit_time[0]
        x_unit = 'fit units'
    dt = float(np.median(np.diff(fit_time)))
    sw = max(1, int(round(args.smooth_days / max(dt, 1e-12))))

    print("=" * 84)
    print("BTC RESIDUAL SCALING READOUT")
    print("=" * 84)
    print("SETUP")
    print(f"  time_mode       : {args.time_mode}")
    print()
    print("MODES")
    print(f"  main            : T={T_main:.3f}{period_unit}  idx={idx_main}")
    print(f"  sub             : T={T_sub:.3f}{period_unit}   idx={idx_sub}")
    print()
    print("AMPLITUDE SCALING (|z_sub| ~ c * |z_main|^2)")
    print(f"  clip threshold  : {threshold:.3e}   [lower {args.clip_pct:.0f}% removed]")
    print(f"  kept samples    : {int(np.sum(mask))}")
    print(f"  main mean/std/cv: {np.mean(am):.3e} / {np.std(am):.3e} / {cv_main:.3f}")
    print(f"  sub  mean/std/cv: {np.mean(asub):.3e} / {np.std(asub):.3e} / {np.std(asub)/(np.mean(asub)+1e-30):.3f}")
    print(f"  slope (c)       : {slope:.6e}")
    print(f"  corr(|z_m|^2, |z_s|): {corr:+.6f}")
    print(f"  R^2 through origin  : {r2_origin:.6f}")
    print(f"  identifiable    : {'yes' if cv_main >= 0.05 else 'no'}   [main CV >= 5%]")
    print("=" * 84)

    if args.show:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

        ax = axes[0]
        ax.plot(t_plot, smooth_real_series(amp_main, sw), lw=0.8, color='tab:blue', label='main envelope')
        ax.plot(t_plot, smooth_real_series(amp_sub, sw), lw=0.8, color='tab:red', label='harm envelope')
        ax.set_xlabel(x_unit)
        ax.set_ylabel('amplitude')
        ax.set_title('Mode envelopes (smoothed)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

        ax = axes[1]
        ax.scatter(am2, asub, s=0.5, alpha=0.3, c='lime')
        am2_line = np.linspace(0, am2.max(), 100)
        ax.plot(am2_line, slope * am2_line, 'w--', lw=1.0, label=f'c={slope:.4f}')
        ax.set_xlabel('$|z_{main}|^2$')
        ax.set_ylabel('$|z_{sub}|$')
        ax.set_title(f'Scaling: corr={corr:+.3f}  $R^2$={r2_origin:.3f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

        ax = axes[2]
        residual = asub - pred
        ax.scatter(am, residual, s=0.5, alpha=0.3, c='orange')
        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('$|z_{main}|$')
        ax.set_ylabel('residual')
        ax.set_title('Scaling residuals')
        ax.grid(True, alpha=0.25)

        fig.suptitle(f'BTC Amplitude Scaling  ssm_dim={args.ssm_dim}  poly={args.poly_degree}  time={args.time_mode}', fontsize=11)
        plt.show()


if __name__ == '__main__':
    main()
