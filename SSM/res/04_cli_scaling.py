#!/usr/bin/env python3
"""04: Amplitude-scaling diagnostics on LPPL simulation harmonics."""

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


def _oscillatory_pair_indices(eigvals):
    return sorted(
        [k for k, ev in enumerate(eigvals) if ev.imag > 1e-10],
        key=lambda k: abs(eigvals[k].imag),
    )


def _scaling_metrics(amp_main, amp_sub, clip_pct):
    threshold = float(np.percentile(amp_main, clip_pct))
    mask = np.asarray(amp_main) >= threshold
    am = np.asarray(amp_main)[mask]
    asub = np.asarray(amp_sub)[mask]
    am2 = am ** 2
    slope = float(np.dot(am2, asub) / (np.dot(am2, am2) + 1e-30))
    pred_masked = slope * am2
    pred_full = slope * (np.asarray(amp_main) ** 2)
    corr = float(np.corrcoef(am2, asub)[0, 1]) if len(am2) >= 2 else float('nan')
    ss_res = float(np.sum((asub - pred_masked) ** 2))
    r2_origin = 1.0 - ss_res / (float(np.sum(asub ** 2)) + 1e-30)
    x_center = am2 - float(np.median(am2))
    y_center = asub - float(np.median(asub))
    corr_centered = float(np.corrcoef(x_center, y_center)[0, 1]) if len(am2) >= 2 else float('nan')
    return {
        'threshold_main': threshold,
        'clip_frac': float(clip_pct) / 100.0,
        'mask': mask,
        'slope_zero': slope,
        'corr': corr,
        'corr_centered': corr_centered,
        'r2_zero': float(r2_origin),
        'cv_main': float(np.std(am) / (np.mean(am) + 1e-30)),
        'cv_sub': float(np.std(asub) / (np.mean(asub) + 1e-30)),
        'main_mean': float(np.mean(amp_main)),
        'main_std': float(np.std(amp_main)),
        'harm_mean': float(np.mean(amp_sub)),
        'harm_std': float(np.std(amp_sub)),
        'y_harm': np.asarray(amp_sub),
        'y_hat_full': pred_full,
        'scaling_identifiable': bool(np.std(am) / (np.mean(am) + 1e-30) >= 0.05),
    }


def _fit_and_extract(ctx, ssm_dim, poly_degree, clip_pct):
    """Fit SSM, extract oscillatory pairs, compute phase-lock + scaling."""
    res = fit_ssm(ctx, ssm_dim, poly_degree=poly_degree, compute_prediction=False)
    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :ssm_dim]
    eigvals, eigvecs = np.linalg.eig(linear_part)

    pos_idx = _oscillatory_pair_indices(eigvals)
    if len(pos_idx) < 2:
        return None, None, eigvals, pos_idx

    idx_main, idx_sub = int(pos_idx[0]), int(pos_idx[1])

    pc_red = ctx['pc'][:, :ssm_dim].T
    Z = np.linalg.inv(eigvecs) @ pc_red
    z_main = Z[idx_main, :]
    z_sub = Z[idx_sub, :]
    amp_main = np.abs(z_main)
    amp_sub = np.abs(z_sub)
    delta_unwrapped = 2.0 * np.unwrap(np.angle(z_main)) - np.unwrap(np.angle(z_sub))
    delta_wrapped = np.mod(delta_unwrapped, 2.0 * np.pi)
    delta_principal = np.angle(np.exp(1j * delta_wrapped))
    mean_complex = np.mean(np.exp(1j * delta_wrapped))

    T_main = (2.0 * np.pi / abs(eigvals[idx_main].imag)) / DAYS_PER_YEAR
    T_sub = (2.0 * np.pi / abs(eigvals[idx_sub].imag)) / DAYS_PER_YEAR

    t_days = ctx['days_vecs'].astype(float)
    t_years = (t_days - t_days[0]) / DAYS_PER_YEAR

    report = {
        'ctx': ctx,
        't_years': t_years,
        'eigvals': eigvals,
        'idx_main': idx_main,
        'idx_sub': idx_sub,
        'T_main': float(T_main),
        'T_sub': float(T_sub),
        'ratio_vs_half': float(T_sub / (T_main / 2.0)),
        'amp_main': amp_main,
        'amp_sub': amp_sub,
        'R': float(np.abs(mean_complex)),
        'median_abs_delta_deg': float(np.median(np.abs(np.degrees(delta_principal)))),
    }

    scaling = _scaling_metrics(amp_main, amp_sub, clip_pct)
    return report, scaling, eigvals, pos_idx


def _print_report(report, scaling, mode_tag):
    t_years = report['t_years']
    print("=" * 88)
    print(f"LPPL SIMULATION SCALING READOUT -- {mode_tag}")
    print("=" * 88)
    print("DATA")
    print(f"  kept vectors      : {report['ctx']['N']}")
    print("  dt                : 1.000000d")
    print(f"  kept span         : {t_years[-1] - t_years[0]:.3f}y")
    print()
    print("MODE CHOICE")
    print(f"  main pair         : idx={report['idx_main']}  T={report['T_main']:.3f}y")
    print(f"  harmonic pair     : idx={report['idx_sub']}  T={report['T_sub']:.3f}y")
    print(f"  T_sub/(T_main/2)  : {report['ratio_vs_half']:.3f}")
    print()
    print("PHASE SANITY")
    print(f"  mean resultant R  : {report['R']:.6f}")
    print(f"  median |delta phi|: {report['median_abs_delta_deg']:.2f} deg")
    print()
    print("AMPLITUDE SCALING")
    print(f"  threshold main    : {scaling['threshold_main']:.3e}   [lower {int(round(100 * scaling['clip_frac']))}% clipped]")
    print(f"  kept samples      : {int(np.sum(scaling['mask']))}")
    print(f"  main mean/std/cv  : {scaling['main_mean']:.3e} / {scaling['main_std']:.3e} / {scaling['cv_main']:.3f}")
    print(f"  harm mean/std/cv  : {scaling['harm_mean']:.3e} / {scaling['harm_std']:.3e} / {scaling['cv_sub']:.3f}")
    print(f"  slope (A2w~c*A^2) : {scaling['slope_zero']:.6e}")
    print(f"  corr(A^2,A2w)     : {scaling['corr']:.6f}")
    print(f"  corr(centered)    : {scaling['corr_centered']:.6f}")
    print(f"  R2 through origin : {scaling['r2_zero']:.6f}")
    print(f"  identifiable      : {'yes' if scaling['scaling_identifiable'] else 'weak'}   [main-envelope CV >= 5%]")
    print("=" * 88)


def _plot_scaling(report, scaling):
    t_years = report['t_years']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(t_years, report['amp_main'], color='tab:blue', lw=1.0, label='main envelope')
    ax.plot(t_years, report['amp_sub'], color='tab:red', lw=1.0, label='harm envelope')
    ax.set_title('Mode envelopes')
    ax.set_xlabel('years')
    ax.set_ylabel('amplitude')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    ax.plot(t_years, scaling['y_harm'], color='tab:red', lw=1.0, label='observed harm envelope')
    ax.plot(t_years, scaling['y_hat_full'], color='black', lw=1.0, label='predicted quadratic fit')
    ax.set_title('Observed vs predicted harmonic envelope')
    ax.set_xlabel('years')
    ax.set_ylabel('amplitude')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[2]
    resid_full = scaling['y_harm'] - scaling['y_hat_full']
    resid_lim = max(float(np.quantile(np.abs(resid_full), 0.995)), 1e-9)
    resid_linthresh = max(0.05 * resid_lim, 1e-6)
    ax.plot(t_years, resid_full, color='tab:gray', lw=1.0)
    ax.axhline(0.0, color='black', lw=0.8)
    ax.set_ylim(-resid_lim, resid_lim)
    ax.set_yscale('symlog', linthresh=resid_linthresh)
    ax.set_title('Residual over time')
    ax.set_xlabel('years')
    ax.set_ylabel('harm minus quadratic fit')
    ax.grid(True, alpha=0.25)
    ax.text(
        0.03, 0.97,
        f"Phase-lock:\nR = {report['R']:.3f}\nmedian |dphi| = {report['median_abs_delta_deg']:.1f} deg",
        transform=ax.transAxes, va='top', ha='left', fontsize=8,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '0.5'},
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description='04: Amplitude-scaling diagnostics on LPPL simulation')
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--t_final', type=float, default=DEFAULT_T_FINAL)
    ap.add_argument('--n_eval', type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--ssm_dim', type=int, default=2, metavar='INT')
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2, metavar='POLY')
    ap.add_argument('--clip_pct', type=float, default=20.0,
                    help='lower main-envelope percentile removed')
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

    report, scaling, eigvals, pos_idx = _fit_and_extract(
        ctx, args.ssm_dim, args.poly_degree, args.clip_pct)
    if report is None:
        print(
            f"Need at least 2 oscillatory pairs (got {len(pos_idx)}). "
            f"Use --ssm_dim 4 --poly 1."
        )
        return

    _print_report(report, scaling, mode_tag)

    if args.show:
        _plot_scaling(report, scaling)
        plt.show()


if __name__ == '__main__':
    main()
