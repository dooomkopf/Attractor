#!/usr/bin/env python3
"""03: Phase-lock diagnostics on LPPL simulation harmonics."""

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


def _fit_and_extract(ctx, ssm_dim, poly_degree):
    """Fit SSM, extract eigenvalues, identify 2 oscillatory pairs, compute phase-lock."""
    res = fit_ssm(ctx, ssm_dim, poly_degree=poly_degree, compute_prediction=False)
    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :ssm_dim]
    eigvals, eigvecs = np.linalg.eig(linear_part)

    pos_idx = _oscillatory_pair_indices(eigvals)
    if len(pos_idx) < 2:
        return None, eigvals, pos_idx

    idx_main, idx_sub = int(pos_idx[0]), int(pos_idx[1])

    pc_red = ctx['pc'][:, :ssm_dim].T
    Z = np.linalg.inv(eigvecs) @ pc_red
    z_main = Z[idx_main, :]
    z_sub = Z[idx_sub, :]
    amp_main = np.abs(z_main)
    amp_sub = np.abs(z_sub)
    phi_main = np.angle(z_main)
    phi_sub = np.angle(z_sub)
    phi_main_u = np.unwrap(phi_main)
    phi_sub_u = np.unwrap(phi_sub)
    delta_unwrapped = 2.0 * phi_main_u - phi_sub_u
    delta_wrapped = np.mod(delta_unwrapped, 2.0 * np.pi)
    delta_principal = np.angle(np.exp(1j * delta_wrapped))
    mean_complex = np.mean(np.exp(1j * delta_wrapped))

    T_main = (2.0 * np.pi / abs(eigvals[idx_main].imag)) / DAYS_PER_YEAR
    T_sub = (2.0 * np.pi / abs(eigvals[idx_sub].imag)) / DAYS_PER_YEAR

    # fit / ode errors
    D_obs = ctx['D_c'].T
    diff = D_obs - res['pred_obs']
    fit_err = float(np.linalg.norm(diff) / (np.linalg.norm(D_obs) + 1e-30))
    dp_dt = np.gradient(ctx['pc'][:, :ssm_dim], ctx['days_vecs'].astype(float), axis=0)
    rhs = res['ssm'].reduced_dynamics.predict(ctx['pc'][:, :ssm_dim])
    edge = max(5, ctx['N'] // 100)
    sl = slice(edge, -edge)
    ode_err = float(np.linalg.norm(dp_dt[sl] - rhs[sl]) / (np.linalg.norm(dp_dt[sl]) + 1e-30))

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
        'phi_main': phi_main_u,
        'phi_sub': phi_sub_u,
        'delta_phase': delta_principal,
        'R': float(np.abs(mean_complex)),
        'mean_angle_deg': float(np.degrees(np.angle(mean_complex))),
        'median_abs_delta_deg': float(np.median(np.abs(np.degrees(delta_principal)))),
        'fit_err': fit_err,
        'ode_err': ode_err,
    }
    return report, eigvals, pos_idx


def _print_report(report, ssm_dim, poly_degree, mode_tag):
    t_years = report['t_years']
    print("=" * 88)
    print(f"LPPL SIMULATION PHASE READOUT -- {mode_tag}")
    print("=" * 88)
    print("DATA")
    print(f"  kept vectors      : {report['ctx']['N']}")
    print("  dt                : 1.000000d")
    print(f"  kept span         : {t_years[-1] - t_years[0]:.3f}y")
    print()
    print("MODEL CHOICE")
    print(f"  ssm_dim           : {ssm_dim}")
    print(f"  poly              : {poly_degree}")
    print(f"  main pair         : idx={report['idx_main']}  T={report['T_main']:.3f}y  Re={report['eigvals'][report['idx_main']].real:+.4e}")
    print(f"  harmonic pair     : idx={report['idx_sub']}  T={report['T_sub']:.3f}y  Re={report['eigvals'][report['idx_sub']].real:+.4e}")
    print(f"  T_sub/(T_main/2)  : {report['ratio_vs_half']:.3f}")
    print()
    print("PHASE LOCK")
    print(f"  mean resultant R  : {report['R']:.6f}   [1=locked, 0=uniform]")
    print(f"  mean delta phi    : {report['mean_angle_deg']:.2f} deg")
    print(f"  median |delta phi|: {report['median_abs_delta_deg']:.2f} deg")
    print()
    print("FIT QUALITY")
    print(f"  fit error (Frob)  : {report['fit_err']:.3e}")
    print(f"  ODE error (Frob)  : {report['ode_err']:.3e}")
    print("=" * 88)


def _plot_phase(report, zoom_cycles):
    t_years = report['t_years']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(t_years, report['amp_main'], color='tab:blue', lw=1.0,
            label=f"main envelope ({report['T_main']:.2f}y)")
    ax.plot(t_years, report['amp_sub'], color='tab:red', lw=1.0,
            label=f"harm envelope ({report['T_sub']:.2f}y)")
    ax.set_title('Mode envelopes')
    ax.set_xlabel('years')
    ax.set_ylabel('amplitude')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[0, 1]
    zoom_span_years = float(zoom_cycles) * float(report['T_main'])
    mask_zoom = t_years <= zoom_span_years
    main_wrapped = np.angle(np.exp(1j * report['phi_main']))
    harm_wrapped = np.angle(np.exp(1j * (report['phi_sub'] / 2.0)))
    ax.plot(t_years[mask_zoom], main_wrapped[mask_zoom], color='tab:blue', lw=0.9, label='main')
    ax.plot(t_years[mask_zoom], harm_wrapped[mask_zoom], color='tab:red', lw=0.9, label='harm/2')
    ax.set_title(f'Phases over first {zoom_cycles:g} main cycles')
    ax.set_xlabel('years')
    ax.set_ylabel('phase [rad]')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[1, 0]
    delta_deg = np.degrees(report['delta_phase'])
    mean_deg = float(report['mean_angle_deg'])
    span = max(15.0, float(np.quantile(np.abs(delta_deg - mean_deg), 0.995)) * 1.05)
    span = min(span, 180.0)
    ax.plot(t_years, delta_deg, color='tab:green', lw=0.9)
    ax.axhline(mean_deg, color='gray', ls='--', lw=0.9)
    ax.set_ylim(mean_deg - span, mean_deg + span)
    ax.set_title(f'$\\Delta\\phi$ over time around mean ({mean_deg:.1f} deg)')
    ax.set_xlabel('years')
    ax.set_ylabel('$\\Delta\\phi$ [deg]')
    ax.grid(True, alpha=0.25)

    axes[1, 1].remove()
    ax = fig.add_subplot(2, 2, 4, projection='polar')
    ax.hist(report['delta_phase'], bins=36, color='tab:green', alpha=0.75)
    ax.set_title('Polar histogram of phase combination')
    return fig


def main():
    ap = argparse.ArgumentParser(description='03: Phase-lock diagnostics on LPPL simulation')
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--t_final', type=float, default=DEFAULT_T_FINAL)
    ap.add_argument('--n_eval', type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--ssm_dim', type=int, default=2, metavar='INT')
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2, metavar='POLY')
    ap.add_argument('--phase_zoom_cycles', type=float, default=10.0)
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

    report, eigvals, pos_idx = _fit_and_extract(ctx, args.ssm_dim, args.poly_degree)
    if report is None:
        print(
            f"Need at least 2 oscillatory pairs (got {len(pos_idx)}). "
            f"Use --ssm_dim 4 --poly 1."
        )
        return

    _print_report(report, args.ssm_dim, args.poly_degree, mode_tag)

    if args.show:
        _plot_phase(report, args.phase_zoom_cycles)
        plt.show()


if __name__ == '__main__':
    main()
