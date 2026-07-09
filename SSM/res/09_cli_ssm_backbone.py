#!/usr/bin/env python3
"""09: Backbone readout from fitted reduced dynamics on LPPL simulation."""

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
from analyze_wang.ssm_spectral import spectral_analysis, choose_E, format_spectral

try:
    plt.style.use(os.path.join(_ATTRACTOR, 'hz.mplstyle'))
except Exception:
    pass


def _backbone_report(res, ctx, ssm_dim, poly_degree):
    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :ssm_dim]
    eigvals, eigvecs = np.linalg.eig(linear_part)

    osc = [ev for ev in eigvals if ev.imag > 1e-6]
    if not osc:
        raise RuntimeError('No oscillatory mode found; backbone not applicable')

    lam = sorted(osc, key=lambda z: abs(z.imag))[0]
    alpha_0 = float(lam.real)
    omega_0 = float(abs(lam.imag))
    T_0 = float((2.0 * np.pi / omega_0) / DAYS_PER_YEAR)

    eigvals_full, eigvecs_full = np.linalg.eig(linear_part)
    idx_main = sorted(
        [k for k, ev in enumerate(eigvals_full) if ev.imag > 1e-6],
        key=lambda k: abs(eigvals_full[k].imag),
    )[0]
    pc_r = ctx['pc'][:, :ssm_dim]
    Z = np.linalg.inv(eigvecs_full) @ pc_r.T
    z_main = Z[idx_main, :]
    rho = np.abs(z_main)

    has_nonlinear = poly_degree >= 2 and ssm_dim == 2
    if has_nonlinear:
        rhs = res['ssm'].reduced_dynamics.predict(pc_r)
        Z_dot = np.linalg.inv(eigvecs_full) @ rhs.T
        inst = Z_dot[idx_main, :] / (z_main + 1e-30)
        inst_omega = np.abs(np.imag(inst))
        inst_alpha = np.real(inst)
    else:
        inst_omega = np.full_like(rho, omega_0)
        inst_alpha = np.full_like(rho, alpha_0)

    return {
        'alpha_0': alpha_0,
        'omega_0': omega_0,
        'T_0': T_0,
        'rho': rho,
        'inst_omega': inst_omega,
        'inst_alpha': inst_alpha,
        'has_nonlinear': has_nonlinear,
    }


def _print_report(ctx, res, report, ssm_dim, poly_degree, mode_tag):
    dt = float(np.median(np.diff(ctx['days_vecs'])))
    t_days = ctx['days_vecs'].astype(float)
    t_years = (t_days - t_days[0]) / DAYS_PER_YEAR

    D_obs = ctx['D_c'].T
    diff = D_obs - res['pred_obs']
    fit_err = float(np.linalg.norm(diff) / (np.linalg.norm(D_obs) + 1e-30))

    dp_dt = np.gradient(ctx['pc'][:, :ssm_dim], ctx['days_vecs'].astype(float), axis=0)
    rhs = res['ssm'].reduced_dynamics.predict(ctx['pc'][:, :ssm_dim])
    edge = max(5, ctx['N'] // 100)
    sl = slice(edge, -edge)
    ode_err = float(np.linalg.norm(dp_dt[sl] - rhs[sl]) / (np.linalg.norm(dp_dt[sl]) + 1e-30))

    print("=" * 88)
    print(f"LPPL SIMULATION BACKBONE READOUT -- {mode_tag}")
    print("=" * 88)
    print("DATA")
    print(f"  kept vectors      : {ctx['N']}")
    print(f"  dt                : {dt:.6f}d")
    print(f"  kept span         : {t_years[-1] - t_years[0]:.3f}y")
    print()
    print("MODEL CHOICE")
    print(f"  ssm_dim           : {ssm_dim}")
    print(f"  poly              : {poly_degree}")
    print(f"  fit error (Frob)  : {fit_err:.3e}")
    print(f"  ODE error (Frob)  : {ode_err:.3e}")
    print()
    print("BACKBONE")
    print(f"  alpha_0           : {report['alpha_0']:+.6e}   {'[stable]' if report['alpha_0'] < 0 else '[unstable]'}")
    print(f"  omega_0           : {report['omega_0']:.6e} rad/d")
    print(f"  T_0               : {report['T_0']:.3f}y")
    print(f"  rho range         : [{report['rho'].min():.3e}, {report['rho'].max():.3e}]")
    print(f"  rho median        : {np.median(report['rho']):.3e}")
    if report['has_nonlinear']:
        print(f"  omega range       : [{report['inst_omega'].min():.4e}, {report['inst_omega'].max():.4e}]")
        print(f"  alpha range       : [{report['inst_alpha'].min():.4e}, {report['inst_alpha'].max():.4e}]")
        print("  backbone mode     : nonlinear")
    else:
        print("  backbone mode     : flat (linear reduced dynamics)")
    print("=" * 88)


def _plot_report(report):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    axes[0].scatter(report['rho'], report['inst_omega'] / (2.0 * np.pi) * DAYS_PER_YEAR, s=0.3, alpha=0.3, c='cyan')
    axes[0].set_xlabel('rho')
    axes[0].set_ylabel('f [1/year]')
    axes[0].set_title('Backbone frequency vs rho')
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(report['rho'], report['inst_alpha'], s=0.3, alpha=0.3, c='orange')
    axes[1].axhline(0.0, color='gray', ls='--', lw=0.8)
    axes[1].set_xlabel('rho')
    axes[1].set_ylabel('alpha')
    axes[1].set_title('Growth rate vs rho')
    axes[1].grid(True, alpha=0.25)
    return fig


def main():
    ap = argparse.ArgumentParser(description='09: Backbone readout (polar reduced dynamics) on LPPL simulation')
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--t_final', type=float, default=DEFAULT_T_FINAL)
    ap.add_argument('--n_eval', type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--ssm_dim', type=int, default=2, metavar='INT', help='reduced model dimension')
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2, metavar='POLY', help='polynomial order')
    ap.add_argument('--wang-off', action='store_true',
                    help='quasi-2D: Z_A=0, Z_B=0, Z_MIX=8e-5')
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true', help='open matplotlib plots (default)')
    show_group.add_argument('--no-show', dest='show', action='store_false', help='skip matplotlib plots')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    p = build_params(wang_off=args.wang_off)
    mode_tag = "QUASI-2D" if args.wang_off else "FULL 3D"

    ctx, tau, pca_res, t, traj = build_lppl_context(
        p, args.M, args.years, args.t_final, args.n_eval)

    res = fit_ssm(ctx, args.ssm_dim, poly_degree=args.poly_degree, compute_prediction=False)
    report = _backbone_report(res, ctx, args.ssm_dim, args.poly_degree)
    _print_report(ctx, res, report, args.ssm_dim, args.poly_degree, mode_tag)

    if args.show:
        _plot_report(report)
        plt.show()


if __name__ == '__main__':
    main()
