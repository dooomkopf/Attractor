#!/usr/bin/env python3
"""CLI: phase coupling test between main and sub mode in BTC residuals.
Analog to Wang 03_cli_phase — 4-panel plot: envelopes, phases, delta-phi, polar hist."""

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
from analyze_residuals.common import analysis_time_vector, identify_modes, smooth_phase_series, smooth_real_series, time_mode_rate_unit
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
    ap = argparse.ArgumentParser(description='Phase coupling test (2*phi_main - phi_sub)')
    ap.add_argument('--filename', type=str, default=DEFAULT_FILENAME)
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--start_idx', type=int, default=START_IDX)
    ap.add_argument('--ssm_dim', type=int, default=2)
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2)
    ap.add_argument('--time_mode', choices=['linear', 'log'], default='linear')
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
            f"{exc}\n"
            f"Current setup (ssm_dim={args.ssm_dim}, poly={args.poly_degree}, "
            f"time_mode={args.time_mode}) yields too few oscillatory pairs.\n"
            f"Phase coupling test needs >=2 complex eigenvalue pairs.\n"
            f"Use:  --ssm_dim 4 --poly 1   (or higher ssm_dim).\n"
            f"For all-pairs phase test see ./03b_cli_phase_all.py"
        )
        return
    lam_main = eigvals[idx_main]
    lam_sub = eigvals[idx_sub]
    period_unit = "y" if args.time_mode == "linear" else "fit"
    period_scale = DAYS_PER_YEAR if args.time_mode == "linear" else 1.0
    T_main = (2.0 * np.pi / abs(lam_main.imag)) / period_scale
    T_sub = (2.0 * np.pi / abs(lam_sub.imag)) / period_scale

    pc = ctx['pc'][:, :args.ssm_dim].T
    V_inv = np.linalg.inv(eigvecs)
    Z = V_inv @ pc
    z_main, z_sub = Z[idx_main, :], Z[idx_sub, :]
    amp_main, amp_sub = np.abs(z_main), np.abs(z_sub)
    phi_main, phi_sub = np.angle(z_main), np.angle(z_sub)
    phi_main_u, phi_sub_u = np.unwrap(phi_main), np.unwrap(phi_sub)

    psi_unwrapped = 2.0 * phi_main_u - phi_sub_u
    psi_wrapped = np.mod(psi_unwrapped, 2.0 * np.pi)
    psi_principal = np.angle(np.exp(1j * psi_wrapped))

    R = float(np.abs(np.mean(np.exp(1j * psi_wrapped))))
    mean_angle_deg = float(np.degrees(np.angle(np.mean(np.exp(1j * psi_wrapped)))))
    median_abs_deg = float(np.median(np.abs(np.degrees(psi_principal))))

    fit_time = np.asarray(time_vec, dtype=float)
    if args.time_mode == 'linear':
        t_plot = (fit_time - fit_time[0]) / DAYS_PER_YEAR
        x_unit = 'years'
    else:
        t_plot = fit_time - fit_time[0]
        x_unit = 'fit units'
    fit_rel = fit_time - fit_time[0]
    A_mat = np.vstack([fit_rel, np.ones_like(fit_rel)]).T
    slope, intercept = np.linalg.lstsq(A_mat, psi_unwrapped, rcond=None)[0]
    expected_drift = 2.0 * lam_main.imag - lam_sub.imag
    drift_unit = time_mode_rate_unit(args.time_mode)

    dt = float(np.median(np.diff(fit_time)))
    sw = max(1, int(round(args.smooth_days / max(dt, 1e-12))))

    print("=" * 84)
    print("BTC RESIDUAL PHASE READOUT")
    print("=" * 84)
    print("SETUP")
    print(f"  time_mode       : {args.time_mode}")
    print()
    print("MODES")
    print(f"  main            : T={T_main:.3f}{period_unit}  Re={lam_main.real:+.4e}  idx={idx_main}")
    print(f"  sub             : T={T_sub:.3f}{period_unit}   Re={lam_sub.real:+.4e}  idx={idx_sub}")
    print(f"  T_sub/(T_main/2): {T_sub/(T_main/2):.3f}   [1.0 = exact 2:1]")
    print()
    print("PHASE LOCK (psi = 2*phi_main - phi_sub)")
    print(f"  mean resultant R  : {R:.6f}   [1=locked, 0=uniform]")
    print(f"  mean delta phi    : {mean_angle_deg:.2f} deg")
    print(f"  median |delta phi|: {median_abs_deg:.2f} deg")
    print()
    print("DRIFT ANALYSIS")
    print(f"  psi drift rate    : {slope:.6e} rad/{drift_unit}")
    print(f"  expected if indep : {expected_drift:.6e} rad/{drift_unit}")
    print(f"  ratio actual/exp  : {slope/expected_drift:.4f}" if abs(expected_drift) > 1e-15 else
          f"  ratio actual/exp  : n/a")
    print("=" * 84)

    if args.show:
        LEGEND_KW = dict(loc='upper left', fontsize=10, facecolor='#1A1A1A',
                         edgecolor='#808080', labelcolor='#E0E0E0', framealpha=0.85)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        ax = axes[0, 0]
        ax.plot(t_plot, smooth_real_series(amp_main, sw), lw=1.4, color='#FFB04A',
                label=f'Master envelope T={T_main:.2f}{period_unit}')
        ax.plot(t_plot, smooth_real_series(amp_sub, sw), lw=1.4, color='#B197FC',
                label=f'Slave (Harmonic) envelope T={T_sub:.2f}{period_unit}')
        ax.set_title('Mode amplitudes (smoothed)', fontsize=11)
        ax.set_xlabel(x_unit, fontsize=12)
        ax.set_ylabel('amplitude', fontsize=12)
        ax.tick_params(labelsize=11)
        ax.legend(**LEGEND_KW)

        ax = axes[0, 1]
        ax.plot(t_plot, smooth_phase_series(phi_main, sw), lw=1.2, color='#FFB04A',
                label='Master phase')
        ax.plot(t_plot, smooth_phase_series(phi_sub / 2.0, sw), lw=1.2, color='#B197FC',
                label='Slave (Harmonic) phase / 2')
        ax.set_title('Wrapped phases (smoothed)', fontsize=11)
        ax.set_xlabel(x_unit, fontsize=12)
        ax.set_ylabel('phase [rad]', fontsize=12)
        ax.tick_params(labelsize=11)
        ax.legend(**LEGEND_KW)

        ax = axes[1, 0]
        delta_deg = np.degrees(psi_principal)
        delta_deg_s = smooth_real_series(delta_deg, sw)
        ax.plot(t_plot, delta_deg_s, lw=1.2, color='#69DB7C')
        ax.axhline(mean_angle_deg, color='#808080', ls='--', lw=0.9)
        ax.set_ylim(mean_angle_deg - 90, mean_angle_deg + 90)
        ax.set_title(rf'$\psi$ over time (R={R:.3f}, mean={mean_angle_deg:.1f}$^\circ$)',
                     fontsize=11)
        ax.set_xlabel(x_unit, fontsize=12)
        ax.set_ylabel(r'$\psi$ [deg]', fontsize=12)
        ax.tick_params(labelsize=11)

        ax = fig.add_subplot(2, 2, 4, projection='polar')
        axes[1, 1].remove()
        ax.hist(psi_wrapped, bins=36, color='#FFB04A', alpha=0.75)
        ax.set_title(r'Polar histogram of $\psi = 2\phi_{\mathrm{main}} - \phi_{\mathrm{sub}}$',
                     fontsize=11)

        with plt.rc_context({'text.usetex': False}):
            fig.suptitle(
                f'BTC Residuals: Phase Lock Test  '
                f'[ssm_dim={args.ssm_dim}, poly={args.poly_degree}, time={args.time_mode}]',
                fontsize=13, color='#CCCCCC', y=0.99,
                fontname='Comfortaa', fontweight='bold')

        plt.subplots_adjust(top=0.93, bottom=0.08, hspace=0.40, wspace=0.30)
        plt.show()


if __name__ == '__main__':
    main()
