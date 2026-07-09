#!/usr/bin/env python3
"""05: Scan ssm_dim x poly on LPPL simulation — build context ONCE, loop over fits."""

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


def _parse_csv_ints(text):
    return [int(part.strip()) for part in text.split(',') if part.strip()]


def _oscillatory_pair_indices(eigvals):
    return sorted(
        [k for k, ev in enumerate(eigvals) if ev.imag > 1e-10],
        key=lambda k: abs(eigvals[k].imag),
    )


def _fmt_year(value):
    return '-' if value is None or np.isnan(value) else f'{value:.3f}y'


def _fmt_num(value, fmt):
    return '-' if value is None or np.isnan(value) else format(value, fmt)


def _fmt_pct(value):
    return '-' if value is None or np.isnan(value) else f'{value:.1f}%'


def _scan_case(ctx, dim, poly):
    """Single (dim, poly) fit on prebuilt ctx. Returns row dict."""
    res = fit_ssm(ctx, dim, poly_degree=poly, compute_prediction=False)
    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :dim]
    eigvals, eigvecs = np.linalg.eig(linear_part)
    pos_idx = _oscillatory_pair_indices(eigvals)

    D_obs = ctx['D_c'].T
    diff = D_obs - res['pred_obs']
    fit_err = float(np.linalg.norm(diff) / (np.linalg.norm(D_obs) + 1e-30))

    dp_dt = np.gradient(ctx['pc'][:, :dim], ctx['days_vecs'].astype(float), axis=0)
    rhs = res['ssm'].reduced_dynamics.predict(ctx['pc'][:, :dim])
    edge = max(5, ctx['N'] // 100)
    sl = slice(edge, -edge)
    ode_err = float(np.linalg.norm(dp_dt[sl] - rhs[sl]) / (np.linalg.norm(dp_dt[sl]) + 1e-30))

    row = {
        'label': f's{dim}p{poly}',
        'dim': dim,
        'poly': poly,
        'pairs': len(pos_idx),
        'fit_err': fit_err,
        'ode_err': ode_err,
        'T_main': np.nan,
        'T_sub': np.nan,
        'detune_pct': np.nan,
        'R': np.nan,
        'median_abs_deg': np.nan,
        'sub_Re': np.nan,
    }

    if len(pos_idx) >= 1:
        idx_main = pos_idx[0]
        row['T_main'] = float((2.0 * np.pi / abs(eigvals[idx_main].imag)) / DAYS_PER_YEAR)

    if len(pos_idx) >= 2:
        idx_main, idx_sub = int(pos_idx[0]), int(pos_idx[1])
        T_main = (2.0 * np.pi / abs(eigvals[idx_main].imag)) / DAYS_PER_YEAR
        T_sub = (2.0 * np.pi / abs(eigvals[idx_sub].imag)) / DAYS_PER_YEAR

        pc_red = ctx['pc'][:, :dim].T
        Z = np.linalg.inv(eigvecs) @ pc_red
        z_main = Z[idx_main, :]
        z_sub = Z[idx_sub, :]
        delta_unwrapped = 2.0 * np.unwrap(np.angle(z_main)) - np.unwrap(np.angle(z_sub))
        delta_wrapped = np.mod(delta_unwrapped, 2.0 * np.pi)
        delta_principal = np.angle(np.exp(1j * delta_wrapped))
        mean_complex = np.mean(np.exp(1j * delta_wrapped))

        row['T_main'] = float(T_main)
        row['T_sub'] = float(T_sub)
        row['detune_pct'] = float(abs(T_sub / (T_main / 2.0) - 1.0) * 100.0)
        row['R'] = float(np.abs(mean_complex))
        row['median_abs_deg'] = float(np.median(np.abs(np.degrees(delta_principal))))
        row['sub_Re'] = float(eigvals[idx_sub].real)

    return row


def _print_table(rows, mode_tag):
    print("=" * 132)
    print(f"LPPL SIMULATION HARMONIC SCAN -- {mode_tag}")
    print("=" * 132)
    print("cfg    dim  poly  pairs  fit_err   ode_err   T_main    T_sub   2:1_detune   R        med|dphi|   sub_Re")
    print("-" * 132)
    for row in rows:
        print(
            f"{row['label']:>5s}  "
            f"{row['dim']:>3d}  "
            f"{row['poly']:>4d}  "
            f"{row['pairs']:>5d}  "
            f"{row['fit_err']:>7.2e}  "
            f"{row['ode_err']:>7.2e}  "
            f"{_fmt_year(row['T_main']):>8s}  "
            f"{_fmt_year(row['T_sub']):>8s}  "
            f"{_fmt_pct(row['detune_pct']):>10s}  "
            f"{_fmt_num(row['R'], '.5f'):>7s}  "
            f"{_fmt_num(row['median_abs_deg'], '.2f'):>11s}  "
            f"{_fmt_num(row['sub_Re'], '+.2e'):>8s}"
        )
    print("=" * 132)


def _plot(rows, mode_tag):
    labels = [row['label'] for row in rows]
    x = np.arange(len(rows))
    T_main = np.array([row['T_main'] for row in rows], dtype=float)
    T_sub = np.array([row['T_sub'] for row in rows], dtype=float)
    detune = np.array([row['detune_pct'] for row in rows], dtype=float)
    R = np.array([row['R'] for row in rows], dtype=float)
    med = np.array([row['median_abs_deg'] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, T_main, marker='o', color='tab:blue', label='main')
    ax.plot(x, T_sub, marker='o', color='tab:red', label='harm')
    ax.set_title('Learned periods')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel('years')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[0, 1]
    ax.plot(x, detune, marker='o', color='tab:purple')
    ax.axhline(5.0, color='gray', ls='--', lw=0.9, alpha=0.8, label='5%')
    ax.set_title('2:1 detuning')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel('percent')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[1, 0]
    ax.plot(x, R, marker='o', color='tab:green')
    ax.set_title('Phase-lock strength R')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel('R')
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.plot(x, med, marker='o', color='tab:orange')
    ax.set_title('Median $|\\Delta\\phi|$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_ylabel('deg')
    ax.grid(True, alpha=0.25)

    fig.suptitle(f'LPPL Simulation Harmonic Scan -- {mode_tag}', fontsize=11)
    return fig


def main():
    ap = argparse.ArgumentParser(description='05: Scan ssm_dim x poly on LPPL simulation')
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--t_final', type=float, default=DEFAULT_T_FINAL)
    ap.add_argument('--n_eval', type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--scan_ssm_dim', type=str, default='2,3,4,5', metavar='CSV',
                    help='comma-separated ssm_dim values')
    ap.add_argument('--scan_poly', type=str, default='1,2', metavar='CSV',
                    help='comma-separated poly values')
    ap.add_argument('--wang-off', action='store_true',
                    help='quasi-2D: Z_A=0, Z_B=0, Z_MIX=8e-5')
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    dims = _parse_csv_ints(args.scan_ssm_dim)
    polys = _parse_csv_ints(args.scan_poly)
    if not dims or not polys:
        ap.error('--scan_ssm_dim and --scan_poly must both contain at least one integer')

    p = build_params(wang_off=args.wang_off)
    mode_tag = "QUASI-2D" if args.wang_off else "FULL 3D"

    # Build context ONCE
    ctx, tau, pca_res, t, traj = build_lppl_context(
        p, args.M, args.years, args.t_final, args.n_eval)

    rows = []
    for dim in dims:
        for poly in polys:
            try:
                rows.append(_scan_case(ctx, dim, poly))
            except Exception as exc:
                rows.append({
                    'label': f's{dim}p{poly}',
                    'dim': dim,
                    'poly': poly,
                    'pairs': 0,
                    'fit_err': np.nan,
                    'ode_err': np.nan,
                    'T_main': np.nan,
                    'T_sub': np.nan,
                    'detune_pct': np.nan,
                    'R': np.nan,
                    'median_abs_deg': np.nan,
                    'sub_Re': np.nan,
                })
                print(f"warning: dim={dim}, poly={poly} failed: {exc}")

    _print_table(rows, mode_tag)
    if args.show:
        _plot(rows, mode_tag)
        plt.show()


if __name__ == '__main__':
    main()
