#!/usr/bin/env python3
"""CLI: parameter sweep over ssm_dim x poly for BTC residuals.
Analog to Wang 05_cli_scan — table + 4-panel summary plot."""

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

from analyze_residuals.constants import DEFAULT_FILENAME, DEFAULT_M, DEFAULT_YEARS, START_IDX
from analyze_residuals.common import analysis_time_vector
from analyze_residuals.ssm_learn import run_slave_test
from analyze_residuals.data import build_residual_context

try:
    plt.style.use('/home/hz/Data/Attractor/hz.mplstyle')
except Exception:
    plt.rcParams.update({
        'figure.facecolor': '#0a0a0a', 'axes.facecolor': '#1a1a1a',
        'text.color': '#CCCCCC', 'axes.labelcolor': '#CCCCCC',
        'xtick.color': '#CCCCCC', 'ytick.color': '#CCCCCC',
    })


def _period_value(ev, time_mode):
    period = 2.0 * np.pi / abs(ev.imag)
    if time_mode == 'linear':
        return period / 365.25
    return period


def main():
    ap = argparse.ArgumentParser(description='BTC residual SSM sweep (scan_ssm_dim x scan_poly)')
    ap.add_argument('--filename', type=str, default=DEFAULT_FILENAME, help='input CSV')
    ap.add_argument('--M', type=int, default=DEFAULT_M, help='embedding dimension')
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS, help='embedding window in years')
    ap.add_argument('--start_idx', type=int, default=START_IDX, help='row offset in CSV')
    ap.add_argument('--time_mode', choices=['linear', 'log'], default='linear',
                    help='analysis clock for reduced dynamics and slave fits')
    ap.add_argument('--scan_ssm_dim', type=str, default='2,3,4', metavar='CSV', help='comma-separated ssm_dim values')
    ap.add_argument('--scan_poly', type=str, default='1,2,3', metavar='CSV', help='comma-separated poly values')
    ap.add_argument('--max_slave_pc', type=int, default=6, help='highest PC index tested as slave')
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    has_scan_dim = '--scan_ssm_dim' in sys.argv
    has_scan_poly = '--scan_poly' in sys.argv
    if has_scan_dim != has_scan_poly:
        ap.error('--scan_ssm_dim and --scan_poly must be provided together')

    dims = [int(x.strip()) for x in args.scan_ssm_dim.split(',')]
    polys = [int(x.strip()) for x in args.scan_poly.split(',')]
    period_unit = 'y' if args.time_mode == 'linear' else 'fit'

    payload = build_residual_context(args.filename, args.M, args.years, args.start_idx, time_mode=args.time_mode)
    ctx = payload['ctx']
    time_vec = analysis_time_vector(ctx)

    rows = []
    print("=" * 110)
    print("BTC RESIDUAL SSM SWEEP")
    print("=" * 110)
    print(f"  time_mode: {args.time_mode}")
    print(f"  {'dim':>3s}  {'poly':>4s}  {'fit_err':>8s}  {'pairs':>5s}  {'T_main':>8s}  "
          f"{'T_sub':>8s}  {'PC3_R2':>7s}  {'PC4_R2':>7s}  {'PC5_R2':>7s}  {'PC6_R2':>7s}  "
          f"{'h.slaved':>8s}  {'p.slaved':>8s}")
    print("  " + "-" * 107)

    for dim in dims:
        for poly in polys:
            result = run_slave_test(
                ctx,
                dim,
                poly,
                args.max_slave_pc,
                time_vec=time_vec,
                time_mode=args.time_mode,
            )
            eig = result['eigvals']
            osc = sorted([ev for ev in eig if ev.imag > 1e-6], key=lambda z: abs(z.imag))
            T_main = _period_value(osc[0], args.time_mode) if osc else None
            T_sub = _period_value(osc[1], args.time_mode) if len(osc) >= 2 else None
            T_main_txt = f"{T_main:.3f}{period_unit}" if T_main is not None else "-"
            T_sub_txt = f"{T_sub:.3f}{period_unit}" if T_sub is not None else "-"

            sr = result['slave_results']
            pc_r2 = {s['pc']: s['r2'] for s in sr}
            slaved = sum(1 for s in sr if s['verdict'] == 'HARM.SLAVED')
            partial = sum(1 for s in sr if s['verdict'] == 'PART.SLAVED')

            def r2_txt(k):
                return f"{pc_r2[k]:.3f}" if k in pc_r2 else "master"

            print(f"  {dim:>3d}  {poly:>4d}  {result['fit_err']:>8.3f}  "
                  f"{len(osc):>5d}  {T_main_txt:>8s}  {T_sub_txt:>8s}  "
                  f"{r2_txt(2):>7s}  {r2_txt(3):>7s}  {r2_txt(4):>7s}  {r2_txt(5):>7s}  "
                  f"{slaved:>8d}  {partial:>8d}")

            rows.append({
                'dim': dim,
                'poly': poly,
                'fit_err': result['fit_err'],
                'n_osc': len(osc),
                'T_main': T_main,
                'T_sub': T_sub,
                'pc3_r2': pc_r2.get(2),
                'pc4_r2': pc_r2.get(3),
                'slaved': slaved,
                'partial': partial,
            })

    print("  " + "-" * 107)
    print("  R2 >= 0.70 = HARM.SLAVED, 0.30-0.70 = PART.SLAVED, < 0.30 = INDEPENDENT")
    print("  part.slaved = harmonically coupled to master via poly nonlinearity (NOT an independent frequency)")
    print("=" * 110)

    if args.show and rows:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        labels = [f"s{r['dim']}p{r['poly']}" for r in rows]
        x = np.arange(len(rows))

        ax = axes[0, 0]
        ax.bar(x, [r['fit_err'] for r in rows], color='tab:blue', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_ylabel('fit_err')
        ax.set_title('Decoder fit error')
        ax.grid(True, alpha=0.25)

        ax = axes[0, 1]
        T_vals = [r['T_main'] if r['T_main'] is not None else 0.0 for r in rows]
        ax.bar(x, T_vals, color='tab:orange', alpha=0.8)
        if args.time_mode == 'linear':
            ax.axhline(3.86, color='white', ls='--', lw=0.8, alpha=0.5, label='~3.86y halving')
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_ylabel('T_main (years)' if args.time_mode == 'linear' else 'T_main (fit units)')
        ax.set_title('Main period')
        ax.grid(True, alpha=0.25)

        ax = axes[1, 0]
        pc3 = [r['pc3_r2'] if r['pc3_r2'] is not None else 0 for r in rows]
        pc4 = [r['pc4_r2'] if r['pc4_r2'] is not None else 0 for r in rows]
        ax.bar(x - 0.15, pc3, 0.3, label='PC3 $R^2$', color='lime', alpha=0.8)
        ax.bar(x + 0.15, pc4, 0.3, label='PC4 $R^2$', color='cyan', alpha=0.8)
        ax.axhline(0.70, color='white', ls='--', lw=0.8, alpha=0.5, label='SLAVED threshold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_ylabel('$R^2$')
        ax.set_title('Slave test $R^2$')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

        ax = axes[1, 1]
        s_vals = [r['slaved'] for r in rows]
        p_vals = [r['partial'] for r in rows]
        ax.bar(x - 0.15, s_vals, 0.3, label='harm.slaved', color='lime', alpha=0.8)
        ax.bar(x + 0.15, p_vals, 0.3, label='part.slaved', color='yellow', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_ylabel('count')
        ax.set_title('Harm.slaved / part.slaved count')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

        fig.suptitle(f'BTC Residual SSM Sweep  time={args.time_mode}', fontsize=11)
        plt.show()


if __name__ == '__main__':
    main()
