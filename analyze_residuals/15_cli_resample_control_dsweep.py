#!/usr/bin/env python3
"""15: fracdiff-d-Sweep — Tabelle SSM-Mode vs LPPL-Fit-Mode.

Sweept fracdiff_d von 0.18..0.98 (Schritt 0.1) und gibt fuer jede
Pre-Whitening-Variante die SSM-Master-Mode (lambda) und den LPPL-Fit-Mode
(lambda) auf der log-clock-PC1 aus.

Spalten:
  d | (Variante A: linear-time fracdiff) SSM_lin SSM_log Fit_log |
      (Variante B: log10-time fracdiff)  SSM_lin SSM_log Fit_log

Nur Print, kein Plot.

CLI:
  ./15_cli_resample_control_dsweep.py --ssm_dim 2 --poly 2
"""
import argparse
import importlib
import os
import sys

import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if '/home/hz/Data/Attractor' not in sys.path:
    sys.path.insert(0, '/home/hz/Data/Attractor')
if '/home/hz/Data/BTC3' not in sys.path:
    sys.path.insert(0, '/home/hz/Data/BTC3')

from analyze_residuals.constants import (  # noqa: E402
    DEFAULT_FILENAME, DEFAULT_M, DEFAULT_YEARS, START_IDX,
)

_preproc = importlib.import_module('analyze_residuals.15_cli_resample_control_preproc')
_pipes = importlib.import_module('analyze_residuals.15_cli_resample_control_pipelines')
_plot = importlib.import_module('analyze_residuals.15_cli_resample_control_plot')

load_and_prewhiten = _preproc.load_and_prewhiten
prewhiten_in_log_clock = _preproc.prewhiten_in_log_clock
build_linear_RS = _pipes.build_linear_RS
build_log_RS = _pipes.build_log_RS
fit_and_extract = _pipes.fit_and_extract
_ssm_mode_fit_pc1 = _plot._ssm_mode_fit_pc1


def _sg_smooth_pc1(ctx, smooth_days):
    days = np.asarray(ctx['days_vecs'], dtype=float)
    fit_time = np.asarray(ctx['fit_time_vecs'], dtype=float)
    pc1 = np.asarray(ctx['pc'][:, 0], dtype=float)
    dt = float(np.median(np.diff(fit_time)))
    if ctx['clock_x_log10']:
        geo_mean = float(np.sqrt(days.min() * days.max()))
        half = smooth_days * 0.5
        win_log = np.log10(geo_mean + half) - np.log10(max(geo_mean - half, 1.0))
        sw = max(1, int(round(win_log / max(dt, 1e-12))))
    else:
        sw = max(1, int(round(smooth_days / max(dt, 1e-12))))
    win = sw if sw % 2 == 1 else sw + 1
    polyorder = 3
    if win < polyorder + 2:
        win = polyorder + 3
    if win >= len(pc1):
        win = (len(pc1) // 2) * 2 - 1
    return savgol_filter(pc1, window_length=win, polyorder=polyorder)


def _ssm_lam(ctx, ext):
    """Return lambda aus SSM (log-clock: 10^T_main) oder str-Marker."""
    if ext.get('lam_main') is None or ext.get('T_main') is None:
        return None
    if ctx['clock_x_log10']:
        return 10.0 ** ext['T_main']
    return None  # linear-clock: lambda macht so keinen Sinn (T in Tagen)


def _ssm_T(ctx, ext):
    """Return T aus SSM (log-clock: log10d, linear-clock: days)."""
    if ext.get('T_main') is None:
        return None
    return ext['T_main']


def _fit_lam(ctx, ext, smooth_days):
    """Run LPPL fit on SG-smoothed log-clock PC1, return lam_fit or None."""
    if not ctx['clock_x_log10']:
        return None
    if ext.get('lam_main') is None:
        return None
    pc1_s = _sg_smooth_pc1(ctx, smooth_days)
    ctx_for_fit = dict(ctx)
    ctx_for_fit['_pc1_smoothed_for_fit'] = pc1_s
    fit_result = _ssm_mode_fit_pc1(ctx_for_fit, ext)
    if fit_result is None or fit_result[0] is None:
        return None
    _, _T_log10, lam_fit, r2, omega_fit, n_osc = fit_result
    return dict(lam=lam_fit, omega=omega_fit, n_osc=n_osc, r2=r2)


def main():
    ap = argparse.ArgumentParser(
        description='fracdiff-d sweep + SSM/Fit mode comparison')
    ap.add_argument('--filename', default=DEFAULT_FILENAME)
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--start_idx', type=int, default=START_IDX)
    ap.add_argument('--ssm_dim', type=int, default=2)
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2)
    ap.add_argument('--smooth_days', type=float, default=180.0)
    ap.add_argument('--d_min', type=float, default=0.18)
    ap.add_argument('--d_max', type=float, default=0.98)
    ap.add_argument('--d_step', type=float, default=0.1)
    args = ap.parse_args()

    d_grid = np.arange(args.d_min, args.d_max + 1e-9, args.d_step)
    print('=' * 110)
    print(f'fracdiff-d sweep  (M={args.M}, years={args.years}, '
          f'ssm_dim={args.ssm_dim}, poly={args.poly_degree})')
    print('=' * 110)
    print(f"{'d':>5} || "
          f"{'A=lin-fracdiff: lin-clk T':>26} | "
          f"{'log-clk lam_SSM':>17} | "
          f"{'log-clk Fit (omega,R2)':>26} || "
          f"{'B=log-fracdiff: lin-clk T':>26} | "
          f"{'log-clk lam_SSM':>17} | "
          f"{'log-clk Fit (omega,R2)':>26}")
    print('-' * 200)

    def _T_lin_str(ext_lin):
        T = ext_lin.get('T_main')
        return (f"T={T:.0f}d ({T/365.25:.2f}y)" if T is not None else 'no osc')

    def _lam_log_str(ctx_log, ext_log):
        lam = _ssm_lam(ctx_log, ext_log)
        return (f'{lam:.3f}' if lam is not None else 'no osc')

    def _fit_str(ctx_log, ext_log, smooth_days):
        f = _fit_lam(ctx_log, ext_log, smooth_days)
        return (f'{f["lam"]:.3f} ({f["omega"]:.1f}, R2={f["r2"]:.3f})'
                if f is not None else 'n/a')

    for d in d_grid:
        d = float(d)
        # Variante A: linear-fracdiff (gleicher Input fuer beide SSMs)
        pp = load_and_prewhiten(args.filename, args.start_idx,
                                use_prewhiten=True, fracdiff_d=d,
                                trim_percent=0.0)
        days_A, input_A = pp['days'], pp['log_res_pw']
        ctx_lin_A = build_linear_RS(days_A, input_A, args.M, args.years, len(days_A))
        ext_lin_A = fit_and_extract(ctx_lin_A, args.ssm_dim, args.poly_degree)
        ctx_log_A = build_log_RS(days_A, input_A, args.M, args.years)
        ext_log_A = fit_and_extract(ctx_log_A, args.ssm_dim, args.poly_degree)

        # Variante B: log-fracdiff (gleicher Input fuer beide SSMs)
        days_B, input_B, _ = prewhiten_in_log_clock(
            pp['days'], pp['log_res_raw'], fracdiff_d=d, trim_percent=0.0)
        ctx_lin_B = build_linear_RS(days_B, input_B, args.M, args.years, len(days_B))
        ext_lin_B = fit_and_extract(ctx_lin_B, args.ssm_dim, args.poly_degree)
        ctx_log_B = build_log_RS(days_B, input_B, args.M, args.years)
        ext_log_B = fit_and_extract(ctx_log_B, args.ssm_dim, args.poly_degree)

        print(f"{d:>5.2f} || "
              f"{_T_lin_str(ext_lin_A):>26} | "
              f"{_lam_log_str(ctx_log_A, ext_log_A):>17} | "
              f"{_fit_str(ctx_log_A, ext_log_A, args.smooth_days):>26} || "
              f"{_T_lin_str(ext_lin_B):>26} | "
              f"{_lam_log_str(ctx_log_B, ext_log_B):>17} | "
              f"{_fit_str(ctx_log_B, ext_log_B, args.smooth_days):>26}")

    print('=' * 110)


if __name__ == '__main__':
    main()
