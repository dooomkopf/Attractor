#!/usr/bin/env python3
"""15: Pre-Whitening + Uhr-Vergleich der SSM-Pipelines auf BTC-Residuen.

Workflow (kanonisch wie /home/hz/Data/BTC3/bf_filt.py):
  1. ziel.csv + QuantReg(0.01)              -> log_residuals
  2. FractionalDifferencing(d=0.98)         -> Pre-Whitening (Hosking 1981)
     spectral_slope = 2*d ~ 1.96 -> Random-Walk-Trend wird flach
  3. masked range (days >= start_idx)
  4. ZWEI SSM-Pipelines auf prewhitened residuen:
        linear-time clock  (np.interp linear)
        log10-time clock   (np.interp log10)

Plot 4 Zeilen x 3 Spalten — eine Uhr pro Reihe (gruen=linear, blau=log10):
  Row 1   raw residuals (linear axis)   PSD raw (linear-time)   eigenvalues UNFILTERED
  Row 2   raw residuals (log10 axis)    PSD raw (log10-time)    eigenvalues UNFILTERED
  Row 3   prewhitened (linear axis)     PC1/PC2 (linear-clock)  eigenvalues PRE-WHITENED
  Row 4   prewhitened (log10 axis)      PC1/PC2 (log10-clock)   eigenvalues PRE-WHITENED

Zwei Figures:
  *_pwlin.png  -> linear-time fracdiff (alle SSMs auf linear-prewhitened input)
  *_pwlog.png  -> log10-time fracdiff  (alle SSMs auf log-prewhitened input)

Module:
  _15_preproc.py    QuantReg + fracdiff (linear/log)
  _15_pipelines.py  Embedding + SSM-Fit + Mode-Picker (1:1 ssmlearn_res, 13_*-Style)
  _15_spectra.py    PSD via raw FFT + 4x zero-pad (1:1 paper-fft-bubbles-soc.py)
  _15_plot.py       Plot helpers (incl. SG smoothing)

CLI:
  ./15_cli_resample_control.py --ssm_dim 2 --poly 2
  ./15_cli_resample_control.py --no_prewhiten
"""

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if '/home/hz/Data/Attractor' not in sys.path:
    sys.path.insert(0, '/home/hz/Data/Attractor')
if '/home/hz/Data/BTC3' not in sys.path:
    sys.path.insert(0, '/home/hz/Data/BTC3')

HZ_MPLSTYLE = '/home/hz/Data/hz.mplstyle'
if os.path.exists(HZ_MPLSTYLE):
    plt.style.use(HZ_MPLSTYLE)
    mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']

import importlib  # noqa: E402

from analyze_residuals.constants import (  # noqa: E402
    DEFAULT_FILENAME, DEFAULT_M, DEFAULT_YEARS, START_IDX, DAYS_PER_YEAR,
)

# leading-digit module names need importlib (Python identifier rule)
_preproc = importlib.import_module('analyze_residuals.15_cli_resample_control_preproc')
_pipes = importlib.import_module('analyze_residuals.15_cli_resample_control_pipelines')
_spectra = importlib.import_module('analyze_residuals.15_cli_resample_control_spectra')
_plot = importlib.import_module('analyze_residuals.15_cli_resample_control_plot')

load_and_prewhiten = _preproc.load_and_prewhiten
prewhiten_in_log_clock = _preproc.prewhiten_in_log_clock
build_linear_RS = _pipes.build_linear_RS
build_log_RS = _pipes.build_log_RS
fit_and_extract = _pipes.fit_and_extract
psd_linear = _spectra.psd_linear
psd_logtime = _spectra.psd_logtime
pc1_dominant_period = _spectra.pc1_dominant_period
frame = _plot.frame
plot_residual_ts = _plot.plot_residual_ts
plot_psd_linear = _plot.plot_psd_linear
plot_psd_log = _plot.plot_psd_log
plot_pc = _plot.plot_pc
plot_eigs = _plot.plot_eigs

COLOR_LIN = '#69DB7C'   # green = linear-time clock
COLOR_LOG = '#4DABF7'   # blue  = log10-time clock


def main():
    ap = argparse.ArgumentParser(
        description='Pre-whitening + clock comparison: linear-time vs log10-time')
    ap.add_argument('--filename', default=DEFAULT_FILENAME)
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--start_idx', type=int, default=START_IDX)
    ap.add_argument('--ssm_dim', type=int, default=2)
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2)
    ap.add_argument('--no_prewhiten', action='store_true')
    ap.add_argument('--fracdiff_d', type=float, default=0.98)
    ap.add_argument('--trim_percent', type=float, default=0.0)
    ap.add_argument('--smooth_days', type=float, default=180.0)
    ap.add_argument('--no-show', action='store_true')
    args = ap.parse_args()

    use_prewhiten = not args.no_prewhiten
    print(f'preprocess (prewhiten={use_prewhiten}, d={args.fracdiff_d}, '
          f'trim={args.trim_percent}%)...')
    pp = load_and_prewhiten(args.filename, args.start_idx,
                            use_prewhiten=use_prewhiten,
                            fracdiff_d=args.fracdiff_d,
                            trim_percent=args.trim_percent)
    days = pp['days']
    log_res_raw = pp['log_res_raw']
    log_res_pw = pp['log_res_pw']
    print(f'  N samples = {len(days)}')
    if pp['fracdiff_info']:
        info = pp['fracdiff_info']
        print(f"  fracdiff: n_weights={info['n_weights']}, "
              f"spectral_slope=2*d={info['spectral_slope']:.3f}")

    # Two pre-whitening variants
    days_A, input_A = days, log_res_pw
    if use_prewhiten:
        days_B, input_B, _ = prewhiten_in_log_clock(
            days, log_res_raw,
            fracdiff_d=args.fracdiff_d, trim_percent=args.trim_percent)
    else:
        days_B, input_B = days, log_res_raw

    # Reference SSM on RAW for unfiltered eigenvalue panels
    ctx_lin_raw = build_linear_RS(days, log_res_raw, args.M, args.years, len(days))
    ext_lin_raw = fit_and_extract(ctx_lin_raw, args.ssm_dim, args.poly_degree)
    ctx_log_raw = build_log_RS(days, log_res_raw, args.M, args.years)
    ext_log_raw = fit_and_extract(ctx_log_raw, args.ssm_dim, args.poly_degree)

    ctx_lin_A = build_linear_RS(days_A, input_A, args.M, args.years, len(days_A))
    ext_lin_A = fit_and_extract(ctx_lin_A, args.ssm_dim, args.poly_degree)
    ctx_log_A = build_log_RS(days_A, input_A, args.M, args.years)
    ext_log_A = fit_and_extract(ctx_log_A, args.ssm_dim, args.poly_degree)
    ctx_lin_B = build_linear_RS(days_B, input_B, args.M, args.years, len(days_B))
    ext_lin_B = fit_and_extract(ctx_lin_B, args.ssm_dim, args.poly_degree)
    ctx_log_B = build_log_RS(days_B, input_B, args.M, args.years)
    ext_log_B = fit_and_extract(ctx_log_B, args.ssm_dim, args.poly_degree)

    def _eig_str(e_complex, c):
        if abs(e_complex.imag) < 1e-12:
            return f'{e_complex.real:+.3e} (real)'
        T = 2.0 * np.pi / abs(e_complex.imag)
        if c['clock_x_log10']:
            return f'{e_complex.real:+.2e}{e_complex.imag:+.2e}j  T={T:.3f}log10d (lam={10**T:.3f})'
        return f'{e_complex.real:+.2e}{e_complex.imag:+.2e}j  T={T:.0f}d ({T/DAYS_PER_YEAR:.2f}y)'

    def _row(lbl, c, e):
        pc1_tops = pc1_dominant_period(c)
        pc1_str = ' | '.join(pc1_tops)
        eig_lines = [f"  {'':35s}    eig[{k}] = {_eig_str(ev, c)}"
                     for k, ev in enumerate(e['eigvals'])]
        master = ('NO master picked' if e.get('lam_main') is None
                  else f'idx={e["idx_main"]}')
        return ('\n'.join([
            f"  {lbl:35s}: master = {master}",
            f"  {'':35s}  ALL eigenvalues:",
        ] + eig_lines + [
            f"  {'':35s}  PC1 top-3 periodogram peaks: {pc1_str}",
        ]))

    print('=' * 92)
    print(f'PRE-WHITENING + CLOCK-COMPARISON  '
          f'(d={args.fracdiff_d}, trim={args.trim_percent}%)')
    print('=' * 92)
    print('-- linear-time fracdiff --')
    print(_row('linear-clock SSM', ctx_lin_A, ext_lin_A))
    print(_row('log10-clock  SSM', ctx_log_A, ext_log_A))
    print('-- log10-time fracdiff --')
    print(_row('linear-clock SSM', ctx_lin_B, ext_lin_B))
    print(_row('log10-clock  SSM', ctx_log_B, ext_log_B))
    print('=' * 92)

    f_lin_raw_y, p_lin_raw = psd_linear(log_res_raw, days)
    f_log_raw, p_log_raw = psd_logtime(log_res_raw, days)
    f_lin_A_y, p_lin_A = psd_linear(input_A, days_A)
    f_log_A, p_log_A = psd_logtime(input_A, days_A)
    f_lin_B_y, p_lin_B = psd_linear(input_B, days_B)
    f_log_B, p_log_B = psd_logtime(input_B, days_B)

    def _make_figure(filter_name, days_v, input_v,
                     ctx_lin_v, ext_lin_v, ctx_log_v, ext_log_v,
                     f_lin_y, p_lin, f_log_, p_log_, suffix):
        fig = plt.figure(figsize=(17, 16))
        gs = fig.add_gridspec(4, 3, width_ratios=[1.7, 1.5, 1.0])

        ax = fig.add_subplot(gs[0, 0]); frame(ax, COLOR_LIN)
        plot_residual_ts(ax, days, log_res_raw, COLOR_LIN,
                         'raw log-residuals (linear axis)', x_log10=False)
        ax = fig.add_subplot(gs[0, 1]); frame(ax, COLOR_LIN)
        plot_psd_linear(ax, f_lin_raw_y, p_lin_raw, COLOR_LIN,
                        'PSD raw (linear-time)')
        ax = fig.add_subplot(gs[0, 2]); frame(ax, COLOR_LIN)
        plot_eigs(ax, ext_lin_raw, ctx_lin_raw, COLOR_LIN)
        ax.set_title(r'eigenvalues $M$ — unfiltered (linear-clock)', fontsize=10)

        ax = fig.add_subplot(gs[1, 0]); frame(ax, COLOR_LOG)
        plot_residual_ts(ax, days, log_res_raw, COLOR_LOG,
                         r'raw log-residuals ($\log_{10}$ axis)', x_log10=True)
        ax = fig.add_subplot(gs[1, 1]); frame(ax, COLOR_LOG)
        plot_psd_log(ax, f_log_raw, p_log_raw, COLOR_LOG,
                     'PSD raw (log10-time)')
        ax = fig.add_subplot(gs[1, 2]); frame(ax, COLOR_LOG)
        plot_eigs(ax, ext_log_raw, ctx_log_raw, COLOR_LOG)
        ax.set_title(r'eigenvalues $M$ — unfiltered (log-clock)', fontsize=10)

        ax = fig.add_subplot(gs[2, 0]); frame(ax, COLOR_LIN)
        plot_residual_ts(ax, days_v, input_v, COLOR_LIN,
                         f'{filter_name} residuals (linear axis)', x_log10=False)
        ax = fig.add_subplot(gs[2, 1]); frame(ax, COLOR_LIN)
        plot_pc(ax, ctx_lin_v, smooth_days=args.smooth_days, legend_loc='lower right',
                ctx_unfilt=ctx_lin_raw)
        ymax = float(max(np.max(np.abs(ctx_lin_v['pc'][:, 0])),
                          np.max(np.abs(ctx_lin_v['pc'][:, 1])))) * 1.05
        ax.set_ylim(-ymax, ymax)
        ax.set_title(f'{filter_name} — linear-clock SSM', fontsize=10)
        ax = fig.add_subplot(gs[2, 2]); frame(ax, COLOR_LIN)
        plot_eigs(ax, ext_lin_v, ctx_lin_v, COLOR_LIN)
        ax.set_title(rf'eigenvalues $M$ — {filter_name} (linear-clock)', fontsize=10)

        ax = fig.add_subplot(gs[3, 0]); frame(ax, COLOR_LOG)
        plot_residual_ts(ax, days_v, input_v, COLOR_LOG,
                         f'{filter_name} residuals ($\\log_{{10}}$ axis)',
                         x_log10=True)
        ax = fig.add_subplot(gs[3, 1]); frame(ax, COLOR_LOG)
        plot_pc(ax, ctx_log_v, smooth_days=args.smooth_days,
                legend_loc='lower right', ext=ext_log_v,
                ctx_unfilt=ctx_log_raw)
        ymax = float(max(np.max(np.abs(ctx_log_v['pc'][:, 0])),
                          np.max(np.abs(ctx_log_v['pc'][:, 1])))) * 1.05
        ax.set_ylim(-ymax, ymax)
        ax.set_title(f'{filter_name} — log10-clock SSM', fontsize=10)
        ax = fig.add_subplot(gs[3, 2]); frame(ax, COLOR_LOG)
        plot_eigs(ax, ext_log_v, ctx_log_v, COLOR_LOG)
        ax.set_title(rf'eigenvalues $M$ — {filter_name} (log-clock)', fontsize=10)

        with plt.rc_context({'text.usetex': False}):
            plt.suptitle(
                f'BTC residuals: {filter_name}  '
                f'[ssm_dim={args.ssm_dim}, poly={args.poly_degree}, '
                f'M={args.M}, years={args.years}]',
                color='#CCCCCC', fontsize=13, y=0.995,
                fontname='Comfortaa', fontweight='bold')

        plt.subplots_adjust(top=0.96, bottom=0.04, left=0.05, right=0.98,
                            hspace=0.50, wspace=0.30)

        save_path = os.path.join(
            HERE,
            f'15_resample_control_{suffix}_dim{args.ssm_dim}_poly{args.poly_degree}.png',
        )
        fig.savefig(save_path, dpi=200, facecolor='#0a0a0a')
        print(f'plot saved to {save_path}')
        return fig

    _make_figure('linear-time fracdiff', days_A, input_A,
                 ctx_lin_A, ext_lin_A, ctx_log_A, ext_log_A,
                 f_lin_A_y, p_lin_A, f_log_A, p_log_A, suffix='pwlin')
    _make_figure('log10-time fracdiff', days_B, input_B,
                 ctx_lin_B, ext_lin_B, ctx_log_B, ext_log_B,
                 f_lin_B_y, p_lin_B, f_log_B, p_log_B, suffix='pwlog')

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
