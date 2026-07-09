#!/usr/bin/env python3
"""CLI: spectral analysis (PSD) of BTC residuals per PC.
Analog to Wang 02_cli_harmonics — PSD per channel, dominant freq + 2w marked."""

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_residuals.constants import DEFAULT_FILENAME, DEFAULT_M, DEFAULT_YEARS, START_IDX, DAYS_PER_YEAR
from analyze_residuals.common import analysis_time_vector
from analyze_residuals.data import build_residual_context

try:
    plt.style.use('/home/hz/Data/Attractor/hz.mplstyle')
except Exception:
    plt.rcParams.update({
        'figure.facecolor': '#0a0a0a', 'axes.facecolor': '#1a1a1a',
        'text.color': '#CCCCCC', 'axes.labelcolor': '#CCCCCC',
        'xtick.color': '#CCCCCC', 'ytick.color': '#CCCCCC',
    })


def _analyze_pc(signal, fs, min_freq=1e-5):
    f, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal) // 2))
    mask = f > min_freq
    f_pos, psd_pos = f[mask], psd[mask]
    idx_dom = np.argmax(psd_pos)
    f_dom = f_pos[idx_dom]
    f_2w_target = 2.0 * f_dom
    idx_2w = np.argmin(np.abs(f_pos - f_2w_target))
    f_2w = f_pos[idx_2w]
    ratio = psd_pos[idx_2w] / (psd_pos[idx_dom] + 1e-30)
    return {
        'freq': f_pos, 'psd': psd_pos,
        'f_dom': f_dom, 'psd_dom': psd_pos[idx_dom],
        'f_2w_target': f_2w_target, 'f_2w': f_2w, 'psd_2w': psd_pos[idx_2w],
        'ratio_2w_1w': ratio,
    }


def main():
    ap = argparse.ArgumentParser(description='Spectral analysis of BTC residuals (PSD per PC)')
    ap.add_argument('--filename', type=str, default=DEFAULT_FILENAME)
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--start_idx', type=int, default=START_IDX)
    ap.add_argument('--time_mode', choices=['linear', 'log'], default='linear')
    ap.add_argument('--n_pcs', type=int, default=6)
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    payload = build_residual_context(args.filename, args.M, args.years, args.start_idx, time_mode=args.time_mode)
    ctx = payload['ctx']
    pc = ctx['pc']
    fit_time = analysis_time_vector(ctx)
    dt = float(np.median(np.diff(fit_time)))
    fs = 1.0 / dt
    n_pcs = min(args.n_pcs, pc.shape[1])
    period_scale = DAYS_PER_YEAR if args.time_mode == 'linear' else 1.0
    period_unit = 'y' if args.time_mode == 'linear' else 'fit'
    freq_scale = period_scale

    results = {}
    for k in range(n_pcs):
        results[f'PC{k+1}'] = _analyze_pc(pc[:, k], fs)

    ref = results['PC1']
    print("=" * 80)
    print("BTC RESIDUAL HARMONIC READOUT")
    print("=" * 80)
    print(f"  time_mode         : {args.time_mode}")
    print(f"  N_vec             : {ctx['N']}")
    dt_unit = 'd' if args.time_mode == 'linear' else 'fit'
    print(f"  dt                : {dt:.6f} {dt_unit}")
    print(f"  reference f0      : {ref['f_dom']:.6f}   T={1.0/(ref['f_dom']*period_scale):.3f}{period_unit}")
    print()
    print("CHANNELS")
    for key in [f'PC{k+1}' for k in range(n_pcs)]:
        r = results[key]
        T_dom = 1.0 / (r['f_dom'] * period_scale)
        print(f"  {key:>4s}: f_dom={r['f_dom']:.6f}  T_dom={T_dom:7.3f}{period_unit}  "
              f"f_2w≈{r['f_2w']:.6f}  PSD(2w)/PSD(w)={r['ratio_2w_1w']:.3e}")
    print("=" * 80)

    if args.show:
        n_rows = (n_pcs + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.5 * n_rows), constrained_layout=True)
        axes = axes.ravel() if n_pcs > 3 else [axes] if n_pcs == 1 else axes.ravel()
        for k in range(n_pcs):
            ax = axes[k]
            r = results[f'PC{k+1}']
            f_plot = r['freq'] * freq_scale
            ax.plot(f_plot, r['psd'], lw=1.2)
            ax.axvline(r['f_dom'] * freq_scale, color='tab:blue', lw=0.8, alpha=0.8, label='f0')
            ax.axvline(r['f_2w_target'] * freq_scale, color='tab:red', lw=0.8, ls='--', alpha=0.8, label='2*f0')
            ax.set_title(f'PC{k+1} ({ctx["var"][k]*100:.1f}\\%)')
            ax.set_xlabel('frequency (1/year)' if args.time_mode == 'linear' else 'frequency (1/fit)')
            ax.set_ylabel('PSD')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
        for k in range(n_pcs, len(axes)):
            axes[k].set_visible(False)
        fig.suptitle('BTC Residual PSD per PC', fontsize=11)
        plt.show()


if __name__ == '__main__':
    main()
