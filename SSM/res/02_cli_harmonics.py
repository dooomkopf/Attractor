#!/usr/bin/env python3
"""02: Harmonics -- PSD per PC from LPPL simulation, analog to Wang/BTC."""

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ATTRACTOR = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _ATTRACTOR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from lppl_system import build_params
from SSM_res_lppl_data import (
    build_lppl_context, DEFAULT_M, DEFAULT_YEARS, DEFAULT_T_FINAL,
    DEFAULT_N_EVAL, DAYS_PER_YEAR,
)

try:
    plt.style.use(os.path.join(_ATTRACTOR, 'hz.mplstyle'))
except Exception:
    pass


def _analyze_pc(signal, fs):
    f, psd = welch(signal, fs=fs, nperseg=min(1024, len(signal) // 2))
    mask = f > 1e-6
    fp, pp = f[mask], psd[mask]
    idx_dom = np.argmax(pp)
    f_dom = fp[idx_dom]
    idx_2w = np.argmin(np.abs(fp - 2.0 * f_dom))
    return {'freq': fp, 'psd': pp, 'f_dom': f_dom,
            'f_2w': fp[idx_2w], 'ratio': pp[idx_2w] / (pp[idx_dom] + 1e-30)}


def main():
    ap = argparse.ArgumentParser(description='02: PSD per PC from LPPL simulation')
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--t_final', type=float, default=DEFAULT_T_FINAL)
    ap.add_argument('--n_eval', type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--n_pcs', type=int, default=6)
    ap.add_argument('--wang-off', action='store_true',
                    help='quasi-2D: Z_A=0, Z_B=0, Z_MIX=8e-5')
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    p = build_params(wang_off=args.wang_off)
    mode_tag = "QUASI-2D" if args.wang_off else "FULL 3D"
    ctx, tau, pca_res, _, _ = build_lppl_context(
        p, args.M, args.years, args.t_final, args.n_eval)
    pc = pca_res.pc
    dt = float(np.median(np.diff(ctx['days_vecs'])))
    fs = 1.0 / dt
    n_pcs = min(args.n_pcs, pc.shape[1])

    results = {k: _analyze_pc(pc[:, k], fs) for k in range(n_pcs)}

    print("=" * 80)
    print(f"LPPL SIMULATION HARMONIC READOUT -- {mode_tag}")
    print("=" * 80)
    print(f"  N_vec       : {ctx['N']}")
    print(f"  dt          : {dt:.1f}d")
    print()
    print(f"  {'PC':>4}  {'var':>6}  {'f_dom':>10}  {'T_dom':>8}  {'f_2w':>10}  {'PSD_2w/1w':>12}")
    print("  " + "-" * 60)
    for k in range(n_pcs):
        r = results[k]
        T = 1.0 / (r['f_dom'] * DAYS_PER_YEAR) if r['f_dom'] > 0 else float('inf')
        print(f"  PC{k+1:2d}  {pca_res.var[k]*100:5.1f}%  {r['f_dom']:10.6f}  {T:7.3f}y  "
              f"{r['f_2w']:10.6f}  {r['ratio']:12.3e}")
    print("=" * 80)

    if args.show:
        n_rows = (n_pcs + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.5 * n_rows), constrained_layout=True)
        axes_flat = axes.ravel() if n_pcs > 3 else [axes] if n_pcs == 1 else axes.ravel()
        for k in range(n_pcs):
            ax = axes_flat[k]
            r = results[k]
            f_y = r['freq'] * DAYS_PER_YEAR
            ax.plot(f_y, r['psd'], lw=1.2)
            ax.axvline(r['f_dom'] * DAYS_PER_YEAR, color='tab:blue', lw=0.8, label='f0')
            ax.axvline(2 * r['f_dom'] * DAYS_PER_YEAR, color='tab:red', ls='--', lw=0.8, label='2f0')
            ax.set_title(f'PC{k+1} ({pca_res.var[k]*100:.1f}%)')
            ax.set_xlabel('freq (1/y)')
            ax.set_ylabel('PSD')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7)
        for k in range(n_pcs, len(axes_flat)):
            axes_flat[k].set_visible(False)
        fig.suptitle(f'LPPL Simulation PSD per PC -- {mode_tag}', fontsize=11)
        plt.show()


if __name__ == '__main__':
    main()
