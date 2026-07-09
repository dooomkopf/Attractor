#!/usr/bin/env python3
"""CLI for Wang harmonic parameter scans."""

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_wang.constants import DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D, DEFAULT_IC, DEFAULT_TRANSIENT_FRAC
from analyze_wang.scan import run_b_scan


def _parse_ic(text):
    parts = [float(x.strip()) for x in text.split(',')]
    if len(parts) != 3:
        raise ValueError('IC must have exactly 3 comma-separated floats')
    return tuple(parts)


def _print_table(rows):
    print("=" * 121)
    print("WANG HARMONIC B-SCAN")
    print("=" * 121)
    print("b        f0_x      z_ratio    xz_R      xz_med|dphi|    pc3_ratio   pc_R      pc_med|dphi|")
    print("-" * 121)
    for row in rows:
        print(
            f"{row['b']:7.3f}  "
            f"{row['f0_x']:8.5f}  "
            f"{row['z_harm_ratio']:9.3f}  "
            f"{row['xz_R']:8.5f}  "
            f"{row['xz_med_abs_deg']:14.2f}  "
            f"{row['pc3_harm_ratio']:10.3f}  "
            f"{row['pc_R']:8.5f}  "
            f"{row['pc_med_abs_deg']:14.2f}"
        )
    print("=" * 121)


def _plot(rows):
    b = np.array([row['b'] for row in rows], dtype=float)
    f0_x = np.array([row['f0_x'] for row in rows], dtype=float)
    z_ratio = np.array([row['z_harm_ratio'] for row in rows], dtype=float)
    pc3_ratio = np.array([row['pc3_harm_ratio'] for row in rows], dtype=float)
    xz_R = np.array([row['xz_R'] for row in rows], dtype=float)
    pc_R = np.array([row['pc_R'] for row in rows], dtype=float)
    xz_med = np.array([row['xz_med_abs_deg'] for row in rows], dtype=float)
    pc_med = np.array([row['pc_med_abs_deg'] for row in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(b, f0_x, marker='o', color='tab:blue')
    ax.set_title('Fundamental frequency from x')
    ax.set_xlabel('b')
    ax.set_ylabel('f0')
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(b, z_ratio, marker='o', color='tab:red', label='z harmonic ratio')
    ax.plot(b, pc3_ratio, marker='o', color='tab:purple', label='pc3 harmonic ratio')
    ax.set_title('2w strength')
    ax.set_xlabel('b')
    ax.set_ylabel('PSD(2w)/PSD(w)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[1, 0]
    ax.plot(b, xz_R, marker='o', color='tab:green', label='x -> z')
    ax.plot(b, pc_R, marker='o', color='tab:orange', label='pc1 -> pc3')
    ax.set_title('Phase-lock strength R')
    ax.set_xlabel('b')
    ax.set_ylabel('R')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[1, 1]
    ax.plot(b, xz_med, marker='o', color='tab:green', label='x -> z')
    ax.plot(b, pc_med, marker='o', color='tab:orange', label='pc1 -> pc3')
    ax.set_title('Median |delta phi|')
    ax.set_xlabel('b')
    ax.set_ylabel('deg')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)
    return fig


def main():
    ap = argparse.ArgumentParser(description='Scan Wang parameter b for harmonic diagnostics')
    ap.add_argument('--a', type=float, default=DEFAULT_A)
    ap.add_argument('--b_min', type=float, default=5.6)
    ap.add_argument('--b_max', type=float, default=7.4)
    ap.add_argument('--n_b', type=int, default=10)
    ap.add_argument('--c', type=float, default=DEFAULT_C)
    ap.add_argument('--d', type=float, default=DEFAULT_D)
    ap.add_argument('--ic', type=str, default=",".join(str(v) for v in DEFAULT_IC))
    ap.add_argument('--t_final', type=float, default=120.0)
    ap.add_argument('--n_eval', type=int, default=50000)
    ap.add_argument('--transient_frac', type=float, default=DEFAULT_TRANSIENT_FRAC)
    ap.add_argument('--min_freq', type=float, default=0.01)
    ap.add_argument('--harmonic_window', type=float, default=0.15)
    ap.add_argument('--phase_bandwidth', type=float, default=0.20)
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true', help='open matplotlib plots (default)')
    show_group.add_argument('--no-show', dest='show', action='store_false', help='skip matplotlib plots')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    b_values = np.linspace(args.b_min, args.b_max, args.n_b)
    rows = run_b_scan(
        b_values,
        a=args.a,
        c=args.c,
        d=args.d,
        ic=_parse_ic(args.ic),
        t_final=args.t_final,
        n_eval=args.n_eval,
        transient_frac=args.transient_frac,
        min_freq=args.min_freq,
        harmonic_window=args.harmonic_window,
        phase_bandwidth=args.phase_bandwidth,
    )
    _print_table(rows)
    if args.show:
        _plot(rows)
        plt.show()


if __name__ == '__main__':
    main()
