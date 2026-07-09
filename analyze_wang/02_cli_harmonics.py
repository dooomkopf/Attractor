#!/usr/bin/env python3
"""CLI for harmonic readout on simulated Wang 2-scroll trajectories."""

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
from analyze_wang.harmonics import analyze_channels
from analyze_wang.simulate import principal_coordinate, simulate_trajectory


def _parse_ic(text):
    parts = [float(x.strip()) for x in text.split(',')]
    if len(parts) != 3:
        raise ValueError('IC must have exactly 3 comma-separated floats')
    return tuple(parts)


def _print_report(sim, spec):
    print("=" * 80)
    print("WANG 2-SCROLL HARMONIC READOUT")
    print("=" * 80)
    print("SIMULATION")
    print(f"  kept samples      : {len(sim['t'])}")
    print(f"  dt                : {sim['dt']:.6f}")
    print(f"  kept span         : {sim['t'][-1] - sim['t'][0]:.3f}")
    print(f"  reference channel : {spec['reference_key']}")
    if spec['reference_freq'] is not None:
        print(f"  reference f0      : {spec['reference_freq']:.6f}   T={spec['reference_period']:.3f}")
    else:
        print("  reference f0      : unavailable")
    print()
    print("CHANNELS")
    for key in ['x', 'y', 'z', 'pc1', 'pc2', 'pc3']:
        row = spec['channels'][key]
        f0 = '-' if row['dominant_freq'] is None else f"{row['dominant_freq']:.6f}"
        t0 = '-' if row['dominant_period'] is None else f"{row['dominant_period']:.3f}"
        fh = '-' if row['harmonic_freq'] is None else f"{row['harmonic_freq']:.6f}"
        ratio = '-' if row['harmonic_ratio'] is None else f"{row['harmonic_ratio']:.3e}"
        print(
            f"  {key:>4s}: f_dom={f0:>9}  T_dom={t0:>8}  "
            f"f_2w≈{fh:>9}  PSD(2w)/PSD(w)={ratio:>7}"
        )
    print("=" * 80)


def _plot_spectra(spec):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes = axes.ravel()
    for ax, key in zip(axes, ['x', 'y', 'z', 'pc1', 'pc2', 'pc3']):
        row = spec['channels'][key]
        ax.plot(row['freq_grid'], row['psd_grid'], lw=1.2)
        if row['dominant_freq'] is not None:
            ax.axvline(row['dominant_freq'], color='tab:blue', lw=0.8, alpha=0.8)
        if row['harmonic_target'] is not None:
            ax.axvline(row['harmonic_target'], color='tab:red', lw=0.8, ls='--', alpha=0.8)
        if row['harmonic_freq'] is not None:
            ax.axvline(row['harmonic_freq'], color='tab:orange', lw=0.8, alpha=0.8)
        ax.set_title(key)
        ax.set_xlabel('frequency')
        ax.set_ylabel('PSD')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.25)
    return fig


def main():
    ap = argparse.ArgumentParser(description='Harmonic readout on simulated Wang 2-scroll attractor')
    ap.add_argument('--a', type=float, default=DEFAULT_A)
    ap.add_argument('--b', type=float, default=DEFAULT_B)
    ap.add_argument('--c', type=float, default=DEFAULT_C)
    ap.add_argument('--d', type=float, default=DEFAULT_D)
    ap.add_argument('--ic', type=str, default=",".join(str(v) for v in DEFAULT_IC))
    ap.add_argument('--t_final', type=float, default=150.0)
    ap.add_argument('--n_eval', type=int, default=75000)
    ap.add_argument('--transient_frac', type=float, default=DEFAULT_TRANSIENT_FRAC)
    ap.add_argument('--reference', type=str, default='pc1', choices=['x', 'y', 'z', 'pc1'])
    ap.add_argument('--min_freq', type=float, default=0.01)
    ap.add_argument('--harmonic_window', type=float, default=0.15)
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true',
                            help='open matplotlib plots (default)')
    show_group.add_argument('--no-show', dest='show', action='store_false',
                            help='skip matplotlib plots')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    sim = simulate_trajectory(
        a=args.a,
        b=args.b,
        c=args.c,
        d=args.d,
        ic=_parse_ic(args.ic),
        t_final=args.t_final,
        n_eval=args.n_eval,
        transient_frac=args.transient_frac,
    )
    pcs = principal_coordinate(sim['traj'])
    channel_map = {
        'x': sim['traj'][:, 0],
        'y': sim['traj'][:, 1],
        'z': sim['traj'][:, 2],
        'pc1': pcs['pc'][:, 0],
        'pc2': pcs['pc'][:, 1],
        'pc3': pcs['pc'][:, 2],
    }
    spec = analyze_channels(
        channel_map,
        fs=1.0 / sim['dt'],
        reference_key=args.reference,
        min_freq=args.min_freq,
        rel_window=args.harmonic_window,
    )
    _print_report(sim, spec)

    if args.show:
        _plot_spectra(spec)
        plt.show()


if __name__ == '__main__':
    main()
