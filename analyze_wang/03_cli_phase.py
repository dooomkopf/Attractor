#!/usr/bin/env python3
"""CLI for phase-lock diagnostics on Wang 2-scroll harmonics."""

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
from analyze_wang.phase import phase_lock_report, wrap_to_pi
from analyze_wang.simulate import channel_map_with_pcs, simulate_trajectory


CHANNEL_CHOICES = ['x', 'y', 'z', 'pc1', 'pc2', 'pc3']


def _parse_ic(text):
    parts = [float(x.strip()) for x in text.split(',')]
    if len(parts) != 3:
        raise ValueError('IC must have exactly 3 comma-separated floats')
    return tuple(parts)


def _print_report(sim, spec, phase, main_channel, harm_channel, harm_order):
    print("=" * 84)
    print("WANG 2-SCROLL PHASE READOUT")
    print("=" * 84)
    print("SIMULATION")
    print(f"  kept samples      : {len(sim['t'])}")
    print(f"  dt                : {sim['dt']:.6f}")
    print(f"  kept span         : {sim['t'][-1] - sim['t'][0]:.3f}")
    print()
    print("CHANNEL CHOICE")
    print(f"  main channel      : {main_channel}")
    print(f"  harmonic channel  : {harm_channel}")
    print(f"  harmonic order    : {harm_order}")
    print(f"  main f0           : {spec['channels'][main_channel]['dominant_freq']:.6f}")
    print(f"  target harm freq  : {harm_order * spec['channels'][main_channel]['dominant_freq']:.6f}")
    print()
    print("PHASE LOCK")
    print(f"  mean resultant R  : {phase['R']:.6f}   [1=locked, 0=uniform]")
    print(f"  mean delta phi    : {phase['mean_angle_deg']:.2f} deg")
    print(f"  median |delta phi|: {phase['median_abs_delta_deg']:.2f} deg")
    print("=" * 84)


def _plot_phase(sim, phase, main_channel, harm_channel, harm_order, f0, zoom_cycles):
    t = np.asarray(sim['t'], dtype=float) - float(sim['t'][0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(t, phase['main_envelope'], label=f'{main_channel} envelope', color='tab:blue', lw=1.0)
    ax.plot(t, phase['harm_envelope'], label=f'{harm_channel} envelope', color='tab:red', lw=1.0)
    ax.set_title('Bandpassed envelopes')
    ax.set_xlabel('time')
    ax.set_ylabel('envelope')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[0, 1]
    main_wrapped = np.angle(np.exp(1j * phase['main_phase']))
    harm_wrapped = np.angle(np.exp(1j * phase['harm_phase'] / harm_order))
    if f0 is not None and f0 > 0.0:
        zoom_span = float(zoom_cycles) / float(f0)
        mask_zoom = t <= zoom_span
    else:
        zoom_span = float(t[-1] - t[0])
        mask_zoom = np.ones_like(t, dtype=bool)
    ax.plot(t[mask_zoom], main_wrapped[mask_zoom], color='tab:blue', lw=0.9, label=main_channel)
    ax.plot(t[mask_zoom], harm_wrapped[mask_zoom], color='tab:red', lw=0.9, label=f'{harm_channel}/{harm_order}')
    ax.set_title(f'Phases over first {zoom_cycles:g} main cycles')
    ax.set_xlabel('time')
    ax.set_ylabel('phase [rad]')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[1, 0]
    delta_deg = np.degrees(phase['delta_phase'])
    mean_deg = float(phase['mean_angle_deg'])
    ax.plot(t, delta_deg, color='tab:green', lw=0.9)
    ax.axhline(mean_deg, color='gray', ls='--', lw=0.9)
    ax.set_ylim(mean_deg - 10.0, mean_deg + 10.0)
    ax.set_title(f'$\\Delta\\phi$ over time around mean ({mean_deg:.1f} deg)')
    ax.set_xlabel('time')
    ax.set_ylabel('$\\Delta\\phi$ [deg]')
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    ax.remove()
    ax = fig.add_subplot(2, 2, 4, projection='polar')
    ax.hist(phase['delta_phase'], bins=36, color='tab:green', alpha=0.75)
    ax.set_title('Polar histogram of phase combination')
    return fig


def main():
    ap = argparse.ArgumentParser(description='Phase-lock diagnostics on Wang 2-scroll harmonics')
    ap.add_argument('--a', type=float, default=DEFAULT_A)
    ap.add_argument('--b', type=float, default=DEFAULT_B)
    ap.add_argument('--c', type=float, default=DEFAULT_C)
    ap.add_argument('--d', type=float, default=DEFAULT_D)
    ap.add_argument('--ic', type=str, default=",".join(str(v) for v in DEFAULT_IC))
    ap.add_argument('--t_final', type=float, default=150.0)
    ap.add_argument('--n_eval', type=int, default=75000)
    ap.add_argument('--transient_frac', type=float, default=DEFAULT_TRANSIENT_FRAC)
    ap.add_argument('--main_channel', type=str, default='pc1', choices=CHANNEL_CHOICES)
    ap.add_argument('--harm_channel', type=str, default='pc3', choices=CHANNEL_CHOICES)
    ap.add_argument('--harm_order', type=int, default=2)
    ap.add_argument('--min_freq', type=float, default=0.01)
    ap.add_argument('--harmonic_window', type=float, default=0.15)
    ap.add_argument('--phase_bandwidth', type=float, default=0.20)
    ap.add_argument('--phase_zoom_cycles', type=float, default=10.0, help='cycles shown in wrapped-phase panel')
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true', help='open matplotlib plots (default)')
    show_group.add_argument('--no-show', dest='show', action='store_false', help='skip matplotlib plots')
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
    channels = channel_map_with_pcs(sim['traj'])
    channel_map = {k: channels[k] for k in CHANNEL_CHOICES}
    spec = analyze_channels(
        channel_map,
        fs=1.0 / sim['dt'],
        reference_key=args.main_channel,
        min_freq=args.min_freq,
        rel_window=args.harmonic_window,
    )
    f0 = spec['channels'][args.main_channel]['dominant_freq']
    phase = phase_lock_report(
        channel_map[args.main_channel],
        channel_map[args.harm_channel],
        fs=1.0 / sim['dt'],
        f0=f0,
        harm_order=args.harm_order,
        rel_bw=args.phase_bandwidth,
    )
    _print_report(sim, spec, phase, args.main_channel, args.harm_channel, args.harm_order)

    if args.show:
        _plot_phase(
            sim,
            phase,
            args.main_channel,
            args.harm_channel,
            args.harm_order,
            f0=f0,
            zoom_cycles=args.phase_zoom_cycles,
        )
        plt.show()


if __name__ == '__main__':
    main()
