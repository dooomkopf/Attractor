#!/usr/bin/env python3
"""CLI for amplitude-scaling diagnostics on Wang harmonics."""

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
from analyze_wang.phase import phase_lock_report
from analyze_wang.scaling import scaling_report
from analyze_wang.simulate import channel_map_with_pcs, simulate_trajectory
from analyze_LPPL.plot_utils import save_figure


CHANNEL_CHOICES = ['x', 'y', 'z', 'pc1', 'pc2', 'pc3']


def _parse_ic(text):
    parts = [float(x.strip()) for x in text.split(',')]
    if len(parts) != 3:
        raise ValueError('IC must have exactly 3 comma-separated floats')
    return tuple(parts)


def _print_report(sim, spec, phase, scaling, main_channel, harm_channel, harm_order):
    print("=" * 88)
    print("WANG 2-SCROLL SCALING READOUT")
    print("=" * 88)
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
    print()
    print("PHASE SANITY")
    print(f"  mean resultant R  : {phase['R']:.6f}")
    print(f"  median |delta phi|: {phase['median_abs_delta_deg']:.2f} deg")
    print()
    print("AMPLITUDE SCALING")
    print(f"  threshold main    : {scaling['threshold_main']:.3e}   [lower {int(100*0.2)}% clipped]")
    print(f"  kept samples      : {int(np.sum(scaling['mask']))}")
    print(f"  main mean/std/cv  : {scaling['main_mean']:.3e} / {scaling['main_std']:.3e} / {scaling['main_cv']:.3f}")
    print(f"  harm mean/std/cv  : {scaling['harm_mean']:.3e} / {scaling['harm_std']:.3e} / {scaling['harm_cv']:.3f}")
    print(f"  slope (A2w~c*A^2) : {scaling['slope_zero']:.6e}")
    print(f"  corr(A^2,A2w)     : {scaling['corr']:.6f}")
    print(f"  corr(centered)    : {scaling['corr_centered']:.6f}")
    print(f"  R2 through origin : {scaling['r2_zero']:.6f}")
    print(f"  identifiable      : {'yes' if scaling['scaling_identifiable'] else 'weak'}   [main-envelope CV >= 5%]")
    print("=" * 88)


def _plot_scaling(sim, phase, scaling, main_channel, harm_channel):
    t = np.asarray(sim['t'], dtype=float) - float(sim['t'][0])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(t, phase['main_envelope'], color='tab:blue', lw=1.0, label=f'{main_channel} envelope')
    ax.plot(t, phase['harm_envelope'], color='tab:red', lw=1.0, label=f'{harm_channel} envelope')
    ax.set_title('Bandpassed envelopes')
    ax.set_xlabel('time')
    ax.set_ylabel('envelope')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[1]
    ax.plot(t, scaling['y_harm'], color='tab:red', lw=1.0, label=f'observed {harm_channel} envelope')
    ax.plot(t, scaling['y_hat_full'], color='black', lw=1.0, label='predicted quadratic fit')
    ax.set_title('Observed vs predicted harmonic envelope')
    ax.set_xlabel('time')
    ax.set_ylabel('amplitude')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8)

    ax = axes[2]
    resid_full = scaling['y_harm'] - scaling['y_hat_full']
    resid_lim = max(float(np.quantile(np.abs(resid_full), 0.995)), 1e-9)
    resid_linthresh = max(0.05 * resid_lim, 1e-6)
    ax.plot(t, resid_full, color='tab:gray', lw=1.0)
    ax.axhline(0.0, color='black', lw=0.8)
    ax.set_ylim(-resid_lim, resid_lim)
    ax.set_yscale('symlog', linthresh=resid_linthresh)
    ax.set_title('Residual over time')
    ax.set_xlabel('time')
    ax.set_ylabel(f'{harm_channel} minus quadratic fit')
    ax.grid(True, alpha=0.25)
    phase_text = (
        f"Phase-lock:\n"
        f"R = {phase['R']:.3f}\n"
        f"median |dphi| = {phase['median_abs_delta_deg']:.1f} deg"
    )
    ax.text(
        0.03,
        0.97,
        phase_text,
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': '0.5'},
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description='Amplitude-scaling diagnostics on Wang harmonics')
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
    ap.add_argument('--main_quantile', type=float, default=0.20)
    ap.add_argument('--save', action='store_true', help='Save plot as PNG')
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True, save=True)
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
    scaling = scaling_report(
        main_envelope=phase['main_envelope'],
        harm_envelope=phase['harm_envelope'],
        min_main_quantile=args.main_quantile,
    )
    _print_report(sim, spec, phase, scaling, args.main_channel, args.harm_channel, args.harm_order)

    if args.show or args.save:
        fig = _plot_scaling(sim, phase, scaling, args.main_channel, args.harm_channel)
        if args.save:
            tags = {'main': args.main_channel, 'harm': args.harm_channel, 'ord': args.harm_order}
            filepath = save_figure(fig, '04_scaling', tags, output_dir=HERE)
            print(f"\nPlot saved: {filepath}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == '__main__':
    main()
