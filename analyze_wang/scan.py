"""Parameter-scan helpers for Wang harmonic diagnostics."""

import numpy as np

from analyze_wang.harmonics import analyze_channels
from analyze_wang.phase import phase_lock_report
from analyze_wang.simulate import channel_map_with_pcs, simulate_trajectory


def run_b_scan(
    b_values,
    *,
    a,
    c,
    d,
    ic,
    t_final,
    n_eval,
    transient_frac,
    min_freq,
    harmonic_window,
    phase_bandwidth,
):
    nan_row = lambda b_val: {
        'b': float(b_val), 'f0_x': float('nan'), 'f0_pc1': float('nan'),
        'z_harm_ratio': float('nan'), 'pc3_harm_ratio': float('nan'),
        'xz_R': float('nan'), 'pc_R': float('nan'),
        'xz_med_abs_deg': float('nan'), 'pc_med_abs_deg': float('nan'),
    }
    rows = []
    for b in np.asarray(b_values, dtype=float):
        try:
            sim = simulate_trajectory(
                a=a,
                b=float(b),
                c=c,
                d=d,
                ic=ic,
                t_final=t_final,
                n_eval=n_eval,
                transient_frac=transient_frac,
            )
            channels = channel_map_with_pcs(sim['traj'])
            channel_map = {k: channels[k] for k in ('x', 'y', 'z', 'pc1', 'pc2', 'pc3')}
            fs = 1.0 / sim['dt']

            spec_x = analyze_channels(
                channel_map,
                fs=fs,
                reference_key='x',
                min_freq=min_freq,
                rel_window=harmonic_window,
            )
            spec_pc = analyze_channels(
                channel_map,
                fs=fs,
                reference_key='pc1',
                min_freq=min_freq,
                rel_window=harmonic_window,
            )

            f0_x = spec_x['channels']['x']['dominant_freq']
            f0_pc = spec_pc['channels']['pc1']['dominant_freq']
            phase_xz = phase_lock_report(
                channel_map['x'],
                channel_map['z'],
                fs=fs,
                f0=f0_x,
                harm_order=2,
                rel_bw=phase_bandwidth,
            )
            phase_pc = phase_lock_report(
                channel_map['pc1'],
                channel_map['pc3'],
                fs=fs,
                f0=f0_pc,
                harm_order=2,
                rel_bw=phase_bandwidth,
            )

            rows.append(
                {
                    'b': float(b),
                    'f0_x': f0_x,
                    'f0_pc1': f0_pc,
                    'z_harm_ratio': spec_x['channels']['z']['harmonic_ratio'],
                    'pc3_harm_ratio': spec_pc['channels']['pc3']['harmonic_ratio'],
                    'xz_R': phase_xz['R'],
                    'pc_R': phase_pc['R'],
                    'xz_med_abs_deg': phase_xz['median_abs_delta_deg'],
                    'pc_med_abs_deg': phase_pc['median_abs_delta_deg'],
                }
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"  [SKIP] b={float(b):.3f}: {exc}")
            rows.append(nan_row(b))
    return rows
