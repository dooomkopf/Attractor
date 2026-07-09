#!/usr/bin/env python3
"""07: Spectral screening of fitted linear dynamics on LPPL simulation."""

import argparse
import os
import sys
import logging
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
warnings.filterwarnings("ignore")
logging.getLogger('ridge_regression').setLevel(logging.WARNING)
logging.getLogger('SSMLearn').setLevel(logging.WARNING)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ATTRACTOR = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _ATTRACTOR, os.path.join(_ATTRACTOR, 'SSMLearnPy')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from lppl_system import build_params
from SSM_res_lppl_data import (
    build_lppl_context, DEFAULT_M, DEFAULT_YEARS, DEFAULT_T_FINAL,
    DEFAULT_N_EVAL, DAYS_PER_YEAR,
)
from ssmlearn_res import fit_ssm
from analyze_wang.ssm_spectral import spectral_analysis, choose_E, format_spectral


def main():
    ap = argparse.ArgumentParser(description='07: Spectral screen (resonance + spectral gap) on LPPL simulation')
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--t_final', type=float, default=DEFAULT_T_FINAL)
    ap.add_argument('--n_eval', type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--ssm_dim', type=int, default=2, metavar='INT', help='reduced model dimension')
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2, metavar='POLY', help='polynomial order')
    ap.add_argument('--wang-off', action='store_true',
                    help='quasi-2D: Z_A=0, Z_B=0, Z_MIX=8e-5')
    args = ap.parse_args()

    p = build_params(wang_off=args.wang_off)
    mode_tag = "QUASI-2D" if args.wang_off else "FULL 3D"

    ctx, tau, pca_res, t, traj = build_lppl_context(
        p, args.M, args.years, args.t_final, args.n_eval)

    res = fit_ssm(ctx, args.ssm_dim, poly_degree=args.poly_degree, compute_prediction=False)
    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :args.ssm_dim]

    dt = float(np.median(np.diff(ctx['days_vecs'])))

    spec = spectral_analysis(linear_part)
    try:
        E = choose_E(spec)
        print("=" * 84)
        print(f"LPPL SIMULATION SPECTRAL READOUT -- {mode_tag}")
        print("=" * 84)
        print("DATA")
        print(f"  M                 : {args.M}")
        print(f"  years             : {args.years:.2f}")
        print(f"  tau               : {tau}d")
        print(f"  kept vectors      : {ctx['N']}")
        print(f"  dt                : {dt:.6f}d")
        print()
        print("MODEL CHOICE")
        print(f"  ssm_dim           : {args.ssm_dim}")
        print(f"  poly              : {args.poly_degree}")
        print()
        print("SPECTRAL ANALYSIS")
        for line in format_spectral(spec, E, eq_label='fitted'):
            print(line)
        print("=" * 84)
    except RuntimeError as e:
        print("=" * 84)
        print(f"LPPL SIMULATION SPECTRAL READOUT -- {mode_tag}")
        print("=" * 84)
        print("DATA")
        print(f"  M                 : {args.M}")
        print(f"  years             : {args.years:.2f}")
        print(f"  tau               : {tau}d")
        print(f"  kept vectors      : {ctx['N']}")
        print(f"  dt                : {dt:.6f}d")
        print()
        print("MODEL CHOICE")
        print(f"  ssm_dim           : {args.ssm_dim}")
        print(f"  poly              : {args.poly_degree}")
        print()
        print("SPECTRAL ANALYSIS")
        print(f"  choose_E failed: {e}")
        print("  eigenvalues:")
        for k, ev in enumerate(spec['Lambda']):
            if abs(ev.imag) > 1e-6:
                T_y = (2*np.pi / abs(ev.imag)) / DAYS_PER_YEAR
                print(f"    [{k}] Re={ev.real:+.4e}  Im={ev.imag:+.4e}  T={T_y:.3f}y")
            else:
                print(f"    [{k}] Re={ev.real:+.4e} (real)")
        print("=" * 84)


if __name__ == '__main__':
    main()
