#!/usr/bin/env python3
"""CLI: LPPL reduced dynamics polar form + backbone."""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_residuals.lppl_system import (
    DEFAULT_PARAMS, find_equilibria, shifted_first_order,
)
from analyze_wang.ssm_spectral import spectral_analysis, choose_E
from analyze_wang.ssm_whisker import compute_whisker_order2
from analyze_wang.ssm_backbone import polar_reduced_dynamics, format_backbone


def main():
    ap = argparse.ArgumentParser(description='LPPL reduced dynamics polar form + backbone')
    for k, v in DEFAULT_PARAMS.items():
        ap.add_argument(f'--{k}', type=float, default=v)
    ap.add_argument('--eq', type=str, default='E2', help='equilibrium label (default: E2)')
    args = ap.parse_args()

    p = {k: getattr(args, k) for k in DEFAULT_PARAMS}
    eqs = find_equilibria(p)
    eq_point = None
    for label, pt in eqs:
        if label == args.eq:
            eq_point = pt
            break
    if eq_point is None:
        print(f"ERROR: equilibrium {args.eq} not found")
        sys.exit(1)

    sys_data = shifted_first_order(eq_point, p)
    spec = spectral_analysis(sys_data['A'])
    E = choose_E(spec)
    whisker = compute_whisker_order2(sys_data, E)
    polar = polar_reduced_dynamics(E, whisker)

    print("=" * 84)
    print(f"LPPL SSM BACKBONE AT {args.eq}: ({eq_point[0]:+.6f}, {eq_point[1]:+.6f}, {eq_point[2]:+.6f})")
    print("=" * 84)
    for line in format_backbone(polar):
        print(line)
    print("=" * 84)


if __name__ == '__main__':
    main()
