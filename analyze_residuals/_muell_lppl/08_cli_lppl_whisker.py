#!/usr/bin/env python3
"""CLI: LPPL order-2 SSM whisker computation (cohomological equation)."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_residuals.lppl_system import (
    DEFAULT_PARAMS, find_equilibria, shifted_first_order,
)
from analyze_wang.ssm_spectral import spectral_analysis, choose_E
from analyze_wang.ssm_whisker import compute_whisker_order2, format_whisker


def main():
    ap = argparse.ArgumentParser(description='LPPL order-2 SSM whisker computation')
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

    print("=" * 84)
    print(f"LPPL SSM WHISKER ORDER 2 AT {args.eq}: ({eq_point[0]:+.6f}, {eq_point[1]:+.6f}, {eq_point[2]:+.6f})")
    print("=" * 84)
    for line in format_whisker(whisker, E['Lambda_E']):
        print(line)
    print("=" * 84)


if __name__ == '__main__':
    main()
