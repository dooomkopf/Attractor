#!/usr/bin/env python3
"""CLI: spectral analysis and master/slave mode selection at a scroll center."""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_wang.constants import DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D
from analyze_wang.precheck import run_precheck
from analyze_wang.ssm_system import shifted_first_order
from analyze_wang.ssm_spectral import spectral_analysis, choose_E, format_spectral


def main():
    ap = argparse.ArgumentParser(description='Spectral analysis + master/slave selection at scroll center')
    ap.add_argument('--a', type=float, default=DEFAULT_A)
    ap.add_argument('--b', type=float, default=DEFAULT_B)
    ap.add_argument('--c', type=float, default=DEFAULT_C)
    ap.add_argument('--d', type=float, default=DEFAULT_D)
    ap.add_argument('--eq', type=str, default='S4', help='equilibrium label (default: S4, scroll center)')
    args = ap.parse_args()

    result = run_precheck(args, verbose=False)
    eq_row = None
    for row in result['equilibria']:
        if row['label'] == args.eq:
            eq_row = row
            break
    if eq_row is None:
        print(f"ERROR: equilibrium {args.eq} not found")
        sys.exit(1)

    sys_data = shifted_first_order(eq_row['point'], args.a, args.b, args.c, args.d)
    spec = spectral_analysis(sys_data['A'])
    E = choose_E(spec)

    print("=" * 84)
    for line in format_spectral(spec, E, eq_label=args.eq):
        print(line)
    print("=" * 84)


if __name__ == '__main__':
    main()
