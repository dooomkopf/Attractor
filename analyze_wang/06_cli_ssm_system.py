#!/usr/bin/env python3
"""CLI: display the shifted first-order form at a scroll center."""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_wang.constants import DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D
from analyze_wang.precheck import run_precheck
from analyze_wang.ssm_system import shifted_first_order, verify_shifted_ode, format_system


def main():
    ap = argparse.ArgumentParser(description='Shifted first-order form at scroll center')
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

    print("=" * 72)
    for line in format_system(sys_data, label=args.eq):
        print(line)
    print()

    vfy = verify_shifted_ode(sys_data)
    print(f"  VERIFICATION (random perturbation ||dz||={float(sum(x**2 for x in vfy['dz']))**0.5:.4f}):")
    print(f"    rhs_original = [{', '.join(f'{v:+.8f}' for v in vfy['rhs_original'])}]")
    print(f"    rhs_shifted  = [{', '.join(f'{v:+.8f}' for v in vfy['rhs_shifted'])}]")
    print(f"    error        = {vfy['error']:.2e}")
    print("=" * 72)


if __name__ == '__main__':
    main()
