#!/usr/bin/env python3
"""CLI entry point for Wang 2-scroll model prechecks."""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_wang.constants import DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D, DEFAULT_IC
from analyze_wang.precheck import run_precheck


def _parse_ic(text):
    parts = [float(x.strip()) for x in text.split(',')]
    if len(parts) != 3:
        raise ValueError('IC must have exactly 3 comma-separated floats')
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser(description='SSMtool-style Wang 2-scroll precheck')
    ap.add_argument('--a', type=float, default=DEFAULT_A)
    ap.add_argument('--b', type=float, default=DEFAULT_B)
    ap.add_argument('--c', type=float, default=DEFAULT_C)
    ap.add_argument('--d', type=float, default=DEFAULT_D)
    ap.add_argument('--ic', type=str, default=','.join(str(v) for v in DEFAULT_IC))
    args = ap.parse_args()
    args.ic = _parse_ic(args.ic)
    run_precheck(args)


if __name__ == '__main__':
    main()
