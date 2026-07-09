#!/usr/bin/env python3
"""CLI entry point for the modular residual precheck."""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_residuals.constants import (
    DEFAULT_CYCLES_JSON,
    DEFAULT_FILENAME,
    DEFAULT_M,
    DEFAULT_YEARS,
    START_IDX,
)
from analyze_residuals.precheck import run_precheck, run_precheck_scan


def main():
    ap = argparse.ArgumentParser(description='Residual precheck for harmonic BTC analysis')
    ap.add_argument('--filename', type=str, default=DEFAULT_FILENAME, help='input CSV')
    ap.add_argument('--M', type=int, default=DEFAULT_M, help='embedding dimension')
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS, help='embedding window in years')
    ap.add_argument('--start_idx', type=int, default=START_IDX, help='row offset in CSV')
    ap.add_argument('--ssm_dim', type=int, default=2, metavar='INT', help='reduced model dimension')
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2, metavar='POLY', help='polynomial order')
    ap.add_argument('--time_mode', choices=['linear', 'log'], default='linear', help='analysis clock for reduced dynamics and d/dt estimates')
    ap.add_argument('--res_tol', type=float, default=0.05, help='relative resonance tolerance')
    ap.add_argument('--cycles_json', type=str, default=DEFAULT_CYCLES_JSON, help='halving/cycle boundary JSON')
    ap.add_argument('--loc', action='store_true',
                    help='local readout for H2-H3, H3-H4, and H4+ using the global fit')
    ap.add_argument('--scan_ssm_dim', type=str, default='', metavar='CSV', help='comma-separated ssm_dim values')
    ap.add_argument('--scan_poly', type=str, default='', metavar='CSV', help='comma-separated poly values')
    args = ap.parse_args()
    if args.scan_ssm_dim or args.scan_poly:
        if not (args.scan_ssm_dim and args.scan_poly):
            ap.error('--scan_ssm_dim and --scan_poly must be provided together')
        run_precheck_scan(args)
    else:
        run_precheck(args)


if __name__ == '__main__':
    main()
