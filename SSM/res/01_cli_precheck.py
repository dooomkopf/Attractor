#!/usr/bin/env python3
"""01: Precheck — LPPL simulation, embedding, PCA, SSMLearn fit, eigenvalues, fit_err, ode_err."""

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
from lppl_system import DEFAULT_PARAMS, build_params
from SSM_res_lppl_data import (
    build_lppl_context, DEFAULT_M, DEFAULT_YEARS, DEFAULT_T_FINAL,
    DEFAULT_N_EVAL, DEFAULT_IC, DEFAULT_TRANSIENT_FRAC, DAYS_PER_YEAR,
)
from ssmlearn_res import fit_ssm


def main():
    ap = argparse.ArgumentParser(description='01: SSM Precheck on LPPL simulation')
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--t_final', type=float, default=DEFAULT_T_FINAL)
    ap.add_argument('--n_eval', type=int, default=DEFAULT_N_EVAL)
    ap.add_argument('--transient_frac', type=float, default=DEFAULT_TRANSIENT_FRAC)
    ap.add_argument('--ssm_dim', type=int, default=2, metavar='INT')
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2, metavar='POLY')
    ap.add_argument('--wang-off', action='store_true',
                    help='quasi-2D: Z_A=0, Z_B=0, Z_MIX=8e-5')
    ap.add_argument('--scan_ssm_dim', type=str, default='', metavar='CSV')
    ap.add_argument('--scan_poly', type=str, default='', metavar='CSV')
    args = ap.parse_args()

    if bool(args.scan_ssm_dim) != bool(args.scan_poly):
        ap.error('--scan_ssm_dim and --scan_poly must be provided together')

    p = build_params(wang_off=args.wang_off)
    mode_tag = "QUASI-2D" if args.wang_off else "FULL 3D"

    if args.scan_ssm_dim:
        dims = [int(x) for x in args.scan_ssm_dim.split(',')]
        polys = [int(x) for x in args.scan_poly.split(',')]
        ctx, tau, pca_res, _, _ = build_lppl_context(
            p, args.M, args.years, args.t_final, args.n_eval,
            transient_frac=args.transient_frac)
        print("=" * 100)
        print(f"LPPL PRECHECK SCAN — {mode_tag}")
        print("=" * 100)
        print(f"  {'dim':>3}  {'poly':>4}  {'fit_err':>8}  {'pairs':>5}  {'T_main':>8}  {'T_sub':>8}")
        print("  " + "-" * 90)
        for dim in dims:
            for poly in polys:
                res = fit_ssm(ctx, dim, poly_degree=poly, compute_prediction=False)
                coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
                eigvals = np.linalg.eigvals(coeffs[:, :dim])
                osc = sorted([ev for ev in eigvals if ev.imag > 1e-6], key=lambda z: abs(z.imag))
                D_obs = ctx['D_c'].T
                fe = float(np.linalg.norm(D_obs - res['pred_obs']) / (np.linalg.norm(D_obs) + 1e-30))
                T_main = f"{(2*np.pi/abs(osc[0].imag))/DAYS_PER_YEAR:.2f}y" if osc else "-"
                T_sub = f"{(2*np.pi/abs(osc[1].imag))/DAYS_PER_YEAR:.2f}y" if len(osc) >= 2 else "-"
                print(f"  {dim:>3}  {poly:>4}  {fe:>8.3f}  {len(osc):>5}  {T_main:>8}  {T_sub:>8}")
        print("=" * 100)
        return

    ctx, tau, pca_res, t, traj = build_lppl_context(
        p, args.M, args.years, args.t_final, args.n_eval,
        transient_frac=args.transient_frac)

    res = fit_ssm(ctx, args.ssm_dim, poly_degree=args.poly_degree, compute_prediction=False)
    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :args.ssm_dim]
    eigvals = np.linalg.eigvals(linear_part)

    D_obs = ctx['D_c'].T
    diff = D_obs - res['pred_obs']
    fit_err = float(np.linalg.norm(diff) / (np.linalg.norm(D_obs) + 1e-30))

    dp_dt = np.gradient(ctx['pc'][:, :args.ssm_dim], ctx['days_vecs'].astype(float), axis=0)
    rhs = res['ssm'].reduced_dynamics.predict(ctx['pc'][:, :args.ssm_dim])
    edge = max(5, ctx['N'] // 100)
    sl = slice(edge, -edge)
    ode_err = float(np.linalg.norm(dp_dt[sl] - rhs[sl]) / (np.linalg.norm(dp_dt[sl]) + 1e-30))

    cum_var = float(np.cumsum(pca_res.var)[args.ssm_dim - 1])
    osc = sorted([ev for ev in eigvals if ev.imag > 1e-6], key=lambda z: abs(z.imag))

    print("=" * 84)
    print(f"LPPL SIMULATION PRECHECK — {mode_tag}")
    print("=" * 84)
    print("SIMULATION")
    print(f"  t_final       : {args.t_final:.0f}d")
    print(f"  n_eval        : {args.n_eval}")
    print(f"  y1 range      : [{traj[:,0].min():.6f}, {traj[:,0].max():.6f}]")
    print(f"  Wang coupling : {'OFF (Z_A=0, Z_B=0, Z_MIX=8e-5)' if args.wang_off else 'ON (full)'}")
    print()
    print("EMBEDDING")
    print(f"  M             : {args.M}")
    print(f"  years         : {args.years:.2f}")
    print(f"  tau           : {tau}d")
    print(f"  N_vec         : {ctx['N']}")
    print()
    print("LEARNED / AUS FIT")
    print(f"  cum var       : {100*cum_var:.2f}%")
    print(f"  fit_err       : {fit_err:.4e}")
    print(f"  ode_err       : {ode_err:.4e}")
    print(f"  osc pairs     : {len(osc)}")
    if osc:
        for k, ev in enumerate(osc):
            T_y = (2*np.pi / abs(ev.imag)) / DAYS_PER_YEAR
            print(f"    [{k}] Re={ev.real:+.4e}  Im={ev.imag:+.4e}  T={T_y:.3f}y")
    if len(osc) >= 2:
        T_main = (2*np.pi / abs(osc[0].imag)) / DAYS_PER_YEAR
        T_sub = (2*np.pi / abs(osc[1].imag)) / DAYS_PER_YEAR
        print(f"  2:1 check     : T_main={T_main:.3f}y  T_sub={T_sub:.3f}y  ratio={T_sub/(T_main/2):.3f}")
    print("=" * 84)


if __name__ == '__main__':
    main()
