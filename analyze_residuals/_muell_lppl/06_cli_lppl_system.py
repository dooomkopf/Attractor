#!/usr/bin/env python3
"""CLI: LPPL model precheck — equilibria, eigenvalues, V2.6 gates (analog to Wang precheck)."""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from analyze_residuals.lppl_system import (
    DEFAULT_PARAMS, find_equilibria, lppl_jacobian,
    shifted_first_order, verify_shifted_ode,
)

IMAG_TOL = 1e-6
REAL_TOL = 1e-8


def _format_eig(ev):
    txt = f"{ev.real:+.6f}"
    if abs(ev.imag) > IMAG_TOL:
        txt += f" {'+' if ev.imag >= 0 else '-'} i*{abs(ev.imag):.6f}"
        T = 2.0 * np.pi / abs(ev.imag)
        txt += f"  T={T:.4f}"
    return txt


def _classify(eigvals):
    pos = int(np.sum(np.real(eigvals) > REAL_TOL))
    neg = int(np.sum(np.real(eigvals) < -REAL_TOL))
    osc = int(np.sum(np.imag(eigvals) > IMAG_TOL))
    stable = bool(np.all(np.real(eigvals) < -REAL_TOL))
    if osc >= 1 and stable:
        return 'stable_focus'
    elif osc >= 1 and pos > 0:
        return 'saddle_focus'
    elif pos > 0 and neg > 0:
        return 'saddle'
    elif neg == len(eigvals):
        return 'stable_node'
    else:
        return 'other'


def _spectral_gap(eigvals):
    pos_imag = [k for k, ev in enumerate(eigvals) if ev.imag > IMAG_TOL]
    if not pos_imag:
        return None
    conj = set()
    for k in pos_imag:
        for j in range(len(eigvals)):
            if j != k and abs(eigvals[j] - np.conj(eigvals[k])) < 1e-8:
                conj.add(j)
    master = sorted(set(pos_imag) | conj)
    slave = [k for k in range(len(eigvals)) if k not in master]
    if not slave:
        return None
    re_m = float(np.max(np.real(eigvals[master])))
    re_s = float(np.min(np.real(eigvals[slave])))
    if abs(re_m) < 1e-12:
        return None
    sigma = int(np.fix(re_s / re_m))
    return {'master': master, 'slave': slave, 're_master': re_m, 're_slave': re_s,
            'sigma_out': sigma, 'gap_ok': abs(sigma) >= 2}


def main():
    ap = argparse.ArgumentParser(description='LPPL model precheck (SSMTool V2.6 style)')
    ap.add_argument('--alpha', type=float, default=DEFAULT_PARAMS['alpha'])
    ap.add_argument('--gamma', type=float, default=DEFAULT_PARAMS['gamma'])
    ap.add_argument('--Z_A', type=float, default=DEFAULT_PARAMS['Z_A'])
    ap.add_argument('--Z_B', type=float, default=DEFAULT_PARAMS['Z_B'])
    ap.add_argument('--Z_C', type=float, default=DEFAULT_PARAMS['Z_C'])
    ap.add_argument('--Z_D', type=float, default=DEFAULT_PARAMS['Z_D'])
    ap.add_argument('--Z_E', type=float, default=DEFAULT_PARAMS['Z_E'])
    ap.add_argument('--Z_MIX', type=float, default=DEFAULT_PARAMS['Z_MIX'])
    args = ap.parse_args()

    p = {k: getattr(args, k) for k in DEFAULT_PARAMS}
    eqs = find_equilibria(p)

    print("=" * 84)
    print("LPPL MODEL PRECHECK (M=1 polynomial approximation)")
    print("=" * 84)
    print("SYSTEM")
    print("  dy1/dt = Z_MIX*y1 + (1-Z_MIX)*y2 - Z_A*y2*z")
    print("  dy2/dt = alpha*y2 - gamma*y1^3 + Z_B*y1*z")
    print("  dz/dt  = -Z_C*z + Z_D*y1 + Z_E*y1*y2")
    print()
    print("PARAMETERS")
    for k, v in p.items():
        print(f"  {k:8s} = {v:.6g}")
    print()

    print("GLOBAL GATES")
    print(f"  NOTE           mechanical_form   : 3D first-order — V2.6 compatible")
    print(f"  PASS           autonomy          : autonomous ODE (M=1 approx, no time-dep)")
    print(f"  PASS           analyticity       : polynomial degree 3 (quadratic + cubic y1^3)")
    div = p['Z_MIX'] + p['alpha'] - p['Z_C']
    print(f"  {'PASS' if div < 0 else 'WARN':14s} dissipative       : trace(A_origin) = {div:+.6f}")
    print()

    print("EQUILIBRIA")
    v26_candidates = []
    for label, pt in eqs:
        jac = lppl_jacobian(pt, p)
        eigvals = np.linalg.eigvals(jac)
        eigvals = np.array(sorted(eigvals, key=lambda z: (np.real(z), abs(np.imag(z)))))
        cls = _classify(eigvals)
        print(f"  {label}: ({pt[0]:+.6f}, {pt[1]:+.6f}, {pt[2]:+.6f})  [{cls}]")
        for ev in eigvals:
            print(f"    lambda = {_format_eig(ev)}")

        gap = _spectral_gap(eigvals)
        if gap is not None:
            v26_candidates.append({'label': label, 'point': pt, 'gap': gap, 'eigvals': eigvals})

    print()
    print("SSMTOOL V2.6 SPECTRAL GAP ANALYSIS")
    print("  sigma_out = floor(Re_slave / Re_master)")
    print("  |sigma_out| >= 2 required; negative sign = master unstable, slave stable")
    if not v26_candidates:
        print("  no equilibria with complex pair found")
    for c in v26_candidates:
        g = c['gap']
        master_note = 'unstable' if g['re_master'] > 0 else 'stable'
        status = 'PASS' if g['gap_ok'] else 'WARN'
        print(f"  {c['label']:4s}: sigma_out={g['sigma_out']:+d} (|{abs(g['sigma_out'])}|)  "
              f"Re_master={g['re_master']:+.6f} ({master_note})  "
              f"Re_slave={g['re_slave']:+.6f}  gap={status}")

    print()
    print("STRUCTURAL COMPARISON TO WANG")
    print(f"  z-equation        : dz = -Z_C*z + Z_D*y1 + Z_E*y1*y2   [Z_E={p['Z_E']:.1f}, bilinear like Wang +u*v]")
    print(f"  cross-coupling    : -Z_A*y2*z in dy1, +Z_B*y1*z in dy2  [like Wang -v*w, +u*w]")
    print(f"  extra vs Wang     : -gamma*y1^3 in dy2                   [cubic, Wang has none]")
    print(f"  2w mechanism      : Z_E*y1*y2 drives z at 0 and 2w      [identical to Wang]")

    print()
    v10 = 'blocked'
    v26 = 'available' if any(c['gap']['gap_ok'] for c in v26_candidates) else 'blocked'
    v26_labels = ', '.join(c['label'] for c in v26_candidates if c['gap']['gap_ok'])
    print("SSM PATH SUMMARY")
    print(f"  SSMTOOL V1.0    : {v10:10s} [needs mechanical form]")
    print(f"  SSMTOOL V2.6    : {v26:10s} [{v26_labels or 'none'}]")
    print(f"  SSMLEARN        : available   [from BTC residual trajectory]")

    if v26_candidates:
        c = v26_candidates[0]
        sys_data = shifted_first_order(c['point'], p)
        vfy = verify_shifted_ode(sys_data)
        print()
        print(f"VERIFICATION (shifted ODE at {c['label']}, ||dz||={np.linalg.norm(vfy['dz']):.4f}):")
        print(f"  error = {vfy['error']:.2e}")

    print("=" * 84)


if __name__ == '__main__':
    main()
