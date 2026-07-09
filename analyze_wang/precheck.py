"""SSMtool-style precheck for the Wang 2-scroll system."""

from types import SimpleNamespace

import numpy as np

from .constants import DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D, DEFAULT_IC, IMAG_TOL, REAL_TOL
from .system import equilibria, shifted_quadratic_terms, wang_jacobian


def _coerce_params(args_or_params):
    if isinstance(args_or_params, dict):
        return SimpleNamespace(**args_or_params)
    return args_or_params


def _period_from_eig(ev):
    if abs(ev.imag) <= IMAG_TOL:
        return None
    return float(2.0 * np.pi / abs(ev.imag))


def _format_eig(ev):
    txt = f"{ev.real:+.6f}"
    if abs(ev.imag) > IMAG_TOL:
        txt += f" {'+' if ev.imag >= 0 else '-'} i*{abs(ev.imag):.6f}"
        period = _period_from_eig(ev)
        if period is not None:
            txt += f"  T={period:.3f}"
    return txt


def _classify_equilibrium(eigvals):
    pos = int(np.sum(np.real(eigvals) > REAL_TOL))
    neg = int(np.sum(np.real(eigvals) < -REAL_TOL))
    zero = len(eigvals) - pos - neg
    osc_pairs = int(np.sum(np.imag(eigvals) > IMAG_TOL))
    has_complex_pair = osc_pairs >= 1
    stable = bool(np.all(np.real(eigvals) < -REAL_TOL))
    if has_complex_pair and stable:
        label = 'stable_focus_pair'
    elif has_complex_pair and pos > 0 and neg > 0:
        label = 'saddle_focus_pair'
    elif pos > 0 and neg > 0:
        label = 'real_saddle'
    else:
        label = 'other'
    return {
        'n_pos': pos,
        'n_neg': neg,
        'n_zero': zero,
        'osc_pairs': osc_pairs,
        'has_complex_pair': has_complex_pair,
        'stable': stable,
        'label': label,
    }


def _equilibrium_rows(a, b, c, d):
    rows = []
    for label, point in equilibria(a, b, c, d):
        jac = wang_jacobian(point, a, b, c, d)
        eigvals = np.linalg.eigvals(jac)
        eigvals = np.array(sorted(eigvals, key=lambda z: (np.real(z), abs(np.imag(z)))))
        cls = _classify_equilibrium(eigvals)
        pos_imag = [ev for ev in eigvals if ev.imag > IMAG_TOL]
        period = None if not pos_imag else _period_from_eig(pos_imag[0])
        rows.append({
            'label': label,
            'point': point,
            'jacobian': jac,
            'eigvals': eigvals,
            'class': cls,
            'osc_period': period,
        })
    return rows


def _spectral_gap_at_focus(eigvals):
    pos_imag_idx = [k for k, ev in enumerate(eigvals) if ev.imag > IMAG_TOL]
    if not pos_imag_idx:
        return None
    conj_idx = set()
    for k in pos_imag_idx:
        for j in range(len(eigvals)):
            if j != k and abs(eigvals[j] - np.conj(eigvals[k])) < 1e-8:
                conj_idx.add(j)
    master_idx = sorted(set(pos_imag_idx) | conj_idx)
    slave_idx = [k for k in range(len(eigvals)) if k not in master_idx]
    if not slave_idx:
        return None
    re_master_max = float(np.max(np.real(eigvals[master_idx])))
    re_slave_min = float(np.min(np.real(eigvals[slave_idx])))
    if abs(re_master_max) < 1e-12:
        return None
    sigma_out = int(np.fix(re_slave_min / re_master_max))
    return {
        'master_idx': master_idx,
        'slave_idx': slave_idx,
        're_master_max': re_master_max,
        're_slave_min': re_slave_min,
        'sigma_out': sigma_out,
        'gap_sufficient': abs(sigma_out) >= 2,
    }


def _scan_resonances(eigvals, master_idx, slave_idx, reltol=0.05, max_order=4):
    lam_m = eigvals[master_idx]
    lam_s = eigvals[slave_idx]
    if len(lam_m) != 2 or len(lam_s) == 0:
        return []
    ref = min(abs(ev) for ev in lam_m)
    if ref < 1e-10:
        ref = max(abs(ev) for ev in lam_m)
    abstol = reltol * ref
    hits = []
    for order in range(2, max_order + 1):
        for a in range(order + 1):
            b = order - a
            combo_ev = a * lam_m[0] + b * lam_m[1]
            for si, sev in enumerate(lam_s):
                if abs(combo_ev - sev) < abstol:
                    hits.append({'combo': (a, b), 'slave_idx': slave_idx[si], 'mismatch': float(abs(combo_ev - sev))})
    return hits


def _global_gates(a, b, c, d, eq_rows):
    mechanical_form = {
        'passed': False,
        'detail': '3D first-order Wang system — no M,C,K split (V1.0 needs it, V2.6 does not)',
    }
    autonomy = {'passed': True, 'detail': 'autonomous ODE'}
    analyticity = {'passed': True, 'detail': 'polynomial vector field of degree 2'}
    dissipative = {
        'passed': (a - b - c) < 0.0,
        'detail': f'divergence = a-b-c = {a-b-c:+.3f}',
    }
    stable_focuses = [row for row in eq_rows if row['class']['stable'] and row['class']['has_complex_pair']]
    stable_gate = {
        'passed': bool(stable_focuses),
        'detail': 'has asymptotically stable equilibrium with complex pair'
        if stable_focuses else 'no asymptotically stable equilibrium with complex pair',
    }
    return {
        'mechanical_form': mechanical_form,
        'autonomy': autonomy,
        'analyticity': analyticity,
        'dissipative': dissipative,
        'stable_focus_pair': stable_gate,
    }


def _harmonic_interpretation(eq_rows):
    quadratics = shifted_quadratic_terms()
    oscillatory_eqs = [row for row in eq_rows if row['class']['has_complex_pair']]
    can_have_independent_second_pair = False
    local_second_harmonic_plausible = bool(oscillatory_eqs)
    reason = (
        '3D local linearization can contain at most one complex pair; a separate second oscillatory mode '
        'cannot appear locally as another pair. Quadratic terms (-v*w, +u*w, +u*v) can still generate DC and 2ω content.'
    )
    slave_channel = (
        'If u,v oscillate near ω on a 2D master oscillation, the +u*v term drives w at 0 and 2ω; '
        'the cross-couplings -v*w and +u*w feed that harmonic back into x/y observables.'
    )
    return {
        'quadratic_terms_shifted': quadratics,
        'oscillatory_equilibria': [row['label'] for row in oscillatory_eqs],
        'independent_second_pair_local': can_have_independent_second_pair,
        'second_harmonic_slave_plausible': local_second_harmonic_plausible,
        'reason': reason,
        'slave_channel': slave_channel,
    }


def _scroll_assignment(eq_rows, ic):
    ic = np.asarray(ic, dtype=float)
    foci = [row for row in eq_rows if row['class']['has_complex_pair']]
    if len(foci) < 2:
        return {}
    z_vals = np.array([row['point'][2] for row in foci])
    z_ic = float(ic[2])
    z_mid = float(np.median(z_vals))
    if z_ic >= z_mid:
        scroll_set = {row['label'] for row in foci if row['point'][2] >= z_mid}
    else:
        scroll_set = {row['label'] for row in foci if row['point'][2] < z_mid}
    return {row['label']: (row['label'] in scroll_set) for row in foci}


def run_precheck(args_or_params, verbose=True):
    args = _coerce_params(args_or_params)
    a = float(getattr(args, 'a', DEFAULT_A))
    b = float(getattr(args, 'b', DEFAULT_B))
    c = float(getattr(args, 'c', DEFAULT_C))
    d = float(getattr(args, 'd', DEFAULT_D))
    ic = getattr(args, 'ic', DEFAULT_IC)

    eq_rows = _equilibrium_rows(a, b, c, d)
    gates = _global_gates(a, b, c, d, eq_rows)
    harmonic = _harmonic_interpretation(eq_rows)
    direct_ssmtool_v10 = all(g['passed'] for g in gates.values())

    scroll_map = _scroll_assignment(eq_rows, ic)

    v26_candidates = []
    for row in eq_rows:
        if not row['class']['has_complex_pair']:
            continue
        gap = _spectral_gap_at_focus(row['eigvals'])
        if gap is None:
            continue
        resonances = _scan_resonances(
            row['eigvals'], gap['master_idx'], gap['slave_idx'],
        )
        v26_candidates.append({
            'label': row['label'],
            'gap': gap,
            'resonances': resonances,
            'is_scroll_center': scroll_map.get(row['label'], False),
        })

    v26_available = any(c['gap']['gap_sufficient'] and not c['resonances'] for c in v26_candidates)

    result = {
        'fixed_inputs': {'a': a, 'b': b, 'c': c, 'd': d, 'ic': ic},
        'equilibria': eq_rows,
        'gates': gates,
        'harmonic': harmonic,
        'scroll_map': scroll_map,
        'direct_ssmtool_v10': direct_ssmtool_v10,
        'v26_candidates': v26_candidates,
        'v26_available': v26_available,
        'ssmlearn_available': True,
    }

    if verbose:
        print("=" * 84)
        print("WANG 2-SCROLL PRECHECK")
        print("=" * 84)
        print("FIXED / VORGEGEBEN")
        print(f"  a={a:.6f}  b={b:.6f}  c={c:.6f}  d={d:.6f}")
        print(f"  IC=({ic[0]:+.2f}, {ic[1]:+.2f}, {ic[2]:+.2f})")
        print()
        print("GLOBAL GATES")
        for key, gate in gates.items():
            if key == 'mechanical_form':
                status = 'NOTE '
            elif gate['passed']:
                status = 'PASS '
            else:
                status = 'BLOCK (V1.0)'
            print(f"  {status:14s} {key:18s}: {gate['detail']}")
        print()
        print("EQUILIBRIA")
        for row in eq_rows:
            x, y, z = row['point']
            scroll_tag = ''
            if scroll_map.get(row['label'], False):
                scroll_tag = '  ◄ scroll center'
            print(f"  {row['label']}: ({x:+.6f}, {y:+.6f}, {z:+.6f})  [{row['class']['label']}]{scroll_tag}")
            for ev in row['eigvals']:
                print(f"    lambda = {_format_eig(ev)}")
        print()
        q = harmonic['quadratic_terms_shifted']
        print("HARMONIC INTERPRETATION")
        print(f"  independent 2nd pair : {harmonic['independent_second_pair_local']}")
        print(f"  2w slave expected    : {harmonic['second_harmonic_slave_plausible']}"
              f"   [quadratic NL on 2D SSM generates 2w at order 2]")
        print(f"  shifted coords       : u = x-x_eq, v = y-y_eq, w = z-z_eq")
        print(f"  shifted quadratics   : du {q['du']}, dv {q['dv']}, dw {q['dw']}")
        print(f"  reason               : {harmonic['reason']}")
        print(f"  mechanism            : {harmonic['slave_channel']}")
        print()
        print("SSMTOOL V2.6 SPECTRAL GAP ANALYSIS")
        print("  sigma_out = floor(Re_slave / Re_master)")
        print("  |sigma_out| >= 2 required; negative sign = master unstable (Re>0), slave stable (Re<0)")
        if not v26_candidates:
            print("  no equilibria with complex pair found")
        for cand in v26_candidates:
            gap = cand['gap']
            status = 'PASS' if gap['gap_sufficient'] else 'WARN'
            res_txt = 'none' if not cand['resonances'] else f"{len(cand['resonances'])} near-resonance(s)"
            scroll_tag = ' ◄ scroll' if cand.get('is_scroll_center') else ''
            master_note = 'unstable' if gap['re_master_max'] > 0 else 'stable'
            print(f"  {cand['label']:4s}: sigma_out={gap['sigma_out']:+d} (|{abs(gap['sigma_out'])}|)  "
                  f"Re_master={gap['re_master_max']:+.4f} ({master_note})  "
                  f"Re_slave={gap['re_slave_min']:+.4f}  "
                  f"gap={status}  res={res_txt}{scroll_tag}")
        print()
        print("SSM PATH SUMMARY")
        v10_txt = 'possible' if direct_ssmtool_v10 else 'blocked'
        v26_txt = 'available' if v26_available else 'blocked'
        scroll_eqs = [c['label'] for c in v26_candidates
                      if c['gap']['gap_sufficient'] and not c['resonances'] and c.get('is_scroll_center')]
        all_eqs = [c['label'] for c in v26_candidates
                   if c['gap']['gap_sufficient'] and not c['resonances']]
        v26_detail = ', '.join(all_eqs) if all_eqs else 'no suitable equilibria'
        print(f"  SSMTOOL V1.0    : {v10_txt:10s} [needs stable focus + mechanical form]")
        print(f"  SSMTOOL V2.6    : {v26_txt:10s} [{v26_detail}]")
        if scroll_eqs:
            print(f"  -> scroll centers   : {', '.join(scroll_eqs)}   [closest to IC, recommended for SSM computation]")
        print(f"  SSMLEARN        : available   [full state (x,y,z), no embedding needed]")
        print("=" * 84)
    return result
