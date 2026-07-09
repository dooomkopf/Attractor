"""Residual-analysis precheck: embedding budget, eigenpairs, and resonance diagnostics."""
import logging
import os
import sys
from types import SimpleNamespace
import numpy as np
from .amplitude import build_amplitude_support
from .common import analysis_time_vector, identify_modes, parse_int_csv, time_mode_rate_unit
from .constants import DAYS_PER_YEAR
from .cycles import build_loc_segment_rows
from .data import build_residual_context
from .precheck_report import format_precheck_report

ATTRACTOR_DIR = '/home/hz/Data/Attractor'
if ATTRACTOR_DIR not in sys.path:
    sys.path.insert(0, ATTRACTOR_DIR)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
logging.getLogger('ridge_regression').setLevel(logging.WARNING)
logging.getLogger('SSMLearn').setLevel(logging.WARNING)
from ssmlearn_res import fit_ssm  # noqa: E402

def _coerce_params(args_or_params):
    if isinstance(args_or_params, dict):
        return SimpleNamespace(**args_or_params)
    return args_or_params

def _period_years(ev, time_mode='linear'):
    period = 2.0 * np.pi / abs(ev.imag)
    if time_mode == 'linear':
        return period / DAYS_PER_YEAR
    return period

def _find_conjugate_index(eigvals, idx):
    target = np.conj(eigvals[idx])
    candidates = [
        (j, abs(eigvals[j] - target))
        for j in range(len(eigvals))
        if j != idx
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])[0]

def _mode_combinations(dim, max_order):
    combos = []
    if dim != 2 or max_order < 2:
        return combos
    for order in range(2, max_order + 1):
        for a in range(order + 1):
            b = order - a
            combos.append((a, b))
    return combos

def _ev_label(ev, time_mode='linear'):
    if abs(ev.imag) > 1e-10:
        suffix = 'y' if time_mode == 'linear' else 'fit'
        return f"{_period_years(ev, time_mode):.3f}{suffix} (Re={ev.real:+.2e})"
    return f"real (Re={ev.real:+.2e})"

def _scan_near_resonances(lambda_master, lambda_targets, reltol, max_order):
    if len(lambda_master) != 2 or len(lambda_targets) == 0:
        return []
    ref = min(abs(ev) for ev in lambda_master)
    if ref < 1e-10:
        ref = max(abs(ev) for ev in lambda_master)
    abstol = reltol * ref
    rows = []
    for combo in _mode_combinations(2, max_order):
        combo_ev = combo[0] * lambda_master[0] + combo[1] * lambda_master[1]
        for target_idx, target_ev in enumerate(lambda_targets):
            mismatch = abs(combo_ev - target_ev)
            if mismatch < abstol:
                rows.append({
                    'combo': combo,
                    'target_index': int(target_idx),
                    'target_eigenvalue': target_ev,
                    'mismatch_abs': float(mismatch),
                    'mismatch_rel_master_ref': float(mismatch / (ref + 1e-30)),
                })
    return rows


def _resonance_and_gap_check(eigvals, reltol, max_order=4, time_mode='linear'):
    pos_idx = [k for k, ev in enumerate(eigvals) if ev.imag > 1e-10]
    pos_idx_sorted = sorted(pos_idx, key=lambda k: abs(eigvals[k].imag))
    if not pos_idx_sorted:
        return {
            'available': False,
            'reason': 'no oscillatory pair in fitted linear reduced dynamics',
        }

    idx_main = pos_idx_sorted[0]
    idx_main_conj = _find_conjugate_index(eigvals, idx_main)
    if idx_main_conj is None:
        return {
            'available': False,
            'reason': 'failed to locate conjugate partner for main oscillatory mode',
        }

    master_indices = [idx_main, idx_main_conj]
    slave_indices = [k for k in range(len(eigvals)) if k not in master_indices]
    lambda_master = eigvals[master_indices]
    lambda_slave = eigvals[slave_indices]

    master_real = float(np.max(np.real(lambda_master)))
    slave_real_min = float(np.min(np.real(lambda_slave))) if len(lambda_slave) > 0 else None
    stable_gap_ok = (
        len(lambda_slave) > 0
        and master_real < 0.0
        and slave_real_min is not None
        and slave_real_min < master_real
    )

    sigma_in = None
    sigma_out = None
    sigma_note = None
    if stable_gap_ok:
        lambda_all = np.concatenate([lambda_master, lambda_slave])
        sigma_in = int(np.fix(np.min(np.real(lambda_all)) / np.max(np.real(lambda_master))))
        sigma_out = int(np.fix(np.min(np.real(lambda_slave)) / np.max(np.real(lambda_master))))
    else:
        sigma_note = (
            'SSMtool-style sigma_in/sigma_out need a decaying master pair and faster decaying slave modes'
        )

    outer_resonances = _scan_near_resonances(lambda_master, lambda_slave, reltol, max_order)
    inner_resonances = _scan_near_resonances(lambda_master, lambda_master, reltol, max_order)
    inner_resonances = [
        row for row in inner_resonances
        if row['combo'] not in [(1, 1)]  # not possible here, but keep explicit
    ]

    result = {
        'available': True,
        'master_indices': [int(idx) for idx in master_indices],
        'slave_indices': [int(idx) for idx in slave_indices],
        'master_modes': [_ev_label(ev, time_mode) for ev in lambda_master],
        'slave_modes': [_ev_label(ev, time_mode) for ev in lambda_slave],
        'master_real_max': master_real,
        'slave_real_min': slave_real_min,
        'stable_gap_ok': bool(stable_gap_ok),
        'sigma_in': sigma_in,
        'sigma_out': sigma_out,
        'sigma_note': sigma_note,
        'outer_resonances': outer_resonances,
        'inner_resonances': inner_resonances,
        'res_tol': float(reltol),
        'scan_order_max': int(max_order),
    }

    if len(pos_idx_sorted) >= 2:
        idx_sub = pos_idx_sorted[1]
        lam_main = eigvals[idx_main]
        lam_sub = eigvals[idx_sub]
        ratio_vs_half_period = float(_period_years(lam_sub, time_mode) / (_period_years(lam_main, time_mode) / 2.0))
        result['harmonic_2to1'] = {
            'sub_index': int(idx_sub),
            'sub_mode': _ev_label(lam_sub, time_mode),
            'ratio_vs_half_period': ratio_vs_half_period,
            'detuning_frac': float(abs(ratio_vs_half_period - 1.0)),
            'near_2to1': bool(abs(ratio_vs_half_period - 1.0) <= reltol),
        }
    else:
        result['harmonic_2to1'] = None

    return result


def run_precheck(args_or_params, verbose=True):
    args = _coerce_params(args_or_params)
    time_mode = getattr(args, 'time_mode', 'linear')
    payload = build_residual_context(args.filename, args.M, args.years, args.start_idx, time_mode=time_mode)
    ctx = payload['ctx']
    TAU = payload['TAU']
    time_vec = analysis_time_vector(ctx)

    res = fit_ssm(ctx, args.ssm_dim, poly_degree=args.poly_degree, compute_prediction=False, time_vec=time_vec)
    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :args.ssm_dim]
    eigvals_lin, eigvecs_lin = np.linalg.eig(linear_part)

    D_obs = ctx['D_c'].T
    diff_obs = D_obs - res['pred_obs']
    fit_err_frob = float(np.linalg.norm(diff_obs) / (np.linalg.norm(D_obs) + 1e-30))

    pc_reduced = ctx['pc'][:, :args.ssm_dim]
    days_vec_f = np.asarray(ctx['days_vecs'], dtype=float)
    fit_time = np.asarray(time_vec, dtype=float)
    dp_dt = np.gradient(pc_reduced, fit_time, axis=0)
    rhs_at_samples = res['ssm'].reduced_dynamics.predict(pc_reduced)
    n_samples = pc_reduced.shape[0]
    edge = max(5, n_samples // 100)
    sl = slice(edge, -edge) if n_samples > 2 * edge else slice(None)
    dyn_diff = dp_dt[sl] - rhs_at_samples[sl]
    ode_err_frob = float(
        np.linalg.norm(dyn_diff) / (np.linalg.norm(dp_dt[sl]) + 1e-30)
    )

    positive_imag = [ev for ev in eigvals_lin if ev.imag > 1e-10]
    positive_imag = sorted(positive_imag, key=lambda ev: abs(ev.imag))
    oscillatory_modes = [
        {
            'period_years': float(_period_years(ev, time_mode)),
            'growth_rate': float(ev.real),
            'eigenvalue': ev,
        }
        for ev in positive_imag
    ]

    harmonic_candidate = None
    if len(positive_imag) >= 2:
        idx_main, idx_sub = identify_modes(eigvals_lin)
        lam_main = eigvals_lin[idx_main]
        lam_sub = eigvals_lin[idx_sub]
        T_main = float(_period_years(lam_main, time_mode))
        T_sub = float(_period_years(lam_sub, time_mode))
        harmonic_candidate = {
            'idx_main': int(idx_main),
            'idx_sub': int(idx_sub),
            'T_main': T_main,
            'T_sub': T_sub,
            'ratio_vs_half_period': float(T_sub / (T_main / 2.0)) if T_main > 0 else float('nan'),
        }

    resonance_check = _resonance_and_gap_check(
        eigvals_lin,
        reltol=float(getattr(args, 'res_tol', 0.05)),
        max_order=4,
        time_mode=time_mode,
    )

    amplitude_support = build_amplitude_support(
        eigvals_lin,
        eigvecs_lin,
        pc_reduced.T,
        days_vec_f,
    )

    sub_period = None if harmonic_candidate is None or time_mode != 'linear' else harmonic_candidate['T_sub']
    loc_segments = build_loc_segment_rows(
        ctx['days_vecs'].astype(float),
        getattr(args, 'cycles_json', ''),
        main_period_years=oscillatory_modes[0]['period_years'] if oscillatory_modes and time_mode == 'linear' else None,
        sub_period_years=sub_period,
        support_mask=None if amplitude_support is None else amplitude_support['support_mask'],
    )

    cum_var = float(np.cumsum(ctx['var'])[args.ssm_dim - 1])
    fixed_inputs = {
        'filename': payload['filename'],
        'M': int(args.M),
        'years': float(args.years),
        'TAU': int(TAU),
        'W': int(ctx['W']),
        'TAU_linear_days': int(ctx.get('TAU_linear_days', TAU)),
        'TAU_log_steps': None if time_mode != 'log' else int(ctx.get('TAU_log_steps', ctx['TAU'])),
        'TAU_effective_start_days': None if time_mode != 'log' else float(ctx.get('TAU_effective_start_days', 0.0)),
        'W_target_days': float(ctx.get('W_target_days', (args.M - 1) * TAU)),
        'W_effective_start_days': float(ctx.get('W_effective_start_days', ctx['W'])),
        'start_idx': int(args.start_idx),
        'ssm_dim': int(args.ssm_dim),
        'poly_degree': int(args.poly_degree),
        'time_mode': time_mode,
        'time_rate_unit': time_mode_rate_unit(time_mode),
        'res_tol': float(getattr(args, 'res_tol', 0.05)),
    }
    learned_outputs = {
        'N_vec': int(ctx['N']),
        'cum_var': cum_var,
        'fit_err_frobenius': fit_err_frob,
        'ode_err_frobenius': ode_err_frob,
        'n_osc_pairs': len(positive_imag),
        'oscillatory_modes': oscillatory_modes,
        'harmonic_candidate': harmonic_candidate,
        'resonance_check': resonance_check,
        'amplitude_support': amplitude_support,
        'segment_counts': loc_segments,
        'loc_segments': loc_segments,
    }
    result = {
        'filename': payload['filename'],
        'M': int(args.M),
        'years': float(args.years),
        'TAU': int(TAU),
        'W': int(ctx['W']),
        'N_vec': int(ctx['N']),
        'ssm_dim': int(args.ssm_dim),
        'poly_degree': int(args.poly_degree),
        'cum_var': cum_var,
        'fit_err_frobenius': fit_err_frob,
        'ode_err_frobenius': ode_err_frob,
        'ctx': ctx,
        'fit': res,
        'linear_part': linear_part,
        'eigvals_lin': eigvals_lin,
        'eigvecs_lin': eigvecs_lin,
        'n_osc_pairs': len(positive_imag),
        'oscillatory_modes': oscillatory_modes,
        'harmonic_candidate': harmonic_candidate,
        'resonance_check': resonance_check,
        'segment_counts': loc_segments,
        'loc_segments': loc_segments,
        'fixed_inputs': fixed_inputs,
        'learned_outputs': learned_outputs,
    }

    if verbose:
        for line in format_precheck_report(args, result, lambda ev: _ev_label(ev, time_mode)):
            print(line)
    return result


def run_precheck_scan(args_or_params):
    args = _coerce_params(args_or_params)
    ssm_dims = parse_int_csv(args.scan_ssm_dim)
    poly_degrees = parse_int_csv(args.scan_poly)
    if not ssm_dims:
        raise ValueError('scan_ssm_dim must contain at least one integer')
    if not poly_degrees:
        raise ValueError('scan_poly must contain at least one integer')

    rows = []
    print("=" * 137)
    print("RESIDUAL PRECHECK SCAN")
    print("=" * 137)
    print(f"time_mode: {getattr(args, 'time_mode', 'linear')}")
    print(
        "ssm_dim  poly  pairs  fit_err   ode_err   "
        "periods                                   "
        "2:1_detune  near_2to1  sub_Re   sub_sup%  tail_sup%  collapse_y"
    )
    print("-" * 137)

    for ssm_dim in ssm_dims:
        for poly_degree in poly_degrees:
            params = vars(args).copy()
            params['ssm_dim'] = int(ssm_dim)
            params['poly_degree'] = int(poly_degree)
            result = run_precheck(SimpleNamespace(**params), verbose=False)
            res_check = result['resonance_check']
            harmonic = res_check.get('harmonic_2to1')
            period_unit = 'y' if getattr(args, 'time_mode', 'linear') == 'linear' else 'fit'
            periods = ", ".join(
                f"{mode['period_years']:.3f}{period_unit}" for mode in result['oscillatory_modes']
            ) or "none"
            if harmonic is None:
                detune_txt = "-"
                near_txt = "-"
                sub_re_txt = "-"
                sub_sup_txt = "-"
                tail_sup_txt = "-"
                collapse_txt = "-"
            else:
                detune_txt = f"{100.0 * harmonic['detuning_frac']:.1f}%"
                near_txt = "yes" if harmonic['near_2to1'] else "no"
                sub_ev = result['eigvals_lin'][harmonic['sub_index']]
                sub_re_txt = f"{sub_ev.real:+.2e}"
                amp = result['learned_outputs']['amplitude_support']
                sub_sup_txt = f"{100.0 * amp['support_fraction']:.1f}" if amp is not None else "-"
                tail_sup_txt = f"{100.0 * amp['support_fraction_tail']:.1f}" if amp is not None else "-"
                collapse_txt = (
                    f"{amp['collapse_time_years']:.2f}" if amp is not None and amp['collapse_time_years'] is not None else "-"
                )

            print(
                f"{ssm_dim:>7}  {poly_degree:>4}  {result['n_osc_pairs']:>5}  "
                f"{result['fit_err_frobenius']:>8.2e}  "
                f"{result['ode_err_frobenius']:>8.2e}  "
                f"{periods:<40}  {detune_txt:>8}  {near_txt:>9}  {sub_re_txt:>8}  "
                f"{sub_sup_txt:>8}  {tail_sup_txt:>9}  {collapse_txt:>10}"
            )

            rows.append({
                'ssm_dim': int(ssm_dim),
                'poly_degree': int(poly_degree),
                'n_osc_pairs': int(result['n_osc_pairs']),
                'fit_err_frobenius': float(result['fit_err_frobenius']),
                'ode_err_frobenius': float(result['ode_err_frobenius']),
                'periods_years': [float(m['period_years']) for m in result['oscillatory_modes']],
                'harmonic_detuning_frac': None if harmonic is None else float(harmonic['detuning_frac']),
                'harmonic_near_2to1': None if harmonic is None else bool(harmonic['near_2to1']),
                'sub_growth_rate': None if harmonic is None else float(result['eigvals_lin'][harmonic['sub_index']].real),
                'sub_support_fraction': None if harmonic is None or amp is None else float(amp['support_fraction']),
                'tail_support_fraction': None if harmonic is None or amp is None else float(amp['support_fraction_tail']),
                'collapse_time_years': None if harmonic is None or amp is None else amp['collapse_time_years'],
            })

    print("-" * 137)
    print("Legend:")
    print("  pairs        : number of oscillatory pairs in fitted linear reduced dynamics")
    print("  fit_err      : ||D - decoder(reduce(D))||_F / ||D||_F, polynomial SSM fit")
    print("  ode_err      : ||dp/dt - R(p)||_F / ||dp/dt||_F, one-step ODE residual (edge-trimmed)")
    print("  2:1_detune   : |T_sub/(T_main/2) - 1| in percent")
    print("  sub_Re       : growth/decay rate of the candidate harmonic mode")
    print("  sub_sup%     : fraction of samples with supported submode amplitude")
    print("  tail_sup%    : same fraction in the last 20% of the run")
    print("  collapse_y   : last supported time in years since embedding start")
    print("=" * 137)
    return rows
