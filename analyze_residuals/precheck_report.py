"""Formatting and interpretation helpers for the residual precheck report."""

import numpy as np

from .amplitude import format_amplitude_support


def _status_label(level):
    if level == 'pass':
        return 'PASS'
    if level == 'block':
        return 'BLOCK'
    return 'WARN'


def _build_global_gates(result):
    fixed = result['fixed_inputs']
    learned = result['learned_outputs']
    resonance = learned['resonance_check']
    harmonic = learned['harmonic_candidate']
    amp = learned['amplitude_support']

    rows = []
    rows.append({
        'level': 'pass',
        'name': 'embedding_budget',
        'detail': f"N_vec={learned['N_vec']} after W={fixed['W']}d delay embedding",
    })

    var_level = 'pass' if learned['cum_var'] >= 0.80 else 'warn'
    rows.append({
        'level': var_level,
        'name': 'variance_capture',
        'detail': f"{100.0 * learned['cum_var']:.2f}% cumulative PCA variance at ssm_dim={fixed['ssm_dim']}",
    })

    osc_level = 'pass' if learned['n_osc_pairs'] >= 1 else 'block'
    rows.append({
        'level': osc_level,
        'name': 'primary_oscillation',
        'detail': f"fitted reduced dynamics contains {learned['n_osc_pairs']} oscillatory pair(s)",
    })

    if harmonic is None:
        harm_level = 'block' if fixed['ssm_dim'] >= 4 else 'warn'
        harm_detail = 'no second oscillatory pair found in the chosen reduced model'
    else:
        detune = abs(harmonic['ratio_vs_half_period'] - 1.0)
        harm_level = 'pass' if detune <= fixed['res_tol'] else 'warn'
        harm_detail = (
            f"T_sub/(T_main/2)={harmonic['ratio_vs_half_period']:.3f} "
            f"(detuning {100.0 * detune:.1f}%)"
        )
    rows.append({
        'level': harm_level,
        'name': 'harmonic_candidate',
        'detail': harm_detail,
    })

    if harmonic is None or amp is None:
        sup_level = 'warn'
        sup_detail = 'submode amplitude support unavailable'
    else:
        sup_level = 'pass' if amp['support_fraction'] >= 0.80 else 'warn'
        sup_detail = (
            f"global={100.0 * amp['support_fraction']:.1f}%  "
            f"tail={100.0 * amp['support_fraction_tail']:.1f}%"
        )
    rows.append({
        'level': sup_level,
        'name': 'submode_support',
        'detail': sup_detail,
    })

    if not resonance['available']:
        gap_level = 'warn'
        gap_detail = resonance['reason']
    elif resonance['stable_gap_ok']:
        gap_level = 'pass'
        gap_detail = (
            f"sigma_in={resonance['sigma_in']}  sigma_out={resonance['sigma_out']} "
            "[classical fast-slave gap available]"
        )
    else:
        gap_level = 'warn'
        gap_detail = resonance['sigma_note']
    rows.append({
        'level': gap_level,
        'name': 'classical_fast_slave',
        'detail': gap_detail,
    })
    return rows


def _build_harmonic_interpretation(result):
    fixed = result['fixed_inputs']
    learned = result['learned_outputs']
    harmonic = learned['harmonic_candidate']
    amp = learned['amplitude_support']
    resonance = learned['resonance_check']

    independent_second_pair = fixed['ssm_dim'] >= 4
    near_2to1 = harmonic is not None and abs(harmonic['ratio_vs_half_period'] - 1.0) <= fixed['res_tol']
    strong_support = amp is not None and amp['support_fraction'] >= 0.80
    collapse_txt = 'none'
    if amp is not None and amp['collapse_late'] and amp['collapse_time_years'] is not None:
        collapse_txt = f"{amp['collapse_time_years']:.2f}y"

    if independent_second_pair:
        reason = (
            'ssm_dim>=4 allows two complex pairs in the reduced linear model. '
            'A near-2:1 secondary pair is therefore allowed, but phase and slave tests '
            'must still decide whether it is a resonant companion or an independent mode.'
        )
    else:
        reason = (
            'ssm_dim<4 cannot carry a second independent oscillatory pair in the reduced linear model. '
            'Any extra ~2y structure must then be harmonic, slaved, or observation-induced.'
        )

    if resonance['available'] and resonance['stable_gap_ok']:
        mechanism = (
            'The fitted reduced spectrum shows a faster-decaying slave sector, so a classical '
            'SSMtool-style fast slave remains dynamically plausible.'
        )
    else:
        mechanism = (
            'The fitted reduced spectrum does not show a faster-decaying slave pair. '
            'Read the secondary component as a resonant harmonic candidate first, not as a classical fast slave.'
        )

    return {
        'independent_second_pair': independent_second_pair,
        'near_2to1': near_2to1,
        'strong_support': strong_support,
        'collapse_txt': collapse_txt,
        'reason': reason,
        'mechanism': mechanism,
    }


def _build_path_summary(result):
    fixed = result['fixed_inputs']
    learned = result['learned_outputs']
    harmonic = learned['harmonic_candidate']
    amp = learned['amplitude_support']
    resonance = learned['resonance_check']

    if fixed['ssm_dim'] < 4:
        two_pair_txt = 'blocked     [chosen ssm_dim cannot represent two complex pairs]'
    elif harmonic is None:
        two_pair_txt = 'blocked     [no second oscillatory pair identified]'
    else:
        detune = abs(harmonic['ratio_vs_half_period'] - 1.0)
        label = 'plausible' if detune <= fixed['res_tol'] else 'fragile'
        two_pair_txt = f"{label:10s} [detuning {100.0 * detune:.1f}%]"

    support_txt = 'unavailable'
    if amp is not None:
        support_txt = (
            f"present     [global {100.0 * amp['support_fraction']:.1f}%, "
            f"tail {100.0 * amp['support_fraction_tail']:.1f}%, collapse {amp['collapse_time_years']:.2f}y]"
            if amp['collapse_time_years'] is not None else
            f"present     [global {100.0 * amp['support_fraction']:.1f}%]"
        )

    gap_txt = (
        'available   [classical fast-slave gap in fitted reduced spectrum]'
        if resonance['available'] and resonance['stable_gap_ok']
        else 'blocked     [no faster-decaying slave pair in fitted reduced spectrum]'
    )

    return [
        ('single-master', f"test with --ssm_dim 2/3 controls   [current --ssm_dim={fixed['ssm_dim']}]"),
        ('two-pair path', two_pair_txt),
        ('support path', support_txt),
        ('fast-slave path', gap_txt),
        ('next readout', '03 phase, 04 scaling, 05 scan'),
    ]


def format_precheck_report(args, result, ev_label_func):
    fixed = result['fixed_inputs']
    learned = result['learned_outputs']
    oscillatory_modes = learned['oscillatory_modes']
    harmonic_candidate = learned['harmonic_candidate']
    resonance_check = learned['resonance_check']
    amplitude_support = learned['amplitude_support']
    loc_segments = result['loc_segments']

    gates = _build_global_gates(result)
    harmonic = _build_harmonic_interpretation(result)
    path_rows = _build_path_summary(result)

    lines = []
    lines.append("=" * 64)
    lines.append("RESIDUAL PRECHECK")
    lines.append("=" * 64)
    lines.append("FIXED / VORGEGEBEN")
    lines.append(f"  file              : {fixed['filename']}")
    lines.append(f"  start_idx         : {fixed['start_idx']}")
    lines.append(f"  M                 : {fixed['M']}   [chosen embedding dimension]")
    lines.append(f"  years             : {fixed['years']:.2f}   [chosen embedding window]")
    if fixed['time_mode'] == 'linear':
        lines.append(f"  TAU               : {fixed['TAU_linear_days']}d   [derived from M and years, not learned]")
        lines.append(f"  W                 : {fixed['W_target_days']:.0f}d   [embedding span, not learned]")
    else:
        lines.append(f"  TAU_linear_ref    : {fixed['TAU_linear_days']}d   [legacy lag scale requested before log-time resampling]")
        lines.append(f"  TAU_log_steps     : {fixed['TAU_log_steps']}   [lag steps on the uniform log(day) grid]")
        lines.append(f"  TAU_start_eff     : {fixed['TAU_effective_start_days']:.1f}d   [effective linear lag near the start of the run]")
        lines.append(f"  W_target          : {fixed['W_target_days']:.0f}d   [requested start-window span before resampling]")
        lines.append(f"  W_start_eff       : {fixed['W_effective_start_days']:.0f}d   [actual start-window span on the uniform log(day) grid]")
    lines.append(f"  cycles_json       : {getattr(args, 'cycles_json', '')}   [strict halving boundaries source]")
    lines.append(f"  ssm_dim           : {fixed['ssm_dim']}   [chosen model dimension, not learned]")
    lines.append(f"  poly              : {fixed['poly_degree']}   [chosen polynomial order, not learned]")
    lines.append(f"  time_mode         : {fixed['time_mode']}   [analysis clock for reduced dynamics and d/dt estimates]")
    lines.append(f"  res_tol           : {fixed['res_tol']:.3f}   [chosen resonance tolerance, not learned]")
    lines.append("")
    lines.append("GLOBAL GATES")
    for row in gates:
        lines.append(f"  {_status_label(row['level']):14s} {row['name']:20s}: {row['detail']}")
    lines.append("")
    lines.append("HARMONIC INTERPRETATION")
    lines.append(f"  independent 2nd pair : {harmonic['independent_second_pair']}")
    lines.append(
        f"  near 2:1 candidate   : {harmonic['near_2to1']}   "
        f"[T_sub close to T_main/2 under current res_tol]"
    )
    lines.append(
        f"  strong sub support   : {harmonic['strong_support']}   "
        f"[global/tail support from amplitude readout]"
    )
    lines.append(f"  late collapse        : {harmonic['collapse_txt']}")
    lines.append(f"  reason               : {harmonic['reason']}")
    lines.append(f"  mechanism            : {harmonic['mechanism']}")
    lines.append("")
    lines.append("SSM PATH SUMMARY")
    for name, detail in path_rows:
        lines.append(f"  {name:16s}: {detail}")
    lines.append("")
    lines.append("LEARNED / AUS DATEN + FIT")
    lines.append(f"  N_vec             : {learned['N_vec']}   [available delay vectors after embedding]")
    lines.append(f"  cumulative var    : {100.0 * learned['cum_var']:.2f}%   [PCA, learned]")
    lines.append(
        f"  fit error (Frob)  : {result['fit_err_frobenius']:.2e}   "
        "[||D - decoder(reduce(D))||_F / ||D||_F, polynomial SSM]"
    )
    lines.append(
        f"  ODE error (Frob)  : {result['ode_err_frobenius']:.2e}   "
        f"[||dp/dt - R(p)||_F / ||dp/dt||_F with t={fixed['time_mode']}, one-step ODE residual]"
    )
    lines.append(f"  oscillatory pairs : {learned['n_osc_pairs']}   [from fitted linear reduced dynamics]")
    period_unit = 'y' if fixed['time_mode'] == 'linear' else 'fit'
    if oscillatory_modes:
        periods_txt = ", ".join(
            f"{mode['period_years']:.3f}{period_unit} (Re={mode['growth_rate']:+.2e})"
            for mode in oscillatory_modes
        )
        lines.append(f"  periods           : {periods_txt}   [learned modal periods]")
    else:
        lines.append("  periods           : none")
    if harmonic_candidate is not None:
        lines.append(
            f"  harmonic candidate: T_main={harmonic_candidate['T_main']:.3f}{period_unit}  "
            f"T_sub={harmonic_candidate['T_sub']:.3f}{period_unit}  "
            f"T_sub/(T_main/2)={harmonic_candidate['ratio_vs_half_period']:.3f}   "
            "[near 1.0 => harmonic candidate]"
        )
    else:
        lines.append("  harmonic candidate: unavailable (need at least 2 oscillatory pairs)")
    lines.append("")
    lines.extend(["AMPLITUDE SUPPORT"] + format_amplitude_support(amplitude_support) + [""])
    lines.append("RESONANCE / SPECTRAL GAP")
    period_unit = 'y' if fixed['time_mode'] == 'linear' else 'fit'
    if not resonance_check['available']:
        lines.append(f"  status            : unavailable   [{resonance_check['reason']}]")
    else:
        lines.append(
            f"  main pair         : {', '.join(resonance_check['master_modes'])}   "
            "[slowest oscillatory pair in fitted reduced dynamics]"
        )
        if resonance_check['harmonic_2to1'] is not None:
            h2 = resonance_check['harmonic_2to1']
            lines.append(
                f"  2:1 check         : {h2['sub_mode']}  "
                f"T_sub/(T_main/2)={h2['ratio_vs_half_period']:.3f}  "
                f"detuning={100.0 * h2['detuning_frac']:.1f}%  "
                f"near_2to1={h2['near_2to1']}"
            )
        else:
            lines.append("  2:1 check         : unavailable (need at least 2 oscillatory pairs)")
        if resonance_check['stable_gap_ok']:
            lines.append(
                f"  spectral gap      : sigma_in={resonance_check['sigma_in']}  "
                f"sigma_out={resonance_check['sigma_out']}   "
                "[SSMtool-style gap check on fitted modes]"
            )
        else:
            lines.append(
                "  spectral gap      : unavailable   "
                f"[{resonance_check['sigma_note']}]"
            )
        if resonance_check['outer_resonances']:
            lines.append("  outer resonances  :")
            for row in resonance_check['outer_resonances']:
                lines.append(
                    f"    combo={row['combo']} -> {ev_label_func(row['target_eigenvalue'])}  "
                    f"mismatch={row['mismatch_abs']:.2e}"
                )
        else:
            lines.append(
                f"  outer resonances  : none   "
                f"[scan orders 2..{resonance_check['scan_order_max']} with tol={resonance_check['res_tol']:.3f}]"
            )
        if resonance_check['inner_resonances']:
            lines.append("  inner resonances  :")
            for row in resonance_check['inner_resonances']:
                lines.append(
                    f"    combo={row['combo']} -> {ev_label_func(row['target_eigenvalue'])}  "
                    f"mismatch={row['mismatch_abs']:.2e}"
                )
        else:
            lines.append(
                f"  inner resonances  : none   "
                f"[scan orders 2..{resonance_check['scan_order_max']} with tol={resonance_check['res_tol']:.3f}]"
            )
    lines.append("")
    lines.append("DERIVED DIAGNOSTICS")
    lines.append("  segment counts    :")
    for seg in loc_segments:
        end_txt = f"{seg['end_day']:.0f}" if np.isfinite(seg['end_day']) else "inf"
        lines.append(
            f"    {seg['label']}: N_vec={seg['n_vec']}  "
            f"days=[{seg['start_day']:.0f}, {end_txt})  "
            f"dates=[{seg['start_date_text']}, {seg['end_date_text'] or '+'})   "
            "[not re-fit; just sample count under current global embedding]"
        )
    if getattr(args, 'loc', False):
        lines.append("  local readout     :")
        for seg in loc_segments:
            cyc_main = "-" if seg['cycles_main'] is None else f"{seg['cycles_main']:.2f}"
            cyc_sub = "-" if seg['cycles_sub'] is None else f"{seg['cycles_sub']:.2f}"
            sup = "-" if seg['support_fraction'] is None else f"{100.0 * seg['support_fraction']:.1f}%"
            lines.append(
                f"    {seg['label']}: span={seg['span_days']:.0f}d  "
                f"cycles_main={cyc_main}  cycles_sub={cyc_sub}  sub_support={sup}"
            )
    lines.append("=" * 64)
    return lines
