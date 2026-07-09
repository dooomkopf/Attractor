"""Modal amplitude-support diagnostics for residual prechecks."""

import numpy as np

from .common import identify_modes, smooth_real_series
from .constants import DAYS_PER_YEAR


def _pair_projection_basis(eigvecs_lin, pair_index):
    vec = np.asarray(eigvecs_lin[:, pair_index], dtype=complex)
    basis = np.column_stack([np.real(vec), np.imag(vec)])
    q, _ = np.linalg.qr(basis)
    rank = int(np.linalg.matrix_rank(basis))
    if rank <= 0:
        raise ValueError('oscillatory eigenvector produced a rank-0 real basis')
    return q[:, :rank]


def oscillatory_pair_amplitude(eigvecs_lin, reduced_state, pair_index):
    """Amplitude of one oscillatory pair via orthogonal projection in reduced space."""
    basis = _pair_projection_basis(eigvecs_lin, pair_index)
    projected = basis.T @ np.asarray(reduced_state, dtype=float)
    return np.linalg.norm(projected, axis=0)


def _support_threshold(median_main, rel_threshold):
    return float(rel_threshold) * float(median_main)


def amplitude_support_metrics(
    eigvals,
    eigvecs_lin,
    reduced_state,
    main_index,
    sub_index,
    days_vecs,
    rel_threshold=0.25,
    smooth_fraction=0.05,
):
    """Measure how strongly the candidate harmonic mode is supported in time."""
    amp_main = oscillatory_pair_amplitude(eigvecs_lin, reduced_state, main_index)
    amp_sub = oscillatory_pair_amplitude(eigvecs_lin, reduced_state, sub_index)

    med_main = float(np.median(amp_main))
    med_sub = float(np.median(amp_sub))
    ratio_main_sub = med_main / max(med_sub, 1e-30)
    threshold = _support_threshold(med_main, rel_threshold)

    n_samples = len(amp_sub)
    smooth_samples = max(5, int(round(smooth_fraction * n_samples)))
    amp_main_s = smooth_real_series(amp_main, smooth_samples)
    amp_sub_s = smooth_real_series(amp_sub, smooth_samples)

    support_mask = amp_sub_s >= threshold
    support_fraction = float(np.mean(support_mask))
    support_fraction_tail = float(np.mean(support_mask[int(0.8 * n_samples):])) if n_samples >= 5 else support_fraction

    above_idx = np.where(support_mask)[0]
    if len(above_idx) == 0:
        collapse_idx = None
        collapse_time_days = None
        collapse_time_years = None
        collapse_late = False
    else:
        collapse_idx = int(above_idx.max())
        collapse_time_days = float(days_vecs[collapse_idx])
        collapse_time_years = float((days_vecs[collapse_idx] - days_vecs[0]) / DAYS_PER_YEAR)
        collapse_late = bool(collapse_idx < n_samples - 1)

    return {
        'amp_main': amp_main,
        'amp_sub': amp_sub,
        'amp_main_smooth': amp_main_s,
        'amp_sub_smooth': amp_sub_s,
        'support_mask': support_mask,
        'median_main': med_main,
        'median_sub': med_sub,
        'ratio_main_sub': float(ratio_main_sub),
        'threshold_rel_main': float(rel_threshold),
        'threshold_abs': float(threshold),
        'support_fraction': support_fraction,
        'support_fraction_tail': support_fraction_tail,
        'collapse_index': collapse_idx,
        'collapse_time_days': collapse_time_days,
        'collapse_time_years': collapse_time_years,
        'collapse_late': collapse_late,
        'smooth_samples': int(smooth_samples),
    }


def build_amplitude_support(eigvals, eigvecs_lin, reduced_state, days_vecs):
    """Convenience wrapper that chooses the two slowest oscillatory modes."""
    pos_idx = [k for k, ev in enumerate(eigvals) if ev.imag > 1e-10]
    if len(pos_idx) < 2:
        return None
    idx_main, idx_sub = identify_modes(eigvals)
    return amplitude_support_metrics(
        eigvals,
        eigvecs_lin,
        reduced_state,
        idx_main,
        idx_sub,
        days_vecs,
    )


def format_amplitude_support(metrics):
    """Render a compact amplitude-support block for the precheck report."""
    if metrics is None:
        return [
            "  status            : unavailable   [need at least 2 oscillatory pairs]",
        ]

    collapse_txt = "none"
    if metrics['collapse_late'] and metrics['collapse_time_years'] is not None:
        collapse_txt = f"{metrics['collapse_time_years']:.2f}y"

    return [
        f"  modes             : main/sub selected from the two slowest oscillatory pairs",
        f"  median |z_main|   : {metrics['median_main']:.3e}   [pair-subspace projection norm]",
        f"  median |z_sub|    : {metrics['median_sub']:.3e}   [pair-subspace projection norm]",
        f"  main/sub          : {metrics['ratio_main_sub']:.2f}",
        f"  support threshold : |z_sub| >= {metrics['threshold_rel_main']:.2f} * median(|z_main|) = {metrics['threshold_abs']:.3e}",
        f"  support fraction   : {100.0 * metrics['support_fraction']:.1f}%   [smoothed over {metrics['smooth_samples']} samples]",
        f"  tail support frac  : {100.0 * metrics['support_fraction_tail']:.1f}%   [last 20% of samples]",
        f"  collapse time      : {collapse_txt}   [last smoothed sample above threshold]",
    ]
