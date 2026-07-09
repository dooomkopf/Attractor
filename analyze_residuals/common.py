"""Common utility functions for residual analysis."""

import numpy as np


def build_time_vector(days_vecs, time_mode):
    """Return the analysis clock used by SSMLearn and derivative estimates."""
    days = np.asarray(days_vecs, dtype=float)
    if time_mode == 'linear':
        return days.copy()
    if time_mode == 'log':
        if np.any(days <= 0.0):
            raise ValueError('log time mode requires strictly positive day indices')
        return np.log10(days)
    raise ValueError(f"unsupported time_mode={time_mode!r}")


def analysis_time_vector(ctx):
    """Return the fit-time vector used for the current residual analysis."""
    if 'fit_time_vecs' in ctx:
        return np.asarray(ctx['fit_time_vecs'], dtype=float)
    return np.asarray(ctx['days_vecs'], dtype=float)


def time_mode_rate_unit(time_mode):
    """Human-readable unit for d/dt style quantities under the chosen clock."""
    if time_mode == 'linear':
        return 'd'
    if time_mode == 'log':
        return 'log(day)'
    raise ValueError(f"unsupported time_mode={time_mode!r}")


def identify_modes(eigenvalues):
    """Return the two positive-imag oscillatory modes sorted by |Im|."""
    pos_idx = [k for k, ev in enumerate(eigenvalues) if ev.imag > 1e-10]
    if len(pos_idx) < 2:
        raise RuntimeError(
            f"Brauche mindestens 2 komplexe Paare, aber nur {len(pos_idx)} "
            f"Eigenwerte mit Im>0 gefunden."
        )
    pos_idx_sorted = sorted(pos_idx, key=lambda k: abs(eigenvalues[k].imag))
    return pos_idx_sorted[0], pos_idx_sorted[1]


def smooth_real_series(values, window_samples):
    """Moving-average smoothing with edge padding, matching harmonic_test_phase.py."""
    arr = np.asarray(values, dtype=float)
    if window_samples <= 1:
        return arr.copy()
    kernel = np.ones(int(window_samples), dtype=float)
    kernel /= kernel.sum()
    left = int(window_samples) // 2
    right = int(window_samples) - 1 - left
    padded = np.pad(arr, (left, right), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def smooth_phase_series(phases, window_samples):
    """Circular moving-average smoothing in the complex plane."""
    arr = np.asarray(phases, dtype=float)
    if window_samples <= 1:
        return arr.copy()
    phasor = np.exp(1j * arr)
    real_s = smooth_real_series(phasor.real, window_samples)
    imag_s = smooth_real_series(phasor.imag, window_samples)
    return np.angle(real_s + 1j * imag_s)


def parse_int_csv(text):
    """Parse comma-separated integers, removing duplicates while preserving order."""
    values = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in values:
            values.append(value)
    return values
