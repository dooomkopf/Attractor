"""Phase-lock diagnostics for Wang 2-scroll harmonics."""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


def _bandpass(series, fs, center, rel_bw=0.2, order=4):
    nyq = 0.5 * float(fs)
    width = max(float(center) * float(rel_bw), 1e-6)
    low = max(1e-6, float(center) - width)
    high = min(nyq * 0.98, float(center) + width)
    if not (0.0 < low < high < nyq):
        raise ValueError('invalid bandpass for requested center/bandwidth')
    sos = butter(order, [low / nyq, high / nyq], btype='bandpass', output='sos')
    return sosfiltfilt(sos, np.asarray(series, dtype=float))


def _analytic_phase(series):
    analytic = hilbert(np.asarray(series, dtype=float))
    return {
        'analytic': analytic,
        'phase': np.unwrap(np.angle(analytic)),
        'phase_wrapped': np.angle(analytic),
        'envelope': np.abs(analytic),
    }


def wrap_to_pi(values):
    arr = np.asarray(values, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def circular_resultant(phases):
    vec = np.exp(1j * np.asarray(phases, dtype=float))
    mean_vec = np.mean(vec)
    return {
        'R': float(np.abs(mean_vec)),
        'angle': float(np.angle(mean_vec)),
    }


def phase_lock_report(main_series, harm_series, fs, f0, harm_order=2, rel_bw=0.2):
    main_bp = _bandpass(main_series, fs=fs, center=f0, rel_bw=rel_bw)
    harm_bp = _bandpass(harm_series, fs=fs, center=harm_order * f0, rel_bw=rel_bw)
    main_a = _analytic_phase(main_bp)
    harm_a = _analytic_phase(harm_bp)

    delta = wrap_to_pi(harm_order * main_a['phase'] - harm_a['phase'])
    circ = circular_resultant(delta)
    return {
        'main_bandpassed': main_bp,
        'harm_bandpassed': harm_bp,
        'main_phase': main_a['phase'],
        'harm_phase': harm_a['phase'],
        'main_phase_wrapped': main_a['phase_wrapped'],
        'harm_phase_wrapped': harm_a['phase_wrapped'],
        'main_envelope': main_a['envelope'],
        'harm_envelope': harm_a['envelope'],
        'delta_phase': delta,
        'delta_phase_abs_deg': np.degrees(np.abs(delta)),
        'R': circ['R'],
        'mean_angle': circ['angle'],
        'mean_angle_deg': float(np.degrees(circ['angle'])),
        'median_abs_delta_deg': float(np.median(np.degrees(np.abs(delta)))),
    }
