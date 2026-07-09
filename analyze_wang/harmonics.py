"""Spectral harmonic diagnostics for Wang trajectories."""

import numpy as np
from scipy.signal import find_peaks, welch


def _power_spectrum(series, fs, nperseg=None):
    arr = np.asarray(series, dtype=float)
    arr = arr - np.mean(arr)
    if nperseg is None:
        nperseg = min(8192, max(256, len(arr) // 8))
    freq, psd = welch(arr, fs=fs, window='hann', nperseg=nperseg, detrend='constant')
    return freq, psd


def dominant_frequency(series, fs, min_freq=0.01, nperseg=None):
    freq, psd = _power_spectrum(series, fs, nperseg=nperseg)
    mask = freq >= float(min_freq)
    freq_m = freq[mask]
    psd_m = psd[mask]
    if len(freq_m) == 0:
        return {'freq': None, 'psd': None, 'freq_grid': freq, 'psd_grid': psd}
    peaks, _ = find_peaks(psd_m)
    if len(peaks) == 0:
        idx = int(np.argmax(psd_m))
    else:
        idx = int(peaks[np.argmax(psd_m[peaks])])
    return {
        'freq': float(freq_m[idx]),
        'psd': float(psd_m[idx]),
        'freq_grid': freq,
        'psd_grid': psd,
    }


def harmonic_at(series, fs, f0, rel_window=0.15, nperseg=None):
    freq, psd = _power_spectrum(series, fs, nperseg=nperseg)
    if f0 is None or f0 <= 0.0:
        return {'freq': None, 'psd': None, 'ratio': None}
    target = 2.0 * float(f0)
    lo = (1.0 - rel_window) * target
    hi = (1.0 + rel_window) * target
    mask = (freq >= lo) & (freq <= hi)
    if not np.any(mask):
        return {'freq': None, 'psd': None, 'ratio': None, 'target': target}
    loc = np.argmax(psd[mask])
    freq_h = float(freq[mask][loc])
    psd_h = float(psd[mask][loc])
    idx0 = int(np.argmin(np.abs(freq - f0)))
    psd0 = float(psd[idx0])
    ratio = psd_h / max(psd0, 1e-30)
    return {
        'target': target,
        'freq': freq_h,
        'psd': psd_h,
        'ratio': float(ratio),
    }


def analyze_channels(channel_map, fs, reference_key='pc1', min_freq=0.01, rel_window=0.15):
    if reference_key not in channel_map:
        raise KeyError(f"reference channel '{reference_key}' missing")
    ref_peak = dominant_frequency(channel_map[reference_key], fs=fs, min_freq=min_freq)
    f0 = ref_peak['freq']
    rows = {}
    for key, values in channel_map.items():
        peak = dominant_frequency(values, fs=fs, min_freq=min_freq)
        harm = harmonic_at(values, fs=fs, f0=f0, rel_window=rel_window)
        rows[key] = {
            'dominant_freq': peak['freq'],
            'dominant_period': None if peak['freq'] in (None, 0.0) else float(1.0 / peak['freq']),
            'dominant_psd': peak['psd'],
            'harmonic_freq': harm['freq'],
            'harmonic_target': harm.get('target'),
            'harmonic_ratio': harm['ratio'],
            'freq_grid': peak['freq_grid'],
            'psd_grid': peak['psd_grid'],
        }
    return {
        'reference_key': reference_key,
        'reference_freq': f0,
        'reference_period': None if f0 in (None, 0.0) else float(1.0 / f0),
        'channels': rows,
    }
