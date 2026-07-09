"""Module for 15_cli_resample_control.py — PSD via raw FFT (paper-style).

PSD via raw FFT + 4x zero-padding (no window, no DC-subtract).
1:1 from /home/hz/Data/paper-fft-bubbles-soc.py:60-73.

PC1 dominant-period periodogram: scipy.signal.find_peaks with min-distance
to avoid reporting adjacent zero-padding bins of the same peak.
"""
import numpy as np

from analyze_residuals.constants import DAYS_PER_YEAR


def psd_linear(y, days, zp_factor=4):
    """PSD on uniform daily grid via raw FFT + 4x zero-padding."""
    x_uniform = np.arange(int(days.min()), int(days.max()) + 1)
    y_uniform = np.interp(x_uniform, days, y)
    n = len(y_uniform)
    n_padded = zp_factor * n
    fft_result = np.fft.fft(y_uniform, n=n_padded)
    frequencies = np.fft.fftfreq(n_padded, d=1)
    psd = np.abs(fft_result) ** 2
    return frequencies[:n_padded // 2] * DAYS_PER_YEAR, psd[:n_padded // 2]


def psd_logtime(y, days, zp_factor=4):
    """PSD on uniform log10-day grid via raw FFT + 4x zero-padding."""
    log_d = np.log10(days)
    grid = np.linspace(log_d[0], log_d[-1], len(days))
    y_uniform = np.interp(grid, log_d, y)
    dlog = (log_d[-1] - log_d[0]) / (len(days) - 1)
    n = len(y_uniform)
    n_padded = zp_factor * n
    fft_result = np.fft.fft(y_uniform, n=n_padded)
    frequencies = np.fft.fftfreq(n_padded, d=dlog)
    psd = np.abs(fft_result) ** 2
    return frequencies[:n_padded // 2], psd[:n_padded // 2]


def pc1_dominant_period(ctx, top_k=3):
    """Periodogram of PC1 -> top_k dominant period peaks (separated, not bin-cluster)."""
    from scipy.signal import find_peaks
    pc1 = ctx['pc'][:, 0]
    days_v = ctx['days_vecs']
    if ctx['clock_x_log10']:
        log_d = np.log10(days_v)
        grid = np.linspace(log_d[0], log_d[-1], len(days_v))
        y = np.interp(grid, log_d, pc1)
        d = (log_d[-1] - log_d[0]) / (len(days_v) - 1)
    else:
        t = np.arange(int(days_v.min()), int(days_v.max()) + 1)
        y = np.interp(t, days_v, pc1)
        d = 1.0
    n = len(y)
    n_padded = 4 * n
    f = np.fft.fftfreq(n_padded, d=d)[:n_padded // 2]
    psd = np.abs(np.fft.fft(y, n=n_padded)[:n_padded // 2]) ** 2
    f, psd = f[1:], psd[1:]
    peaks, _ = find_peaks(psd, distance=max(8, n_padded // 200))
    if len(peaks) == 0:
        return ['no peak']
    order = np.argsort(psd[peaks])[::-1][:top_k]
    out = []
    for idx in order:
        k = peaks[idx]
        T = 1.0 / f[k]
        if ctx['clock_x_log10']:
            out.append(f'{T:.3f}log10d (lam={10**T:.3f})')
        else:
            out.append(f'{T:.0f}d ({T / DAYS_PER_YEAR:.2f}y)')
    return out
