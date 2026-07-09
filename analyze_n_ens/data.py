"""Data loading and embedding context for ensemble-n SSM analysis.

Analog zu analyze_residuals/data.py, aber Signal ist die Ensemble-gemittelte
LPPL-Exponentenkurve n(t) statt der log-Residuen der BTC-Preise.

Wiederverwendung:
- attractor_n_ens.attractor_n_ens_compute.compute_ensemble_n_signal
- attractor_n_ens.attractor_n_ens_compute.build_embedding_context_from_signal (LINEAR)
- ssmlearn_res.read_btc_data

Eigene Logik nur fuer log10-Mode (Resampling + manuelles Embedding/PCA mit
denselben PCA-Sign/Rotations-Konventionen wie der Linear-Pfad).
"""

import os
import sys

import numpy as np

from .constants import ATTRACTOR_DIR, DAYS_PER_YEAR, WINDOW_SIZES

if ATTRACTOR_DIR not in sys.path:
    sys.path.insert(0, ATTRACTOR_DIR)

from ssmlearn_res import read_btc_data as _read_btc_data  # noqa: E402
from attractor_n_ens.attractor_n_ens_compute import (  # noqa: E402
    _apply_pca_conventions,
    build_embedding_context_from_signal,
    compute_ensemble_n_signal,
)


def years_to_tau(M, years):
    """Convert embedding window in years to lag TAU (in samples)."""
    return max(1, round(years * DAYS_PER_YEAR / (M - 1)))


def _embed(series, M, tau):
    series = np.asarray(series, dtype=float)
    W = (M - 1) * tau
    if len(series) <= W:
        raise ValueError(f'series too short for embedding: len={len(series)} <= W={W}')
    N = len(series)
    D = np.empty((N - W, M), dtype=float)
    for j in range(M):
        D[:, j] = series[W - j * tau: N - j * tau]
    return D, W


def compute_n_log_uniform_signal(days_all, prices_all, target_mean_w_days):
    """n(t) over a log-uniform window: W(t) = c*t with mean(W) = target_mean_w_days.

    For each day t where window fits inside the data, compute
        n(t) = ( log p(t2) - log p(t1) ) / ( log t2 - log t1 )
    with t1 = t - W/2, t2 = t + W/2.

    Returns (n_signal_array_same_len_as_days_all, half_max, c).
    """
    days = np.asarray(days_all, dtype=float)
    prices = np.asarray(prices_all, dtype=float)
    log_p = np.log(prices)

    pos = days > 0
    if not pos.any():
        raise ValueError('all days <= 0; log-uniform window undefined')

    c = target_mean_w_days / float(np.mean(days[pos]))
    for _ in range(20):
        Ws = c * days
        t1s = days - Ws / 2
        t2s = days + Ws / 2
        valid_mask = (t1s >= days[0]) & (t2s <= days[-1]) & (t1s > 0)
        if not valid_mask.any():
            break
        c_new = target_mean_w_days / float(np.mean(days[valid_mask]))
        if abs(c_new - c) < 1e-8:
            c = c_new
            break
        c = c_new

    n_signal = np.full_like(days, np.nan, dtype=float)
    for i in range(len(days)):
        t = days[i]
        if t <= 0:
            continue
        W = c * t
        t1 = t - W / 2.0
        t2 = t + W / 2.0
        if t1 <= 0 or t1 < days[0] or t2 > days[-1]:
            continue
        lp1 = float(np.interp(t1, days, log_p))
        lp2 = float(np.interp(t2, days, log_p))
        denom = np.log(t2) - np.log(t1)
        if denom <= 0:
            continue
        n_signal[i] = (lp2 - lp1) / denom

    half_max = int(np.ceil(c * float(days.max()) / 2.0))
    return n_signal, half_max, c


def compute_n_log_uniform_ensemble_signal(days_all, prices_all, target_mean_w_list):
    """Average n(t) over several log-uniform windows, each with its own mean width.

    For each mean in target_mean_w_list, build a log-uniform-window n(t).
    Take the per-day nanmean across the list. Returns
    (n_signal, half_max_max, c_list).
    """
    n_stack = []
    half_max_max = 0
    c_list = []
    for mean_w in target_mean_w_list:
        n_k, hm_k, c_k = compute_n_log_uniform_signal(
            days_all, prices_all, float(mean_w),
        )
        n_stack.append(n_k)
        c_list.append(c_k)
        if hm_k > half_max_max:
            half_max_max = hm_k
    n_arr = np.vstack(n_stack)
    n_mean = np.nanmean(n_arr, axis=0)
    return n_mean, int(half_max_max), c_list


def build_n_log_uniform_context(filename, M, years, start_idx,
                                 target_mean_w_days, time_mode='log',
                                 phase_offset=0.0):
    """Build embedding+PCA context using log-uniform window n(t).

    time_mode='log'    -> log10-resampled embedding (SSM B)
    time_mode='linear' -> linear-time embedding (SSM A with same n(t))
    """
    filename_abs = filename if os.path.isabs(filename) else os.path.join(ATTRACTOR_DIR, filename)
    days_all, prices_all, dates_all = _read_btc_data(filename_abs)
    days_all = np.asarray(days_all, dtype=float)
    prices_all = np.asarray(prices_all, dtype=float)

    if hasattr(target_mean_w_days, '__iter__'):
        mean_list = [float(x) for x in target_mean_w_days]
        if len(mean_list) == 1:
            daily_n_all, half_max, c = compute_n_log_uniform_signal(
                days_all, prices_all, mean_list[0],
            )
            c_record = float(c)
        else:
            daily_n_all, half_max, c_list = compute_n_log_uniform_ensemble_signal(
                days_all, prices_all, mean_list,
            )
            c_record = list(c_list)
    else:
        daily_n_all, half_max, c = compute_n_log_uniform_signal(
            days_all, prices_all, float(target_mean_w_days),
        )
        c_record = float(c)

    TAU = years_to_tau(M, years)
    if time_mode == 'log':
        ctx = _build_log_n_ens_context(
            daily_n_all, days_all, M, TAU, start_idx, int(half_max),
            phase_offset=phase_offset,
        )
    elif time_mode == 'linear':
        ctx = build_embedding_context_from_signal(
            daily_n_all, days_all, M, TAU, start_idx,
            half_max=int(half_max), phase_offset=phase_offset,
        )
    else:
        raise ValueError(f'unsupported time_mode={time_mode!r}')

    days_vecs = np.asarray(ctx['days_vecs'], dtype=float)
    if time_mode == 'log':
        ctx['fit_time_vecs'] = np.log10(days_vecs)
    else:
        ctx['fit_time_vecs'] = days_vecs.copy()
    ctx['days_actual_vecs'] = days_vecs.copy()
    ctx['time_mode'] = time_mode
    ctx['half_max'] = int(half_max)
    ctx['log_uniform_c'] = c_record
    ctx['log_uniform_target_mean'] = (
        list(target_mean_w_days) if hasattr(target_mean_w_days, '__iter__')
        else float(target_mean_w_days)
    )
    ctx['daily_n_all'] = daily_n_all
    ctx['days_all'] = days_all

    return {
        'ctx': ctx,
        'TAU': TAU,
        'filename': filename_abs,
        'days_all': days_all,
        'prices_all': prices_all,
        'dates_all': dates_all,
    }


def build_n_log_uniform_log_context(filename, M, years, start_idx,
                                     target_mean_w_days, phase_offset=0.0):
    """Backward-compat wrapper -> build_n_log_uniform_context(time_mode='log')."""
    return build_n_log_uniform_context(filename, M, years, start_idx,
                                        target_mean_w_days, time_mode='log',
                                        phase_offset=phase_offset)


def _build_log_n_ens_context(daily_n_all, days_all, M, TAU, start_idx, half_max,
                              phase_offset=0.0):
    """log10-resampling pendant to build_embedding_context_from_signal."""
    mask_emb = (days_all >= start_idx - half_max) & np.isfinite(daily_n_all)
    sig = daily_n_all[mask_emb]
    days_emb = days_all[mask_emb]

    if len(days_emb) == 0 or days_emb[0] <= 0:
        raise ValueError('log10 mode requires positive days after masking')

    tau_emb = np.log10(days_emb)
    tau_grid = np.linspace(tau_emb[0], tau_emb[-1], len(tau_emb))
    sig_resampled = np.interp(tau_grid, tau_emb, sig)
    days_resampled = (10.0 ** tau_grid).astype(float)

    D, W = _embed(sig_resampled, M, TAU)
    D_mean = D.mean(axis=0)
    D_c = D - D_mean
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc = D_c @ Vt.T
    pc, Vt = _apply_pca_conventions(pc, Vt, phase_offset=phase_offset)
    var = s ** 2 / (s ** 2).sum()

    return {
        'D': D,
        'D_mean': D_mean,
        'D_c': D_c,
        'Vt': Vt,
        'pc': pc,
        'var': var,
        'days_vecs': days_resampled[W:].astype(float),
        'days_emb': days_resampled,
        'mask_emb': mask_emb,
        'M': M,
        'TAU': TAU,
        'W': W,
        'N': D.shape[0],
        'phase_offset': phase_offset,
    }


def build_n_ens_context(filename, M, years, start_idx, time_mode='linear',
                         window_sizes=None, phase_offset=0.0):
    """Build embedding+PCA context for the ensemble-n signal.

    Returns same dict shape as analyze_residuals.data.build_residual_context,
    so analyze_residuals.ssm_learn.run_slave_test can be invoked unchanged.

    time_mode='linear': delegates to build_embedding_context_from_signal
        (existing PCA conventions preserved).
    time_mode='log':    log10-resamples the n(t) signal first, then embeds with
        the same conventions.
    """
    filename_abs = filename if os.path.isabs(filename) else os.path.join(ATTRACTOR_DIR, filename)
    days_all, prices_all, dates_all = _read_btc_data(filename_abs)
    days_all = np.asarray(days_all, dtype=float)
    prices_all = np.asarray(prices_all, dtype=float)

    ws = window_sizes if window_sizes is not None else WINDOW_SIZES
    daily_n_all, half_max = compute_ensemble_n_signal(days_all, prices_all, ws)

    TAU = years_to_tau(M, years)

    if time_mode == 'linear':
        ctx = build_embedding_context_from_signal(
            daily_n_all, days_all, M, TAU, start_idx,
            half_max=int(half_max), phase_offset=phase_offset,
        )
    elif time_mode == 'log':
        ctx = _build_log_n_ens_context(
            daily_n_all, days_all, M, TAU, start_idx, int(half_max),
            phase_offset=phase_offset,
        )
    else:
        raise ValueError(f'unsupported time_mode={time_mode!r}')

    # extra keys for downstream visualisation / SSMLearn
    # FIX: for log-mode, fit_time_vecs must be the log10(day) grid that the
    # SSM was actually fitted on; days_actual_vecs stays in linear days.
    days_vecs = np.asarray(ctx['days_vecs'], dtype=float)
    if time_mode == 'log':
        ctx['fit_time_vecs'] = np.log10(days_vecs)
    else:
        ctx['fit_time_vecs'] = days_vecs.copy()
    ctx['days_actual_vecs'] = days_vecs.copy()
    ctx['time_mode'] = time_mode
    ctx['half_max'] = int(half_max)
    ctx['window_sizes'] = list(ws)
    ctx['daily_n_all'] = daily_n_all
    ctx['days_all'] = days_all

    return {
        'ctx': ctx,
        'TAU': TAU,
        'filename': filename_abs,
        'days_all': days_all,
        'prices_all': prices_all,
        'dates_all': dates_all,
    }
