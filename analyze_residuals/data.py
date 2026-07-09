"""Data loading and preprocessing for residual analysis."""

import os
import sys

import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

from .constants import ATTRACTOR_DIR, DAYS_PER_YEAR
if ATTRACTOR_DIR not in sys.path:
    sys.path.insert(0, ATTRACTOR_DIR)

from ssmlearn_res import (  # noqa: E402
    build_residuals_and_embedding as _build_residuals_and_embedding,
    read_btc_data as _read_btc_data,
)

PERCENTILE = 0.01


def read_btc_data(filename):
    """Wrapper around the existing BTC data reader used by the legacy scripts."""
    return _read_btc_data(filename)


def years_to_tau(M, years):
    """Convert embedding window in years to the lag TAU used by legacy code."""
    return max(1, round(years * DAYS_PER_YEAR / (M - 1)))


def _compute_log_residuals(days_all, prices_all):
    """Compute BTC log-residuals using the existing quantile detrend."""
    log_days_all = np.log(days_all)
    log_btc_all = np.log(prices_all)
    X_all = sm.add_constant(log_days_all)
    qr = QuantReg(log_btc_all, X_all).fit(q=PERCENTILE)
    log_fit_all = qr.predict(X_all)
    return np.log(prices_all / np.exp(log_fit_all))


def _build_log_time_context(filename_abs, M, years, start_idx):
    """Resample residuals onto a uniform log-time grid before embedding."""
    days_all, prices_all, _dates_all = _read_btc_data(filename_abs)
    log_res_all = _compute_log_residuals(days_all, prices_all)

    mask_emb = days_all >= start_idx
    log_res = log_res_all[mask_emb]
    days_emb = days_all[mask_emb].astype(float)
    if len(days_emb) < max(2, M + 1):
        raise ValueError('not enough samples after start_idx for log-time embedding')

    tau_emb = np.log10(days_emb)
    tau_grid = np.linspace(tau_emb[0], tau_emb[-1], len(tau_emb))
    log_res_tau = np.interp(tau_grid, tau_emb, log_res)

    TAU = years_to_tau(M, years)
    W = (M - 1) * TAU
    N = len(log_res_tau)
    if N <= W:
        raise ValueError(
            f'log-time embedding window W={W} exceeds available resampled samples N={N}'
        )

    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = log_res_tau[W - j * TAU : N - j * TAU]

    D_mean = D.mean(axis=0)
    D_c = D - D_mean
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc = D_c @ Vt.T
    var = s**2 / (s**2).sum()

    fit_time_vecs = tau_grid[W:].astype(float)
    days_vecs = 10.0 ** fit_time_vecs
    tau0 = float(tau_grid[0])
    tau_tau = float(tau_grid[min(TAU, len(tau_grid) - 1)])
    tau_w = float(tau_grid[min(W, len(tau_grid) - 1)])
    day0 = float(10.0 ** tau0)
    return {
        'D': D,
        'D_mean': D_mean,
        'D_c': D_c,
        'Vt': Vt,
        'pc': pc,
        'var': var,
        'days_vecs': days_vecs,
        'fit_time_vecs': fit_time_vecs,
        'days_actual_vecs': days_vecs,
        'tau_grid': tau_grid,
        'M': M,
        'TAU': TAU,
        'W': W,
        'N': N - W,
        'log_res': log_res_tau,
        'days_emb': days_emb,
        'time_mode': 'log',
        'TAU_linear_days': int(TAU),
        'TAU_log_steps': int(TAU),
        'TAU_effective_start_days': float(10.0 ** tau_tau - day0),
        'W_target_days': float((M - 1) * TAU),
        'W_effective_start_days': float(10.0 ** tau_w - day0),
    }


def build_residual_context(filename, M, years, start_idx, time_mode='linear'):
    """Build the residual embedding context and return raw BTC arrays as well."""
    filename_abs = filename if os.path.isabs(filename) else os.path.join(ATTRACTOR_DIR, filename)
    TAU = years_to_tau(M, years)
    if time_mode == 'log':
        ctx = _build_log_time_context(filename_abs, M, years, start_idx)
    else:
        ctx = _build_residuals_and_embedding(filename_abs, M, TAU, start_idx)
        ctx['fit_time_vecs'] = ctx['days_vecs'].copy()
        ctx['days_actual_vecs'] = ctx['days_vecs'].copy()
        ctx['time_mode'] = 'linear'
    days_all, prices_all, dates_all = _read_btc_data(filename_abs)
    return {
        'ctx': ctx,
        'TAU': TAU,
        'filename': filename_abs,
        'days_all': days_all,
        'prices_all': prices_all,
        'dates_all': dates_all,
    }


def build_residuals_and_embedding(filename, M, years, start_idx, return_raw=False, time_mode='linear'):
    """Compatibility wrapper for callers that expect the old tuple style."""
    payload = build_residual_context(filename, M, years, start_idx, time_mode=time_mode)
    if return_raw:
        return (
            payload['ctx'],
            payload['TAU'],
            payload['days_all'],
            payload['prices_all'],
            payload['dates_all'],
        )
    return payload['ctx']
