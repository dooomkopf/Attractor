"""Module for 15_cli_resample_control.py — preprocessing (QuantReg + fracdiff).

Reproduces bf_filt.py preprocessing exactly:
  read_btc_data + QuantReg(0.01)  -> log_residuals
  FractionalDifferencing(d=0.98)  -> linear-time fracdiff
  log10-resample then fracdiff    -> log10-time fracdiff variant
"""
import os
import sys

if '/home/hz/Data/Attractor' not in sys.path:
    sys.path.insert(0, '/home/hz/Data/Attractor')
if '/home/hz/Data/BTC3' not in sys.path:
    sys.path.insert(0, '/home/hz/Data/BTC3')

import numpy as np

from analyze_residuals.data import _compute_log_residuals
from ssmlearn_res import read_btc_data
from btc_fracdiff_filter import FractionalDifferencing


def load_and_prewhiten(filename, start_idx, use_prewhiten=True,
                       fracdiff_d=0.98, trim_percent=0.0):
    """linear-time fracdiff variant: fracdiff applied on linear (mostly daily) residuals."""
    days_all, prices_all, _ = read_btc_data(filename)
    log_res_all = _compute_log_residuals(days_all, prices_all)
    if use_prewhiten:
        fd = FractionalDifferencing(d=fracdiff_d)
        log_res_pw_all = fd.filter(log_res_all)
        info = fd.get_filter_info()
    else:
        log_res_pw_all = log_res_all.copy()
        info = None
    mask = days_all >= start_idx
    days_masked = days_all[mask].astype(float)
    log_res_raw = log_res_all[mask]
    log_res_pw = log_res_pw_all[mask]
    if use_prewhiten and trim_percent > 0:
        trim_samples = int(len(log_res_pw) * trim_percent / 100)
        log_res_pw = log_res_pw[trim_samples:]
        log_res_raw = log_res_raw[trim_samples:]
        days_masked = days_masked[trim_samples:]
    else:
        trim_samples = 0
    return dict(days=days_masked, log_res_raw=log_res_raw,
                log_res_pw=log_res_pw, trim_samples=trim_samples,
                fracdiff_info=info, use_prewhiten=use_prewhiten,
                fracdiff_d=fracdiff_d, trim_percent=trim_percent)


def prewhiten_in_log_clock(days, log_res, fracdiff_d=0.98, trim_percent=0.0):
    """log10-time fracdiff variant: log10-resample, then fracdiff on tau-grid."""
    if np.any(days <= 0):
        raise ValueError('log10-time pre-whitening requires strictly positive days')
    tau = np.log10(days)
    tau_grid = np.linspace(tau[0], tau[-1], len(tau))
    y_tau = np.interp(tau_grid, tau, log_res)
    fd = FractionalDifferencing(d=fracdiff_d)
    y_tau_pw = fd.filter(y_tau)
    if trim_percent > 0:
        trim = int(len(y_tau_pw) * trim_percent / 100)
        y_tau_pw = y_tau_pw[trim:]
        tau_grid = tau_grid[trim:]
    days_back = (10.0 ** tau_grid).astype(float)
    return days_back, y_tau_pw, fd.get_filter_info()
