"""Shared harmonic-fit helpers for BTC residual SSM workflow."""

import logging
import os
import sys

import numpy as np

from SSM_res_data import load_data
from SSM_res_embedding import build_embedding, pca

ATTRACTOR_DIR = '/home/hz/Data/Attractor'
if ATTRACTOR_DIR not in sys.path:
    sys.path.insert(0, ATTRACTOR_DIR)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
logging.getLogger('ridge_regression').setLevel(logging.WARNING)
logging.getLogger('SSMLearn').setLevel(logging.WARNING)
from ssmlearn_res import fit_ssm  # noqa: E402


DAYS_PER_YEAR = 365.25


def tau_from_years(M, years):
    return max(1, int(round(float(years) * DAYS_PER_YEAR / max(int(M) - 1, 1))))


def build_ctx(filename, M, years, start_idx):
    TAU = tau_from_years(M, years)
    data = load_data(filename, start_idx)
    D, W = build_embedding(data['signal'], int(M), TAU)
    pca_res = pca(D)
    days_vecs = data['days_emb'][W:]
    ctx = {
        'D_c': D - D.mean(axis=0),
        'Vt': pca_res.Vt,
        'pc': pca_res.pc,
        'var': pca_res.var,
        'days_vecs': days_vecs,
        'N': D.shape[0],
        'W': W,
    }
    return {
        'data': data,
        'ctx': ctx,
        'TAU': TAU,
        'pca_res': pca_res,
    }


def oscillatory_pair_indices(eigvals):
    return sorted(
        [k for k, ev in enumerate(eigvals) if ev.imag > 1e-10],
        key=lambda k: abs(eigvals[k].imag),
    )


def identify_mode_pair_indices(eigvals):
    pos_idx = oscillatory_pair_indices(eigvals)
    if len(pos_idx) < 2:
        raise ValueError('Need at least 2 oscillatory pairs for harmonic mode analysis')
    return int(pos_idx[0]), int(pos_idx[1]), pos_idx


def _fit_errors(ctx, fit, ssm_dim):
    D_obs = ctx['D_c'].T
    diff_obs = D_obs - fit['pred_obs']
    fit_err = float(np.linalg.norm(diff_obs) / (np.linalg.norm(D_obs) + 1e-30))
    dp_dt = np.gradient(ctx['pc'][:, :ssm_dim], ctx['days_vecs'].astype(float), axis=0)
    rhs = fit['ssm'].reduced_dynamics.predict(ctx['pc'][:, :ssm_dim])
    edge = max(5, ctx['N'] // 100)
    sl = slice(edge, -edge) if ctx['N'] > 2 * edge else slice(None)
    ode_err = float(np.linalg.norm(dp_dt[sl] - rhs[sl]) / (np.linalg.norm(dp_dt[sl]) + 1e-30))
    return fit_err, ode_err


def fit_summary(filename, M, years, start_idx, ssm_dim, poly_degree):
    payload = build_ctx(filename, M, years, start_idx)
    ctx = payload['ctx']
    fit = fit_ssm(ctx, ssm_dim, poly_degree=poly_degree, compute_prediction=False)
    coeffs = fit['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :ssm_dim]
    eigvals, eigvecs = np.linalg.eig(linear_part)
    pos_idx = oscillatory_pair_indices(eigvals)
    fit_err, ode_err = _fit_errors(ctx, fit, ssm_dim)
    t_days = ctx['days_vecs'].astype(float)
    return {
        'payload': payload,
        'ctx': ctx,
        'fit': fit,
        'eigvals': eigvals,
        'eigvecs': eigvecs,
        'osc_pair_indices': pos_idx,
        'pair_count': len(pos_idx),
        'fit_err': fit_err,
        'ode_err': ode_err,
        't_days': t_days,
        't_years': (t_days - t_days[0]) / DAYS_PER_YEAR,
    }


def fit_harmonic_modes(filename, M, years, start_idx, ssm_dim, poly_degree):
    summary = fit_summary(filename, M, years, start_idx, ssm_dim, poly_degree)
    payload = summary['payload']
    ctx = summary['ctx']
    fit = summary['fit']
    eigvals = summary['eigvals']
    eigvecs = summary['eigvecs']
    idx_main, idx_sub, pos_idx = identify_mode_pair_indices(eigvals)
    pc_red = ctx['pc'][:, :ssm_dim].T
    Z = np.linalg.inv(eigvecs) @ pc_red
    z_main = Z[idx_main, :]
    z_sub = Z[idx_sub, :]
    amp_main = np.abs(z_main)
    amp_sub = np.abs(z_sub)
    phi_main = np.angle(z_main)
    phi_sub = np.angle(z_sub)
    phi_main_u = np.unwrap(phi_main)
    phi_sub_u = np.unwrap(phi_sub)
    delta_unwrapped = 2.0 * phi_main_u - phi_sub_u
    delta_wrapped = np.mod(delta_unwrapped, 2.0 * np.pi)
    delta_principal = np.angle(np.exp(1j * delta_wrapped))
    mean_complex = np.mean(np.exp(1j * delta_wrapped))
    T_main = (2.0 * np.pi / abs(eigvals[idx_main].imag)) / DAYS_PER_YEAR
    T_sub = (2.0 * np.pi / abs(eigvals[idx_sub].imag)) / DAYS_PER_YEAR

    return {
        **summary,
        'idx_main': idx_main,
        'idx_sub': idx_sub,
        'T_main': float(T_main),
        'T_sub': float(T_sub),
        'ratio_vs_half': float(T_sub / (T_main / 2.0)),
        'z_main': z_main,
        'z_sub': z_sub,
        'amp_main': amp_main,
        'amp_sub': amp_sub,
        'phi_main': phi_main,
        'phi_sub': phi_sub,
        'phi_main_u': phi_main_u,
        'phi_sub_u': phi_sub_u,
        'delta_unwrapped': delta_unwrapped,
        'delta_wrapped': delta_wrapped,
        'delta_phase': delta_principal,
        'delta_principal': delta_principal,
        'R': float(np.abs(mean_complex)),
        'mean_angle_deg': float(np.degrees(np.angle(mean_complex))),
        'median_abs_delta_deg': float(np.median(np.abs(np.degrees(delta_principal)))),
    }


def scaling_metrics(amp_main, amp_sub, clip_pct):
    threshold = float(np.percentile(amp_main, clip_pct))
    mask = np.asarray(amp_main) >= threshold
    am = np.asarray(amp_main)[mask]
    asub = np.asarray(amp_sub)[mask]
    am2 = am ** 2
    slope = float(np.dot(am2, asub) / (np.dot(am2, am2) + 1e-30))
    pred_masked = slope * am2
    pred_full = slope * (np.asarray(amp_main) ** 2)
    corr = float(np.corrcoef(am2, asub)[0, 1]) if len(am2) >= 2 else float('nan')
    ss_res = float(np.sum((asub - pred_masked) ** 2))
    r2_origin = 1.0 - ss_res / (float(np.sum(asub ** 2)) + 1e-30)
    x_center = am2 - float(np.median(am2))
    y_center = asub - float(np.median(asub))
    corr_centered = float(np.corrcoef(x_center, y_center)[0, 1]) if len(am2) >= 2 else float('nan')
    return {
        'threshold': threshold,
        'threshold_main': threshold,
        'mask': mask,
        'am': am,
        'asub': asub,
        'am2': am2,
        'slope': slope,
        'slope_zero': slope,
        'pred_masked': pred_masked,
        'pred_full': pred_full,
        'corr': corr,
        'corr_centered': corr_centered,
        'r2_origin': float(r2_origin),
        'r2_zero': float(r2_origin),
        'cv_main': float(np.std(am) / (np.mean(am) + 1e-30)),
        'cv_sub': float(np.std(asub) / (np.mean(asub) + 1e-30)),
        'main_mean': float(np.mean(amp_main)),
        'main_std': float(np.std(amp_main)),
        'harm_mean': float(np.mean(amp_sub)),
        'harm_std': float(np.std(amp_sub)),
        'x_main_sq': np.asarray(amp_main) ** 2,
        'y_harm': np.asarray(amp_sub),
        'y_hat_full': pred_full,
        'scaling_identifiable': bool(np.std(am) / (np.mean(am) + 1e-30) >= 0.05),
    }
