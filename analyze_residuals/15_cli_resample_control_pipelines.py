"""Module for 15_cli_resample_control.py — embedding + SSM pipelines.

Embedding/PCA block 1:1 from ssmlearn_res.py:121-133 (build_residuals_and_embedding).
Master-mode picker analog 13_cli_lambda2_visualize.py:71-77 (slowest oscillatory).
"""
import os
import sys

if '/home/hz/Data/Attractor' not in sys.path:
    sys.path.insert(0, '/home/hz/Data/Attractor')

import numpy as np

from analyze_residuals.constants import DAYS_PER_YEAR
from ssmlearn_res import fit_ssm


def years_to_tau_samples(M, years, dt_days_per_sample):
    return max(1, round(years * DAYS_PER_YEAR / ((M - 1) * dt_days_per_sample)))


def embed_and_pca(values, M, TAU):
    """1:1 from ssmlearn_res.py:121-133."""
    N = len(values)
    W = (M - 1) * TAU
    if N <= W:
        raise ValueError(f'embedding window W={W} >= N={N}')
    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = values[W - j * TAU: N - j * TAU]
    D_mean = D.mean(axis=0)
    D_c = D - D_mean
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc = D_c @ Vt.T
    var = s ** 2 / (s ** 2).sum()
    return D, D_mean, D_c, Vt, pc, var, W, N


def build_linear_RS(days, log_res, M, years, N_target):
    """Resample onto uniform LINEAR grid (np.interp), then embed."""
    t_grid = np.linspace(days[0], days[-1], N_target)
    log_res_t = np.interp(t_grid, days, log_res)
    dt_per_sample = (days[-1] - days[0]) / (N_target - 1)
    TAU = years_to_tau_samples(M, years, dt_days_per_sample=dt_per_sample)
    D, D_mean, D_c, Vt, pc, var, W, N = embed_and_pca(log_res_t, M, TAU)
    days_vecs = t_grid[W:]
    return dict(label='linear-time clock',
                clock_label=r'day since Genesis (uniform RS)',
                D=D, D_mean=D_mean, D_c=D_c, Vt=Vt, pc=pc, var=var,
                M=M, TAU=TAU, W=W, N=N - W,
                days_vecs=days_vecs, fit_time_vecs=days_vecs.copy(),
                clock_x_log10=False)


def build_log_RS(days, log_res, M, years):
    """Resample onto uniform LOG10 grid + embed.

    1:1 wie analyze_residuals/data.py::_build_log_time_context (Zeile 57):
        TAU = round(years*365.25/(M-1))   # in Sample-Schritten auf log-Gitter
    Reproduziert die analyze_*-Konvention -> log-clock master Mode bei lambda=2.
    """
    if np.any(days <= 0):
        raise ValueError('log10-time pipeline requires strictly positive day values')
    tau_emb = np.log10(days)
    tau_grid = np.linspace(tau_emb[0], tau_emb[-1], len(tau_emb))
    log_res_tau = np.interp(tau_grid, tau_emb, log_res)
    TAU = max(1, round(years * DAYS_PER_YEAR / (M - 1)))
    D, D_mean, D_c, Vt, pc, var, W, N = embed_and_pca(log_res_tau, M, TAU)
    days_vecs = (10.0 ** tau_grid[W:]).astype(float)
    fit_time_vecs = tau_grid[W:].astype(float)
    return dict(label='log10-time clock',
                clock_label=r'$\log_{10}(\mathrm{day})$ since Genesis',
                D=D, D_mean=D_mean, D_c=D_c, Vt=Vt, pc=pc, var=var,
                M=M, TAU=TAU, W=W, N=N - W,
                days_vecs=days_vecs, fit_time_vecs=fit_time_vecs,
                clock_x_log10=True)


def fit_and_extract(ctx, ssm_dim, poly_degree):
    """SSM fit + master-mode extraction (slowest oscillatory pair).
    Same convention as 13_cli_lambda2_visualize.py:71-77.
    """
    time_vec = ctx['fit_time_vecs']
    res = fit_ssm(ctx, ssm_dim, poly_degree=poly_degree,
                  compute_prediction=False, time_vec=time_vec)
    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :ssm_dim]
    eigvals, eigvecs = np.linalg.eig(linear_part)
    out = dict(eigvals=eigvals)
    pos_pairs_idx = [k for k, e in enumerate(eigvals) if e.imag > 1e-12]
    complex_pairs_idx = [k for k, e in enumerate(eigvals) if abs(e.imag) > 1e-12]
    candidates = pos_pairs_idx if pos_pairs_idx else complex_pairs_idx
    if not candidates:
        out.update(idx_main=None, lam_main=None, u=None, T_main=None)
        return out
    idx_main = min(candidates, key=lambda k: abs(eigvals[k].imag))
    lam_main = eigvals[idx_main]
    pc = ctx['pc'][:, :ssm_dim].T
    V_inv = np.linalg.inv(eigvecs)
    u = (V_inv @ pc)[idx_main, :]
    out.update(idx_main=idx_main, lam_main=lam_main, u=u,
               T_main=2.0 * np.pi / abs(lam_main.imag))
    return out
