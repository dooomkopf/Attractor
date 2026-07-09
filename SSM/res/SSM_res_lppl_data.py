"""LPPL simulation data builder for SSM/res/ pipeline.

Integrates the LPPL ODE, extracts y1(t) as observable,
builds delay embedding + PCA context identical to the BTC pipeline.
"""

import numpy as np
from scipy.integrate import solve_ivp

from SSM_res_embedding import build_embedding, pca
from lppl_system import DEFAULT_PARAMS, build_params, lppl_rhs

DAYS_PER_YEAR = 365.25

DEFAULT_T_FINAL = 20000
DEFAULT_N_EVAL = 15000
DEFAULT_IC = (0.01, 0.0, 0.0)
DEFAULT_TRANSIENT_FRAC = 0.125
DEFAULT_M = 35
DEFAULT_YEARS = 3.77


def tau_from_years(M, years):
    return max(1, round(years * DAYS_PER_YEAR / (M - 1)))


def simulate_lppl(p, t_final=DEFAULT_T_FINAL, n_eval=DEFAULT_N_EVAL,
                  ic=DEFAULT_IC, transient_frac=DEFAULT_TRANSIENT_FRAC):
    """Integrate LPPL ODE, return post-transient trajectory."""
    t_eval = np.linspace(0, t_final, n_eval)

    def rhs(t, y):
        return lppl_rhs(y, p)

    sol = solve_ivp(rhs, [0, t_final], list(ic), t_eval=t_eval,
                    method='RK45', rtol=1e-10, atol=1e-12)
    if not sol.success:
        raise RuntimeError(f"LPPL integration failed: {sol.message}")

    n_skip = int(transient_frac * len(sol.t))
    t = sol.t[n_skip:]
    traj = sol.y[:, n_skip:].T
    return t, traj


def build_lppl_context(p, M=DEFAULT_M, years=DEFAULT_YEARS,
                       t_final=DEFAULT_T_FINAL, n_eval=DEFAULT_N_EVAL,
                       ic=DEFAULT_IC, transient_frac=DEFAULT_TRANSIENT_FRAC):
    """Simulate LPPL, extract y1, delay-embed, PCA, return ctx dict.

    Returns dict compatible with fit_ssm() from ssmlearn_res.
    """
    t, traj = simulate_lppl(p, t_final, n_eval, ic, transient_frac)
    y1 = traj[:, 0]

    tau = tau_from_years(M, years)
    D, W = build_embedding(y1, M, tau)
    pca_res = pca(D)
    days_vecs = t[W:]

    ctx = {
        'D_c': D - D.mean(axis=0),
        'Vt': pca_res.Vt,
        'pc': pca_res.pc,
        'var': pca_res.var,
        'days_vecs': days_vecs,
        'N': D.shape[0],
        'W': W,
    }
    return ctx, tau, pca_res, t, traj
