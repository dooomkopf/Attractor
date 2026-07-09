"""Trajectory simulation helpers for the Wang 2-scroll system."""

import numpy as np
from scipy.integrate import solve_ivp

from .constants import DEFAULT_A, DEFAULT_B, DEFAULT_C, DEFAULT_D, DEFAULT_IC, DEFAULT_TRANSIENT_FRAC
from .system import wang_rhs


def simulate_trajectory(
    a=DEFAULT_A,
    b=DEFAULT_B,
    c=DEFAULT_C,
    d=DEFAULT_D,
    ic=DEFAULT_IC,
    t_final=150.0,
    n_eval=75000,
    transient_frac=DEFAULT_TRANSIENT_FRAC,
    rtol=1e-9,
    atol=1e-11,
):
    t_eval = np.linspace(0.0, float(t_final), int(n_eval))
    diverge_limit = 1e3

    def rhs(_t, state):
        return wang_rhs(state, a, b, c, d)

    def _diverged(_t, state):
        return diverge_limit - np.max(np.abs(state))
    _diverged.terminal = True
    _diverged.direction = -1

    sol = solve_ivp(
        rhs,
        (0.0, float(t_final)),
        np.asarray(ic, dtype=float),
        method='RK45',
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=float(t_final) / float(n_eval),
        events=_diverged,
    )
    traj = sol.y.T
    keep0 = int(round(float(transient_frac) * len(traj)))
    keep0 = min(max(keep0, 0), len(traj) - 1)
    return {
        't_all': sol.t,
        'traj_all': traj,
        't': sol.t[keep0:],
        'traj': traj[keep0:],
        'dt': float(sol.t[1] - sol.t[0]),
        'transient_index': int(keep0),
    }


def principal_coordinate(traj):
    centered = np.asarray(traj, dtype=float) - np.mean(traj, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc = centered @ vt.T
    return {
        'centered': centered,
        'pc': pc,
        'vt': vt,
    }


def channel_map_with_pcs(traj):
    pcs = principal_coordinate(traj)
    return {
        'x': np.asarray(traj[:, 0], dtype=float),
        'y': np.asarray(traj[:, 1], dtype=float),
        'z': np.asarray(traj[:, 2], dtype=float),
        'pc1': np.asarray(pcs['pc'][:, 0], dtype=float),
        'pc2': np.asarray(pcs['pc'][:, 1], dtype=float),
        'pc3': np.asarray(pcs['pc'][:, 2], dtype=float),
        'pcs': pcs,
    }
