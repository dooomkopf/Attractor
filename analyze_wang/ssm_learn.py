"""Data-driven SSM fit on Wang trajectory via SSMLearnPy."""

import os
import sys
import logging
import warnings

import numpy as np

ATTRACTOR_DIR = '/home/hz/Data/Attractor'
SSMLEARN_DIR = os.path.join(ATTRACTOR_DIR, 'SSMLearnPy')
if SSMLEARN_DIR not in sys.path:
    sys.path.insert(0, SSMLEARN_DIR)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
warnings.filterwarnings("ignore")
logging.getLogger('ridge_regression').setLevel(logging.WARNING)
logging.getLogger('SSMLearn').setLevel(logging.WARNING)

from ssmlearnpy import SSMLearn  # noqa: E402
from ssmlearnpy.geometry.dimensionality_reduction import LinearChart  # noqa: E402


def prepare_wang_data(sim):
    """Build PCA-based context from Wang (x,y,z) trajectory.

    No delay embedding — full 3D state is the observable.
    """
    traj = np.array(sim['traj'])
    t = np.array(sim['t'], dtype=float)
    N = traj.shape[0]
    obs_dim = traj.shape[1]

    D_c = traj - traj.mean(axis=0)
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc = D_c @ Vt.T
    var = s ** 2 / (s ** 2).sum()

    return {
        'traj': traj,
        't': t,
        'D_c': D_c,
        'Vt': Vt,
        'pc': pc,
        'var': var,
        'N': N,
        'obs_dim': obs_dim,
    }


def fit_ssm_wang(ctx, ssm_dim, poly_degree):
    """Fit SSMLearnPy on Wang (x,y,z) data."""
    D_c = ctx['D_c']
    Vt = ctx['Vt']
    pc = ctx['pc']
    t = ctx['t']

    ssm = SSMLearn(
        t=[t],
        x=[D_c.T],
        ssm_dim=ssm_dim,
        derive_embdedding=False,
        reduced_coordinates=[pc[:, :ssm_dim].T],
        dynamics_type='flow',
        dynamics_structure='generic',
    )
    ssm.encoder = LinearChart(
        n_dim=ssm_dim,
        matrix_representation=Vt[:ssm_dim, :].T,
    )
    ssm.get_parametrization(poly_degree=poly_degree, alpha=0)
    ssm.get_reduced_dynamics(poly_degree=poly_degree, alpha=0)

    eigenvalues = ssm.reduced_dynamics.map_info['eigenvalues_linear_part']
    pred_obs = ssm.decode(pc[:, :ssm_dim].T)

    D_obs = D_c.T
    diff = D_obs - pred_obs
    fit_err_frob = float(np.linalg.norm(diff) / (np.linalg.norm(D_obs) + 1e-30))

    return {
        'ssm': ssm,
        'eigenvalues': eigenvalues,
        'pred_obs': pred_obs,
        'fit_err_frobenius': fit_err_frob,
        'poly_degree': poly_degree,
        'ssm_dim': ssm_dim,
    }


def slave_test(ctx, fit_result):
    """Test if higher PCs are reconstructable from master PCs via the decoder.

    Returns R^2 per slave PC.
    """
    pc = ctx['pc']
    pred_obs = fit_result['pred_obs']
    Vt = ctx['Vt']
    ssm_dim = fit_result['ssm_dim']
    obs_dim = ctx['obs_dim']

    pc_pred = (pred_obs.T - ctx['D_c'].mean(axis=0)) @ Vt.T
    # Actually pred_obs is in mean-centered space, project back to PC
    pc_pred = pred_obs.T @ Vt.T

    results = []
    for k in range(ssm_dim, obs_dim):
        true_k = pc[:, k]
        pred_k = pc_pred[:, k]
        ss_res = np.sum((true_k - pred_k) ** 2)
        ss_tot = np.sum((true_k - true_k.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-30)
        results.append({
            'pc_index': k,
            'r2': float(r2),
            'rms_true': float(np.sqrt(np.mean(true_k ** 2))),
            'rms_error': float(np.sqrt(np.mean((true_k - pred_k) ** 2))),
        })
    return results


def format_learn_results(ctx, fit_result, slave_results):
    """Return list of print lines."""
    lines = []
    lines.append("SSMLEARN FIT RESULTS")
    lines.append(f"  ssm_dim           : {fit_result['ssm_dim']}")
    lines.append(f"  poly_degree       : {fit_result['poly_degree']}")
    lines.append(f"  N samples         : {ctx['N']}")
    lines.append(f"  observable dim    : {ctx['obs_dim']} (x, y, z)")
    lines.append(f"  PCA variance      : {', '.join(f'{v:.4f}' for v in ctx['var'])}")
    lines.append(f"  fit error (Frob)  : {fit_result['fit_err_frobenius']:.4e}")
    lines.append("")

    eig = fit_result['eigenvalues']
    lines.append("  FITTED EIGENVALUES (reduced dynamics linear part):")
    for k, ev in enumerate(eig):
        if abs(ev.imag) > 1e-6:
            T = 2.0 * np.pi / abs(ev.imag)
            lines.append(f"    [{k}] {ev.real:+.6f}{ev.imag:+.6f}i  T={T:.4f}")
        else:
            lines.append(f"    [{k}] {ev.real:+.6f} (real)")
    lines.append("")

    lines.append("  SLAVE TEST (is PC_k reconstructable from PC1..PC_{ssm_dim}?):")
    for sr in slave_results:
        verdict = 'HARM.SLAVED' if sr['r2'] >= 0.70 else ('PART.SLAVED' if sr['r2'] >= 0.30 else 'INDEPENDENT')
        lines.append(f"    PC{sr['pc_index']+1}: R^2={sr['r2']:.4f}  "
                     f"rms_true={sr['rms_true']:.3e}  rms_err={sr['rms_error']:.3e}  "
                     f"-> {verdict}")

    return lines
