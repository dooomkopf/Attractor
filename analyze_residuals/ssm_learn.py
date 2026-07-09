"""Data-driven SSM slave test on BTC residuals via SSMLearnPy."""

import os
import sys
import logging
import warnings

import numpy as np

ATTRACTOR_DIR = '/home/hz/Data/Attractor'
SSMLEARN_DIR = os.path.join(ATTRACTOR_DIR, 'SSMLearnPy')
if SSMLEARN_DIR not in sys.path:
    sys.path.insert(0, SSMLEARN_DIR)
if ATTRACTOR_DIR not in sys.path:
    sys.path.insert(0, ATTRACTOR_DIR)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
warnings.filterwarnings("ignore")
logging.getLogger('ridge_regression').setLevel(logging.WARNING)
logging.getLogger('SSMLearn').setLevel(logging.WARNING)

from ssmlearn_res import fit_ssm  # noqa: E402


def run_slave_test(ctx, ssm_dim, poly_degree, max_slave_pc=8, time_vec=None, time_mode='linear'):
    """Fit SSM and test if higher PCs are polynomial slaves of master PCs.

    Returns dict with fit results and per-PC R^2 values.
    """
    res = fit_ssm(
        ctx,
        ssm_dim,
        poly_degree=poly_degree,
        compute_prediction=False,
        time_vec=time_vec,
    )

    pc = ctx['pc']
    pred_obs = res['pred_obs']
    Vt = ctx['Vt']
    D_c = ctx['D_c']

    D_obs = D_c.T
    diff = D_obs - pred_obs
    fit_err = float(np.linalg.norm(diff) / (np.linalg.norm(D_obs) + 1e-30))

    pc_pred = pred_obs.T @ Vt.T

    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :ssm_dim]
    eigvals = np.linalg.eigvals(linear_part)

    slave_results = []
    n_pcs = min(max_slave_pc, pc.shape[1])
    for k in range(ssm_dim, n_pcs):
        true_k = pc[:, k]
        pred_k = pc_pred[:, k]
        ss_res = np.sum((true_k - pred_k) ** 2)
        ss_tot = np.sum((true_k - true_k.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-30)
        rms_true = float(np.sqrt(np.mean(true_k ** 2)))
        rms_err = float(np.sqrt(np.mean((true_k - pred_k) ** 2)))
        if rms_true > 1e-30:
            verdict = 'HARM.SLAVED' if r2 >= 0.70 else ('PART.SLAVED' if r2 >= 0.07 else 'INDEPENDENT')
        else:
            verdict = 'ZERO'
        slave_results.append({
            'pc': k,
            'r2': float(r2),
            'rms_true': rms_true,
            'rms_err': rms_err,
            'verdict': verdict,
        })

    return {
        'ssm_dim': ssm_dim,
        'poly_degree': poly_degree,
        'time_mode': time_mode,
        'fit_err': fit_err,
        'eigvals': eigvals,
        'slave_results': slave_results,
        'pc_pred': pc_pred,
        'res': res,
    }


def format_slave_test(result, ctx):
    lines = []
    lines.append("SSMLEARN SLAVE TEST")
    lines.append(f"  ssm_dim           : {result['ssm_dim']}")
    lines.append(f"  poly              : {result['poly_degree']}")
    lines.append(f"  time_mode         : {result['time_mode']}")
    lines.append(f"  N samples         : {ctx['N']}")
    lines.append(f"  embedding dim     : {ctx['D_c'].shape[1]}")
    cum_var = float(np.cumsum(ctx['var'])[result['ssm_dim'] - 1])
    lines.append(f"  cumulative var    : {100.0 * cum_var:.2f}%")
    lines.append(f"  fit error (Frob)  : {result['fit_err']:.4e}")
    lines.append("")

    eig = result['eigvals']
    period_unit = 'y' if result['time_mode'] == 'linear' else 'fit'
    period_scale = 365.25 if result['time_mode'] == 'linear' else 1.0
    lines.append("  FITTED EIGENVALUES (reduced dynamics linear part):")
    for k, ev in enumerate(sorted(eig, key=lambda z: (-z.real, abs(z.imag)))):
        if abs(ev.imag) > 1e-6:
            period = (2.0 * np.pi / abs(ev.imag)) / period_scale
            lines.append(f"    [{k}] Re={ev.real:+.4e}  Im={ev.imag:+.4e}  T={period:.3f}{period_unit}")
        else:
            lines.append(f"    [{k}] Re={ev.real:+.4e} (real)")
    lines.append("")

    lines.append("  SLAVE TEST (PC_k from polynomial decoder of PC1..PC_{ssm_dim}):")
    for sr in result['slave_results']:
        bar = '#' * int(min(30, max(0, sr['r2']) * 30))
        lines.append(
            f"    PC{sr['pc']+1:2d}: R^2={sr['r2']:+.4f}  "
            f"rms_true={sr['rms_true']:.3e}  rms_err={sr['rms_err']:.3e}  "
            f"{sr['verdict']:11s}  {bar}"
        )

    slaved = [sr for sr in result['slave_results'] if sr['verdict'] == 'HARM.SLAVED']
    partial = [sr for sr in result['slave_results'] if sr['verdict'] == 'PART.SLAVED']
    indep = [sr for sr in result['slave_results'] if sr['verdict'] == 'INDEPENDENT']
    lines.append("")
    lines.append(f"  SUMMARY: {len(slaved)} harm.slaved, {len(partial)} part.slaved, {len(indep)} independent")
    lines.append(f"  NOTE: part.slaved = harmonically coupled to master (NOT an independent frequency)")

    return lines
