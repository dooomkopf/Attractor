#!/usr/bin/env python3
"""Vergleich: LPPL-Simulation mit/ohne Wang → Delay-Embedding → SSMLearn → 3D-Plot."""

import os
import sys
import logging
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
warnings.filterwarnings("ignore")
logging.getLogger('ridge_regression').setLevel(logging.WARNING)
logging.getLogger('SSMLearn').setLevel(logging.WARNING)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ATTRACTOR = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _ATTRACTOR, os.path.join(_ATTRACTOR, 'SSMLearnPy')]:
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from SSM_res_embedding import build_embedding, pca
from ssmlearn_res import fit_ssm

try:
    plt.style.use(os.path.join(_ATTRACTOR, 'hz.mplstyle'))
except Exception:
    pass

DAYS_PER_YEAR = 365.25


def lppl_rhs(t, y, p):
    y1, y2, z = y
    dy1 = p['Z_MIX'] * y1 + (1.0 - p['Z_MIX']) * y2 - p['Z_A'] * y2 * z
    dy2 = p['alpha'] * y2 - p['gamma'] * y1**3 + p['Z_B'] * y1 * z
    dz = -p['Z_C'] * z + p['Z_D'] * y1 + p['Z_E'] * y1 * y2
    return [dy1, dy2, dz]


def simulate(p, t_span=(0, 20000), n_points=15000, ic=(0.01, 0.0, 0.0)):
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(lppl_rhs, t_span, ic, args=(p,), t_eval=t_eval, method='RK45',
                    rtol=1e-10, atol=1e-12)
    return sol.t, sol.y.T


def embed_and_fit(t, traj_y1, M=35, tau=41, ssm_dim=2, poly=2):
    D, W = build_embedding(traj_y1, M, tau)
    pca_res = pca(D)
    days = t[W:]
    ctx = {
        'D_c': D - D.mean(axis=0),
        'Vt': pca_res.Vt,
        'pc': pca_res.pc,
        'var': pca_res.var,
        'days_vecs': days,
        'N': D.shape[0],
        'W': W,
    }
    res = fit_ssm(ctx, ssm_dim, poly_degree=poly, compute_prediction=False)
    pred_obs = res['pred_obs']
    pc_pred = pred_obs.T @ pca_res.Vt.T
    fit_err = float(np.linalg.norm(ctx['D_c'].T - pred_obs) / (np.linalg.norm(ctx['D_c'].T) + 1e-30))

    slave_r2 = []
    for k in range(ssm_dim, min(8, pca_res.pc.shape[1])):
        true_k = pca_res.pc[:, k]
        pred_k = pc_pred[:, k]
        ss_res = np.sum((true_k - pred_k)**2)
        ss_tot = np.sum((true_k - true_k.mean())**2)
        slave_r2.append(1.0 - ss_res / (ss_tot + 1e-30))

    return ctx, res, pca_res, pc_pred, fit_err, slave_r2


PARAMS_WANG_ON = {
    'alpha': -0.00074, 'gamma': 0.003,
    'Z_A': 0.008, 'Z_B': 0.008, 'Z_C': 0.0039,
    'Z_D': 1e-6, 'Z_E': 2.0, 'Z_MIX': 0.0002,
}
PARAMS_WANG_OFF = {
    'alpha': -0.00074, 'gamma': 0.003,
    'Z_A': 0.0, 'Z_B': 0.0, 'Z_C': 0.0039,
    'Z_D': 1e-6, 'Z_E': 2.0, 'Z_MIX': 8e-5,
}

print("Simulating Wang ON...")
t_on, traj_on = simulate(PARAMS_WANG_ON)
print("Simulating Wang OFF...")
t_off, traj_off = simulate(PARAMS_WANG_OFF)

print(f"Wang ON:  y1 range [{traj_on[:,0].min():.4f}, {traj_on[:,0].max():.4f}]")
print(f"Wang OFF: y1 range [{traj_off[:,0].min():.4f}, {traj_off[:,0].max():.4f}]")

print("\nEmbedding + SSMLearn Wang ON...")
ctx_on, res_on, pca_on, pcpred_on, ferr_on, sr_on = embed_and_fit(t_on, traj_on[:, 0])
print(f"  fit_err={ferr_on:.4f}  slave R2: {['%.3f'%r for r in sr_on]}")

print("Embedding + SSMLearn Wang OFF...")
ctx_off, res_off, pca_off, pcpred_off, ferr_off, sr_off = embed_and_fit(t_off, traj_off[:, 0])
print(f"  fit_err={ferr_off:.4f}  slave R2: {['%.3f'%r for r in sr_off]}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})

for ax, ctx, res, pca_res, pcpred, label, ferr, sr in [
    (axes[0], ctx_on, res_on, pca_on, pcpred_on, 'Wang ON', ferr_on, sr_on),
    (axes[1], ctx_off, res_off, pca_off, pcpred_off, 'Wang OFF', ferr_off, sr_off),
]:
    pc3_true = ctx['D_c'] @ pca_res.Vt[:3].T
    pc3_dec = res['pred_obs'].T @ pca_res.Vt[:3].T
    ax.plot(pc3_true[:,0], pc3_true[:,1], pc3_true[:,2], lw=0.3, alpha=0.5, color='cyan', label='trajectory')
    ax.plot(pc3_dec[:,0], pc3_dec[:,1], pc3_dec[:,2], lw=0.3, alpha=0.5, color='red', label='decoder')
    v = pca_res.var
    ax.set_xlabel(f'PC1 ({v[0]*100:.0f}\\%%)')
    ax.set_ylabel(f'PC2 ({v[1]*100:.0f}\\%%)')
    ax.set_zlabel(f'PC3 ({v[2]*100:.0f}\\%%)')
    sr_txt = ', '.join(f'{r:.2f}' for r in sr[:3])
    ax.set_title(f'{label}\nfit_err={ferr:.3f}  PC3+ $R^2$=[{sr_txt}]')
    ax.legend(fontsize=7)

fig.suptitle('LPPL Simulation: Delay-Embedding von y1 → SSMLearn', fontsize=12)
plt.tight_layout()
plt.show()
