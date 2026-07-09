#!/usr/bin/env python3
"""03b: Phase-Lock test fuer ALLE oszillatorischen Paare in SSM B (log10-clock).

Nur SSM B (log10-time). Fuer linear-clock siehe 03_cli_phase.py.

- ssm_dim default = 9 (4 oszillatorische Paare moeglich)
- Extrahiert alle komplex-konjugierten Eigenwert-Paare
- Berechnet paarweise Phase-Lock R fuer jedes (i, j)-Paar
- Plot: Phasen aller Moden + R-Matrix Heatmap + Amplitudenverlauf

CLI:
  ./03b_cli_phase_all.py --ssm_dim 9 --poly 2
"""

import argparse
import logging
import os
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
warnings.filterwarnings("ignore")
logging.getLogger('ridge_regression').setLevel(logging.WARNING)
logging.getLogger('SSMLearn').setLevel(logging.WARNING)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
if '/home/hz/Data/Attractor' not in sys.path:
    sys.path.insert(0, '/home/hz/Data/Attractor')

HZ_MPLSTYLE = '/home/hz/Data/hz.mplstyle'
if os.path.exists(HZ_MPLSTYLE):
    plt.style.use(HZ_MPLSTYLE)
    mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']

from analyze_residuals.constants import (  # noqa: E402
    DEFAULT_FILENAME, DEFAULT_M, DEFAULT_YEARS, START_IDX, DAYS_PER_YEAR,
)
from analyze_residuals.data import build_residual_context  # noqa: E402
from analyze_residuals.common import analysis_time_vector, smooth_phase_series, smooth_real_series  # noqa: E402
from ssmlearn_res import fit_ssm  # noqa: E402

LEGEND_KW = dict(loc='upper left', fontsize=10, facecolor='#1A1A1A',
                 edgecolor='#808080', labelcolor='#E0E0E0', framealpha=0.85)


def main():
    ap = argparse.ArgumentParser(
        description='Phase-Lock test for ALL oscillatory pairs in SSM B (log10-clock)')
    ap.add_argument('--filename', default=DEFAULT_FILENAME)
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--start_idx', type=int, default=START_IDX)
    ap.add_argument('--ssm_dim', type=int, default=9)
    ap.add_argument('--poly', dest='poly_degree', type=int, default=2)
    ap.add_argument('--harmonic_k', type=int, default=None,
                    help='Polar plot: override which harmonic k to plot vs master. '
                         'Default = auto (lowest k from real-harmonic classification).')
    ap.add_argument('--no-show', action='store_true')
    ap.add_argument('--save', type=str, default=None)
    args = ap.parse_args()
    args.time_mode = 'log'  # SSM B only — hardcoded
    args.smooth_days = 180.0  # not used in log-clock; kept for compat

    payload = build_residual_context(args.filename, args.M, args.years, args.start_idx,
                                      time_mode=args.time_mode)
    ctx = payload['ctx']
    time_vec = analysis_time_vector(ctx)
    res = fit_ssm(ctx, args.ssm_dim, poly_degree=args.poly_degree,
                  compute_prediction=False, time_vec=time_vec)

    coeffs = res['ssm'].reduced_dynamics.map_info['coefficients']
    linear_part = coeffs[:, :args.ssm_dim]
    eigvals, eigvecs = np.linalg.eig(linear_part)

    # Pick positive-imag oscillatory eigenvalues, sorted by |omega| (slowest first)
    pos_idx = sorted(
        [k for k, ev in enumerate(eigvals) if ev.imag > 1e-10],
        key=lambda k: abs(eigvals[k].imag),
    )
    if len(pos_idx) < 2:
        raise RuntimeError(
            f'Need >=2 oscillatory pairs, found {len(pos_idx)} '
            f'(try --ssm_dim larger)')

    period_unit = 'y' if args.time_mode == 'linear' else r'$\log_{10}$d'
    period_scale = DAYS_PER_YEAR if args.time_mode == 'linear' else 1.0
    pairs = []
    for k in pos_idx:
        e = eigvals[k]
        T = (2.0 * np.pi / abs(e.imag)) / period_scale
        if args.time_mode == 'log':
            lam = 10.0 ** T
            tag = f'$T={T:.3f},\\,\\lambda={lam:.3f}$'
        else:
            tag = f'$T={T:.2f}$ y'
        pairs.append({'idx': k, 'mu': e, 'T': T, 'tag': tag})

    # Compute mode trajectories z_k(t) = (V^-1 @ pc)_k for each oscillatory mode
    pc = ctx['pc'][:, :args.ssm_dim].T
    V_inv = np.linalg.inv(eigvecs)
    Z = V_inv @ pc

    fit_time = np.asarray(time_vec, dtype=float)
    if args.time_mode == 'linear':
        t_plot = (fit_time - fit_time[0]) / DAYS_PER_YEAR
        x_unit = 'years since start'
    else:
        t_plot = fit_time - fit_time[0]
        x_unit = r'$\log_{10}$d shift'
    dt = float(np.median(np.diff(fit_time)))
    if args.time_mode == 'linear':
        sw = max(1, int(round(args.smooth_days / max(dt, 1e-12))))
    else:
        # log-clock: dt is in log10(d) units; use ~5% of trajectory length
        sw = max(1, len(fit_time) // 20)

    n_pairs = len(pairs)
    # Pre-compute harmonic classification (master = mode 0)
    om0_im = pairs[0]['mu'].imag
    z0 = Z[pairs[0]['idx'], :]
    phi_0_full = np.unwrap(np.angle(z0))
    amp0 = np.abs(z0)
    classifications = []
    for i in range(1, n_pairs):
        zi = Z[pairs[i]['idx'], :]
        phi_i = np.unwrap(np.angle(zi))
        ampi = np.abs(zi)
        ratio = pairs[i]['mu'].imag / om0_im
        for k in range(2, 9):
            dev = abs(ratio - k) / k
            if dev > 0.20:
                continue
            psi = np.mod(k * phi_0_full - phi_i, 2.0 * np.pi)
            R_k = float(np.abs(np.mean(np.exp(1j * psi))))
            amp_corr = float(np.corrcoef(amp0 ** k, ampi)[0, 1])
            is_harm = (dev < 0.15)
            if not is_harm:
                verdict = '- ratio not integer'
            else:
                if R_k > 0.5:
                    verdict = '*** HARMONIC, SLAVED (strong phase-lock)'
                elif R_k > 0.3:
                    verdict = '** HARMONIC, slaved (moderate lock)'
                elif R_k > 0.15:
                    verdict = '** HARMONIC, weak lock'
                else:
                    verdict = '*** HARMONIC, INDEPENDENT (DSI candidate)'
                if amp_corr > 0.2:
                    verdict += '  +amp-coupled'
                elif amp_corr < -0.2:
                    verdict += '  +amp-anti'
            classifications.append({
                'mode': i, 'k': k, 'R': R_k, 'amp_corr': amp_corr,
                'dev': dev, 'verdict': verdict, 'is_harm': is_harm,
            })

    print('=' * 84)
    print(f'PHASE-LOCK ALL PAIRS  --  ssm_dim={args.ssm_dim}  poly={args.poly_degree}  '
          f'time_mode={args.time_mode}')
    print('=' * 84)
    print(f'  found {n_pairs} oscillatory pairs')
    for i, p in enumerate(pairs):
        tag_clean = p['tag'].replace('$', '').replace('\\,', '  ')
        print(f'    [{i}] T={p["T"]:.4f}  alpha={p["mu"].real:+.3e}  omega={p["mu"].imag:+.3e}')
    print()

    # Build pairwise R-Matrix using psi_ij = phi_i - (omega_i/omega_j) * phi_j
    # For exact harmonics omega_i = k * omega_j, psi reduces to k*phi_j - phi_i.
    # Generic version: rotate sub by ratio omega_i/omega_j to align
    R_mat = np.full((n_pairs, n_pairs), np.nan)
    drift_mat = np.full((n_pairs, n_pairs), np.nan)
    for i in range(n_pairs):
        for j in range(n_pairs):
            if i == j:
                R_mat[i, i] = 1.0
                continue
            mu_i = pairs[i]['mu']
            mu_j = pairs[j]['mu']
            phi_i = np.unwrap(np.angle(Z[pairs[i]['idx'], :]))
            phi_j = np.unwrap(np.angle(Z[pairs[j]['idx'], :]))
            ratio = float(mu_i.imag / mu_j.imag)
            psi = phi_i - ratio * phi_j
            psi_w = np.mod(psi, 2.0 * np.pi)
            R = float(np.abs(np.mean(np.exp(1j * psi_w))))
            R_mat[i, j] = R
            A = np.vstack([fit_time, np.ones_like(fit_time)]).T
            slope, _ = np.linalg.lstsq(A, psi, rcond=None)[0], None
            slope, intercept = np.linalg.lstsq(A, psi, rcond=None)[0]
            drift_mat[i, j] = slope

    print('PAIRWISE PHASE-LOCK R (psi_ij = phi_i - (om_i/om_j)*phi_j)')
    header = '       ' + '  '.join(f' [{j}]   ' for j in range(n_pairs))
    print(header)
    for i in range(n_pairs):
        row = f'  [{i}]  ' + '  '.join(f'{R_mat[i, j]:6.3f}  ' for j in range(n_pairs))
        print(row)
    print()
    print('PAIRWISE DRIFT (rad / time_unit)')
    print(header)
    for i in range(n_pairs):
        row = f'  [{i}]  ' + '  '.join(
            f'{drift_mat[i, j]:+8.3e}' if not np.isnan(drift_mat[i, j]) else '   nan   '
            for j in range(n_pairs)
        )
        print(row)
    print('=' * 84)

    # ---- Plot ----
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95],
                          width_ratios=[1.4, 1.0, 1.0])
    ax_phi = fig.add_subplot(gs[0, :])
    ax_amp = fig.add_subplot(gs[1, :])
    ax_tbl = fig.add_subplot(gs[2, 0])
    ax_R = fig.add_subplot(gs[2, 1])
    ax_pol = fig.add_subplot(gs[2, 2], projection='polar')

    # Per-mode classification: HARMONIC (integer ratio) marked prominently
    mode_class_label = {0: 'MASTER'}
    mode_is_harm = {0: True}
    harm_only = [c for c in classifications if c['is_harm']]
    per_mode_class = {}
    for c in harm_only:
        if c['mode'] not in per_mode_class or c['dev'] < per_mode_class[c['mode']]['dev']:
            per_mode_class[c['mode']] = c
    for i, c in per_mode_class.items():
        if 'INDEPENDENT' in c['verdict']:
            tag = f"{c['k']}. HARM INDEP"
        elif 'SLAVED' in c['verdict'].upper():
            tag = f"{c['k']}. HARM SLAVE"
        elif 'weak' in c['verdict']:
            tag = f"{c['k']}. HARM weak"
        else:
            tag = f"{c['k']}. HARM ?"
        mode_class_label[i] = tag
        mode_is_harm[i] = True
    for i in range(n_pairs):
        if i not in mode_class_label:
            mode_class_label[i] = 'no integer match'
            mode_is_harm[i] = False

    cmap = plt.get_cmap('plasma')
    for i, p in enumerate(pairs):
        col = cmap(0.1 + 0.85 * i / max(1, n_pairs - 1))
        z = Z[p['idx'], :]
        phi_unwrapped = np.unwrap(np.angle(z))
        amp = np.abs(z)
        cls = mode_class_label.get(i, '?')
        is_h = mode_is_harm.get(i, False)
        lw = 2.0 if is_h else 0.8
        alpha_line = 1.0 if is_h else 0.5
        label = f'mode {i} [{cls}]: ' + p['tag']
        phi_smooth = smooth_real_series(phi_unwrapped, sw)
        amp_smooth = smooth_real_series(amp, sw)
        ax_phi.plot(t_plot, phi_smooth, color=col, lw=lw, alpha=alpha_line, label=label)
        ax_amp.plot(t_plot, amp_smooth, color=col, lw=lw, alpha=alpha_line, label=label)

    ax_phi.set_xlabel(x_unit, fontsize=12)
    ax_phi.set_ylabel(r'unwrapped phase $\phi_k(t)$ [rad]', fontsize=12)
    ax_phi.set_title('SSM B — Phase trajectories of all oscillatory modes (log10-clock)')
    ax_phi.tick_params(labelsize=11)
    ax_phi.legend(**LEGEND_KW)

    ax_amp.set_xlabel(x_unit, fontsize=12)
    ax_amp.set_ylabel(r'amplitude $|z_k(t)|$', fontsize=12)
    ax_amp.set_title('Mode amplitudes (smoothed)')
    ax_amp.tick_params(labelsize=11)
    ax_amp.legend(**LEGEND_KW)

    im = ax_R.imshow(R_mat, vmin=0, vmax=1, cmap='viridis', origin='upper')
    ax_R.set_title(r'Pairwise phase-lock $R$ ($1$=locked, $0$=independent)')
    ax_R.set_xticks(range(n_pairs))
    ax_R.set_yticks(range(n_pairs))
    ax_R.set_xticklabels([str(i) for i in range(n_pairs)])
    ax_R.set_yticklabels([str(i) for i in range(n_pairs)])
    ax_R.set_xlabel('mode index $j$', fontsize=12)
    ax_R.set_ylabel('mode index $i$', fontsize=12)
    for i in range(n_pairs):
        for j in range(n_pairs):
            ax_R.text(j, i, f'{R_mat[i, j]:.2f}', ha='center', va='center',
                      color='black' if R_mat[i, j] > 0.5 else 'white', fontsize=9)
    fig.colorbar(im, ax=ax_R, fraction=0.046, pad=0.02)

    # Print classification table (data already computed earlier)
    print('\nHARMONIC CLASSIFICATION (master = mode 0; tested k=2..8):')
    print('  Harmonic if  ratio dev < 15%.  Slave if R high; Independent (DSI) if R low.')
    print('  mode  k   ratio  dev      R      amp_corr   classification')
    for c in classifications:
        ratio = pairs[c['mode']]['mu'].imag / om0_im
        print(f"  [{c['mode']}]  {c['k']}  {ratio:5.2f}  {c['dev']*100:4.1f}%  "
              f"{c['R']:5.3f}  {c['amp_corr']:+5.3f}    {c['verdict']}")
    print()

    # Polar: master (mode 0) vs FIRST REAL HARMONIC = lowest k from classifications
    # (auto-adapts to ssm_dim). Optional override via --harmonic_k.
    om0 = pairs[0]['mu'].imag
    phi_0 = np.unwrap(np.angle(Z[pairs[0]['idx'], :]))
    harm_idx = None
    harm_k = None
    real_harm = sorted([c for c in classifications if c['is_harm']],
                       key=lambda c: c['k'])
    if real_harm:
        if args.harmonic_k is not None:
            override = [c for c in real_harm if c['k'] == int(args.harmonic_k)]
            if override:
                harm_idx = override[0]['mode']
                harm_k = override[0]['k']
        if harm_idx is None:
            harm_idx = real_harm[0]['mode']
            harm_k = real_harm[0]['k']
    elif n_pairs >= 2:
        # Fallback: no strict harmonic, but use mode 1 with closest integer k
        ratio = pairs[1]['mu'].imag / om0
        harm_idx = 1
        harm_k = max(2, round(ratio))
    if harm_idx is not None:
        phi_h = np.unwrap(np.angle(Z[pairs[harm_idx]['idx'], :]))
        psi = np.mod(harm_k * phi_0 - phi_h, 2.0 * np.pi)
        R_h = float(np.abs(np.mean(np.exp(1j * psi))))
        actual_ratio = pairs[harm_idx]['mu'].imag / om0
        ax_pol.hist(psi, bins=36, color='#FFB04A', alpha=0.75)
        ax_pol.set_title(
            rf'$\psi = {harm_k}\,\phi_0 - \phi_{harm_idx}$  (Mode 0 vs {harm_idx})',
            fontsize=11)
        print(f'\nPOLAR PLOT pair: master (mode 0) vs mode {harm_idx} as {harm_k}. harmonic')
        print(f'  omega_{harm_idx}/omega_0 = {actual_ratio:.3f}  '
              f'(k={harm_k}, dev {abs(actual_ratio-harm_k)/harm_k*100:.1f}%)')
        print(f'  R = {R_h:.4f}')
    else:
        ax_pol.text(0, 0, 'no integer harmonic\ndetected',
                    ha='center', va='center', color='#E0E0E0', fontsize=10)
        ax_pol.set_title('Master vs ?', fontsize=10)

    # Theoretical harmonic table (matplotlib Table) — bottom-left subplot
    lam0_obs = 10.0 ** pairs[0]['T']
    ks = list(range(1, 9))
    obs_lam_list = [10.0 ** p['T'] for p in pairs]
    obs_k_list = []
    for ip, p in enumerate(pairs):
        if ip == 0:
            obs_k_list.append(1)
        else:
            obs_k_list.append(round(p['mu'].imag / pairs[0]['mu'].imag))
    cell_text = [['k', r'theor. $\lambda$', r'obs. $\lambda$', 'mode']]
    for k in ks:
        match_idx = None
        for ip, k_obs in enumerate(obs_k_list):
            if k_obs == k:
                match_idx = ip
                break
        if match_idx is not None:
            cell_text.append([str(k), f'{lam0_obs**(1.0/k):.3f}',
                              f'{obs_lam_list[match_idx]:.3f}',
                              f'{match_idx}'])
        else:
            cell_text.append([str(k), f'{lam0_obs**(1.0/k):.3f}', '–', '–'])
    ax_tbl.axis('off')
    tbl = ax_tbl.table(cellText=cell_text, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#808080')
        cell.set_facecolor('#1A1A1A')
        cell.get_text().set_color('#E0E0E0')
        if r == 0:
            cell.set_facecolor('#2A2A2A')
            cell.get_text().set_fontweight('bold')
        elif r > 0 and cell_text[r][2] != '–':
            cell.set_facecolor('#2D4A2D')
    ax_tbl.set_title(
        rf'Theoretical vs observed  ($\lambda_0 = {lam0_obs:.3f}$)',
        fontsize=11, color='#E0E0E0', pad=4)

    with plt.rc_context({'text.usetex': False}):
        plt.suptitle(f'BTC residuals: phase-lock of all SSM B (log10-clock) modes  '
                     f'[ssm_dim={args.ssm_dim}, poly={args.poly_degree}]',
                     color='#CCCCCC', fontsize=13, y=0.99,
                     fontname='Comfortaa', fontweight='bold')

    plt.subplots_adjust(top=0.94, bottom=0.04, hspace=0.50, wspace=0.32)

    if args.save is None:
        args.save = os.path.join(
            HERE,
            f'03b_phase_all_SSM_B_dim{args.ssm_dim}_poly{args.poly_degree}.png',
        )
    fig.savefig(args.save, dpi=200, facecolor='#0a0a0a')
    print(f'plot saved to {args.save}')
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
