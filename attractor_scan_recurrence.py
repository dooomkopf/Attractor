#!/usr/bin/env python3
"""
attractor_scan_recurrence.py — Optimale Embedding-Parameter via Recurrence-Analyse.

Scan τ = 1...τ_max (max = (N-1)/(M-1)), festes M.
Für jedes τ: Delay-Embedding → Recurrence-Matrix (sparse, KDTree) → %DET.
Maximum von %DET → optimales τ.

Interaktiver ε-Slider für Preview.
"""

import argparse
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

plt.style.use('hz.mplstyle')
mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']

PERCENTILE = 0.01
START_IDX  = 1164
HALVINGS   = [datetime(2012, 11, 28), datetime(2016, 7, 9),
              datetime(2020, 5, 11),  datetime(2024, 4, 20)]
_HERE = os.path.dirname(os.path.abspath(__file__))


def read_btc_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                data.append((int(parts[0]), float(parts[1]),
                             datetime.strptime(parts[2], "%d.%m.%Y")))
    days   = np.array([d[0] for d in data])
    prices = np.array([d[1] for d in data])
    dates  = np.array([d[2] for d in data])
    return days, prices, dates


def build_embedding(x, M, tau, step=1):
    """Delay-Embedding: D[i,j] = x[W - j*tau + i], shape (N-W, M).
    step>1: nur jeden step-ten Vektor (Undersampling).
    """
    W = (M - 1) * tau
    N = len(x)
    if W >= N:
        return None
    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = x[W - j * tau : N - j * tau]
    if step > 1:
        D = D[::step]
    return D


def recurrence_stats(D, eps, l_min=2):
    """%REC und %DET aus Delay-Embedding via sparse KDTree.

    %REC = Anteil recurrenter Paare
    %DET = Anteil recurrenter Punkte die auf Diagonalen der Länge >= l_min liegen
    """
    N = len(D)
    tree = cKDTree(D)
    pairs = tree.query_pairs(eps, output_type='ndarray')
    n_pairs = len(pairs)
    total_pairs = N * (N - 1) // 2
    rec = n_pairs / total_pairs if total_pairs > 0 else 0.0

    if n_pairs == 0:
        return rec, 0.0

    # %DET: zähle Punkte auf Diagonalen (i,j) → (i+1,j+1) → ...
    # Nur Kette starten wenn (i-1,j-1) NICHT recurrent (= Diagonalanfang)
    pair_set = set(map(tuple, pairs))
    on_diag = 0

    for p_idx in range(n_pairs):
        i, j = int(pairs[p_idx, 0]), int(pairs[p_idx, 1])
        # Kein Diagonalanfang wenn Vorgänger existiert
        pred = (min(i - 1, j - 1), max(i - 1, j - 1))
        if i > 0 and j > 0 and pred in pair_set:
            continue
        # Diagonale vorwärts verfolgen
        length = 1
        ci, cj = i + 1, j + 1
        while ci < N and cj < N and (min(ci, cj), max(ci, cj)) in pair_set:
            length += 1
            ci += 1
            cj += 1
        if length >= l_min:
            on_diag += length

    det = on_diag / n_pairs if n_pairs > 0 else 0.0
    return float(rec), float(det)


def find_eps_for_target_rec(D, target_rec=0.05, tol=0.005, max_iter=15):
    """Binäre Suche nach ε das target %REC ± tol ergibt."""
    N = len(D)
    tree = cKDTree(D)
    total = N * (N - 1) // 2

    # Grob: Median-Distanz als Startwert
    sample_idx = np.random.default_rng(0).choice(N, size=min(500, N), replace=False)
    sample_dists = []
    for i in sample_idx[:50]:
        d, _ = tree.query(D[i], k=min(20, N))
        sample_dists.extend(d[1:])
    eps_hi = np.percentile(sample_dists, 95)
    eps_lo = np.percentile(sample_dists, 1)

    for _ in range(max_iter):
        eps = (eps_lo + eps_hi) / 2
        n_pairs = len(tree.query_pairs(eps))
        rec = n_pairs / total
        if abs(rec - target_rec) < tol:
            return eps
        if rec < target_rec:
            eps_lo = eps
        else:
            eps_hi = eps
    return eps


def main():
    ap = argparse.ArgumentParser(description='Recurrence-basierte τ-Optimierung')
    ap.add_argument('--M',          type=int,   default=35)
    ap.add_argument('--target_rec', type=float, default=0.05)
    ap.add_argument('--l_min',      type=int,   default=2)
    ap.add_argument('--step',       type=int,   default=3,
                    help='Undersampling: jeden n-ten Vektor (3 = alle 3 Tage)')
    args = ap.parse_args()

    M = args.M
    days_all, prices_all, dates_all = read_btc_data(os.path.join(_HERE, 'ziel.csv'))

    X  = sm.add_constant(np.log(days_all))
    qr = QuantReg(np.log(prices_all), X).fit(q=PERCENTILE)
    log_res_all = np.log(prices_all / np.exp(qr.predict(X)))

    mask_emb = days_all >= START_IDX
    log_res  = log_res_all[mask_emb]
    N        = len(log_res)
    tau_max  = (N - 1) // (M - 1)

    print(f"N={N}  M={M}  τ_max={tau_max}  target_REC={args.target_rec}")
    print()

    # ── Scan ──────────────────────────────────────────────────────────────────
    taus = np.arange(1, tau_max + 1)
    rec_vals = np.empty(len(taus))
    det_vals = np.empty(len(taus))
    eps_vals = np.empty(len(taus))

    for idx, tau in enumerate(taus):
        D = build_embedding(log_res, M, tau, step=args.step)
        if D is None or len(D) < 50:
            rec_vals[idx] = det_vals[idx] = eps_vals[idx] = np.nan
            continue

        eps = find_eps_for_target_rec(D, target_rec=args.target_rec)
        rec, det = recurrence_stats(D, eps, l_min=args.l_min)
        rec_vals[idx] = rec
        det_vals[idx] = det
        eps_vals[idx] = eps

        pct = 100 * (idx + 1) / len(taus)
        print(f"\r  [{pct:5.1f}%]  τ={tau:3d}/{taus[-1]}  ε={eps:.4f}  %REC={rec:.3f}  %DET={det:.3f}    ",
              end='', flush=True)
    print()

    # ── Optimum ───────────────────────────────────────────────────────────────
    valid = np.isfinite(det_vals)
    if valid.any():
        tau_opt = int(taus[valid][np.argmax(det_vals[valid])])
        det_max = np.max(det_vals[valid])
    else:
        tau_opt = None
        det_max = 0

    print(f"\nErgebnis:")
    if tau_opt:
        print(f"  τ_opt (max %DET) = {tau_opt} Tage  (%DET={det_max:.3f})")
    else:
        print(f"  Kein Optimum gefunden")
    print(f"  Aktuell: τ=40, M=35")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # %DET vs τ
    ax1.plot(taus[valid], det_vals[valid], color='#4488FF', linewidth=1.5,
             label=r'\%DET')
    if tau_opt:
        ax1.axvline(tau_opt, color='#FF4444', linewidth=2,
                    label=r'$\tau_{{opt}}$ = ' + f'{tau_opt}d')
    ax1.axvline(40, color='#888888', linewidth=1, linestyle='--',
                label=r'current $\tau$ = 40d')
    ax1.set_ylabel(r'\%DET')
    ax1.set_title(r'\%DET vs $\tau$ (Determinism)')
    ax1.legend(fontsize=9, facecolor='#1A1A1A', edgecolor='#808080',
               labelcolor='#E0E0E0')

    # %REC vs τ
    ax2.plot(taus[valid], rec_vals[valid], color='#44FF44', linewidth=1.5)
    ax2.axhline(args.target_rec, color='#888888', linewidth=1, linestyle=':',
                label=f'target = {args.target_rec}')
    ax2.axvline(40, color='#888888', linewidth=1, linestyle='--')
    ax2.set_ylabel(r'\%REC')
    ax2.set_title(r'\%REC vs $\tau$ (Recurrence Rate)')
    ax2.legend(fontsize=9, facecolor='#1A1A1A', edgecolor='#808080',
               labelcolor='#E0E0E0')

    # ε vs τ
    ax3.plot(taus[valid], eps_vals[valid], color='#FFAA44', linewidth=1.5)
    ax3.axvline(40, color='#888888', linewidth=1, linestyle='--',
                label=r'current $\tau$ = 40d')
    ax3.set_xlabel(r'$\tau$ [days]')
    ax3.set_ylabel(r'$\varepsilon$')
    ax3.set_title(r'Adaptive $\varepsilon(\tau)$ for target \%REC')
    ax3.legend(fontsize=9, facecolor='#1A1A1A', edgecolor='#808080',
               labelcolor='#E0E0E0')

    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(f'Recurrence Scan  [M={M}, target REC={args.target_rec}]',
                     color='#CCCCCC', fontsize=13, y=0.98,
                     fontname='Comfortaa', fontweight='bold')
    plt.subplots_adjust(top=0.92, hspace=0.40)
    _out = os.path.join(_HERE, 'recurrence_scan.png')
    plt.savefig(_out, dpi=150, facecolor='#0a0a0a')
    print(f"\nGespeichert: {_out}")
    plt.show()


if __name__ == '__main__':
    main()
