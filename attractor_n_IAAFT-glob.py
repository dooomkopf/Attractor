#!/usr/bin/env python3
"""
attractor_n_IAAFT-glob.py — IAAFT-Surrogate auf Ensemble-n(t), kompletter Datensatz.
Fenster 1: Original  n(t) + 3D Attraktor
Fenster 2: Surrogat  n(t) + 3D Attraktor
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── PARAMETER ─────────────────────────────────────────────────────────────────
START_IDX    = 1164
TAU          = 30
M            = 50
SMOOTH_SIGMA = 60
N_ITER       = 50
WINDOW_SIZES = list(range(90, 181, 10))   # [90,100,...,180]
CMAP         = plt.cm.coolwarm
# ──────────────────────────────────────────────────────────────────────────────

try:
    plt.style.use('hz.mplstyle')
except Exception:
    plt.rcParams.update({
        'figure.facecolor': 'black', 'axes.facecolor': '#1A1A1A',
        'axes.edgecolor': '#CCCCCC', 'axes.labelcolor': '#CCCCCC',
        'text.color': '#CCCCCC', 'xtick.color': '#CCCCCC',
        'ytick.color': '#CCCCCC', 'grid.color': '#666666',
        'legend.facecolor': '#1A1A1A', 'savefig.facecolor': 'black',
        'font.size': 11,
    })


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


def make_segments(x, y):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def make_segments3d(x, y, z):
    pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def huber_location(x, k=1.345, tol=1e-6, max_iter=100):
    """Huber M-Schätzer für Lageparameter."""
    x     = np.asarray(x, dtype=float)
    mad   = np.median(np.abs(x - np.median(x)))
    scale = mad / 0.6745 if mad > 0 else (np.std(x) if np.std(x) > 0 else 1.0)
    mu    = np.median(x)
    for _ in range(max_iter):
        r       = (x - mu) / scale
        weights = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
        mu_new  = np.sum(weights * x) / np.sum(weights)
        if np.abs(mu_new - mu) < tol:
            mu = mu_new
            break
        mu = mu_new
    return float(mu)


def draw_pdf_fig(sig, surr):
    """Fenster 3: PDF Original vs Surrogat — Laplace-Fit (Huber), kein Clipping."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('black')
    fig.suptitle('PDF Ensemble n(t)  —  Original vs IAAFT-Surrogat',
                 color='#CCCCCC', fontsize=12)

    for ax, data, color, title in [
        (ax_l, sig,  '#90EE90', 'Original'),
        (ax_r, surr, '#FF69B4', 'IAAFT-Surrogat'),
    ]:
        ax.set_facecolor('#1A1A1A')
        ax.tick_params(colors='#CCCCCC')
        ax.set_xlabel('n(t)', color='#E0E0E0')
        ax.set_ylabel('PDF', color='#E0E0E0')
        ax.set_title(title, color='#E0E0E0', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Histogram — voller Datenbereich, kein Clipping
        hist, edges = np.histogram(data, bins=80, density=True)
        centers     = (edges[:-1] + edges[1:]) / 2
        ax.scatter(centers[hist > 0], hist[hist > 0],
                   s=20, alpha=0.7, c=color, edgecolors='none')

        # Laplace-Fit mit Huber
        mu_h = huber_location(data)
        b_h  = np.mean(np.abs(data - mu_h))
        x_fit = np.linspace(edges[0], edges[-1], 500)
        ax.plot(x_fit, (1.0 / (2.0 * b_h)) * np.exp(-np.abs(x_fit - mu_h) / b_h),
                color=color, linewidth=2.0, alpha=0.9,
                label=rf'Lap: $\mu_H$={mu_h:.3f}, $b$={b_h:.3f}')

        ax.set_yscale('log')
        ax.legend(loc='upper right', fontsize=9, facecolor='#1A1A1A',
                  edgecolor='#808080', labelcolor='#E0E0E0')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def iaaft_surrogate(x, rng, n_iter=N_ITER, tol=1e-8):
    n        = len(x)
    X_orig   = np.fft.rfft(x)
    amp      = np.abs(X_orig)
    x_sorted = np.sort(x)
    s        = rng.permutation(x)
    prev_err = np.inf
    for _ in range(n_iter):
        S     = np.fft.rfft(s)
        S_new = amp * np.exp(1j * np.angle(S))
        S_new[0] = X_orig[0]
        if n % 2 == 0:
            S_new[-1] = X_orig[-1]
        s_tmp = np.fft.irfft(S_new, n=n)
        ranks = np.argsort(np.argsort(s_tmp))
        s     = x_sorted[ranks]
        err   = np.sum((np.abs(np.fft.rfft(s)) - amp)**2) / (np.sum(amp**2) + 1e-30)
        if err < tol or abs(prev_err - err) < tol * 1e-3:
            break
        prev_err = err
    return s


def embed_pca(sig, M, TAU, SMOOTH_SIGMA):
    N = len(sig)
    W = (M - 1) * TAU
    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = sig[W - j * TAU : N - j * TAU]
    D_c      = D - D.mean(axis=0)
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc       = D_c @ Vt.T
    var      = s**2 / (s**2).sum()
    pc_s     = pc.copy()
    for j in range(pc_s.shape[1]):
        pc_s[:, j] = gaussian_filter1d(pc[:, j], sigma=SMOOTH_SIGMA)
    return pc, pc_s, var, W


def compute_phase(pc):
    """theta (wrapped), r, theta_uw aus ungeglättetem pc."""
    theta    = np.arctan2(pc[:, 1], pc[:, 0])
    r        = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
    theta_uw = np.unwrap(theta)
    if theta_uw[-1] < theta_uw[0]:
        theta    = -theta
        theta_uw = -theta_uw
    return theta, r, theta_uw


def draw_polar_fig(pc_orig, pc_surr, t_norm, var_orig, var_surr):
    """Fenster 4: 2×2 — Polar + Phase für Original (oben) und Surrogat (unten)."""
    theta_o, r_o, _ = compute_phase(pc_orig)
    theta_s, r_s, _ = compute_phase(pc_surr)

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('black')
    fig.suptitle('Phasenraum  —  Original vs IAAFT-Surrogat',
                 color='#CCCCCC', fontsize=12)

    for row, (theta, r, pc, var, color_title) in enumerate([
        (theta_o, r_o, pc_orig, var_orig, ('Original',  '#90EE90')),
        (theta_s, r_s, pc_surr, var_surr, ('IAAFT-Surrogat', '#FF69B4')),
    ]):
        label, col = color_title

        # ── Polar Plot ────────────────────────────────────────────────────────
        ax_pol = fig.add_subplot(2, 2, row * 2 + 1, projection='polar')
        ax_pol.set_facecolor('#1A1A1A')
        segs_pol = make_segments(theta % (2 * np.pi), r)
        lc_pol   = LineCollection(segs_pol, cmap=CMAP, linewidth=0.8, alpha=0.85)
        lc_pol.set_array(t_norm)
        ax_pol.add_collection(lc_pol)
        ax_pol.set_ylim(0, r.max() * 1.1)
        ax_pol.set_yticklabels([])
        ax_pol.set_rticks([])
        ax_pol.tick_params(colors='#CCCCCC', labelsize=8)
        ax_pol.set_title(f'{label} — θ vs r', color='#E0E0E0', fontsize=10, pad=10)

        # ── Phasen-Plot (θ vs PC3) ────────────────────────────────────────────
        ax_ph = fig.add_subplot(2, 2, row * 2 + 2)
        ax_ph.set_facecolor('#1A1A1A')
        ax_ph.tick_params(colors='#CCCCCC')
        sc = ax_ph.scatter(theta % (2 * np.pi), pc[:, 2],
                           c=t_norm, cmap=CMAP, s=2, alpha=0.7)
        ax_ph.set_xlim(0, 2 * np.pi)
        ax_ph.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax_ph.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'], color='#CCCCCC')
        ax_ph.set_xlabel('θ', color='#E0E0E0')
        ax_ph.set_ylabel(f'PC3 ({var[2]*100:.1f}%)', color='#E0E0E0')
        ax_ph.set_title(f'{label} — θ vs PC3', color='#E0E0E0', fontsize=10)
        ax_ph.grid(True, alpha=0.3, linestyle='--')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def draw_fig(sig, days_emb, pc_s, var, W, t_norm, label):
    fig = plt.figure(figsize=(20, 9))
    fig.patch.set_facecolor('black')
    fig.suptitle(f'{label}  |  M={M} τ={TAU}d  W={(M-1)*TAU}d  '
                 f'kum3={np.cumsum(var)[2]*100:.1f}%',
                 color='#CCCCCC', fontsize=12)

    ax1 = fig.add_axes([0.04, 0.12, 0.42, 0.80])
    ax2 = fig.add_axes([0.50, 0.04, 0.50, 0.92], projection='3d')

    # ── n(t) Signal ──────────────────────────────────────────────────────────
    ax1.set_facecolor('#1A1A1A')
    t_norm_sig = (days_emb - days_emb[0]) / (days_emb[-1] - days_emb[0])
    segs = make_segments(days_emb, sig)
    lc   = LineCollection(segs, cmap=CMAP, linewidth=0.9, alpha=0.9)
    lc.set_array(t_norm_sig)
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.axhline(0, color='#808080', linestyle='--', linewidth=1.0, alpha=0.5)
    ax1.set_xlabel('Day index', color='#E0E0E0')
    ax1.set_ylabel('n(t)  ensemble', color='#E0E0E0')
    ax1.set_title('Ensemble n(t)', color='#E0E0E0', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ── 3D Attraktor ─────────────────────────────────────────────────────────
    segs3d = make_segments3d(pc_s[:, 0], pc_s[:, 1], pc_s[:, 2])
    lc3d   = Line3DCollection(segs3d, cmap=CMAP, linewidth=1.8, alpha=0.9)
    lc3d.set_array(t_norm)
    ax2.add_collection3d(lc3d)
    for arr, setter in [(pc_s[:, 0], ax2.set_xlim),
                        (pc_s[:, 1], ax2.set_ylim),
                        (pc_s[:, 2], ax2.set_zlim)]:
        pad = (arr.max() - arr.min()) * 0.05
        setter(arr.min() - pad, arr.max() + pad)
    for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#444444')
    ax2.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_zlabel(f'PC3 ({var[2]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_title('3D Attraktor', color='#E0E0E0', fontsize=10)
    ax2.tick_params(colors='#CCCCCC')

    return fig


def main():
    days_all, prices_all, _ = read_btc_data('ziel.csv')

    # ── Ensemble n(t) ─────────────────────────────────────────────────────────
    print(f"Berechne Ensemble n(t)  windows={WINDOW_SIZES[0]}..{WINDOW_SIZES[-1]}d ...")
    n_matrix = np.full((len(WINDOW_SIZES), len(days_all)), np.nan)
    for wi, ws in enumerate(WINDOW_SIZES):
        half = ws // 2
        for i in range(half, len(days_all) - half):
            t1, t2 = float(days_all[i - half]), float(days_all[i + half])
            p1, p2 = prices_all[i - half], prices_all[i + half]
            if p1 > 0 and p2 > 0 and t2 > t1:
                denom = np.log(t2 / t1)
                if denom != 0:
                    n_matrix[wi, i] = np.log(p2 / p1) / denom

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        daily_n_all = np.nanmean(n_matrix, axis=0)

    half_max = max(WINDOW_SIZES) // 2
    daily_n_all[:half_max]  = np.nan
    daily_n_all[-half_max:] = np.nan

    # ── Embedding-Signal ──────────────────────────────────────────────────────
    mask_emb = (days_all >= START_IDX - half_max) & np.isfinite(daily_n_all)
    sig      = daily_n_all[mask_emb]
    days_emb = days_all[mask_emb]

    # ── Original PCA ──────────────────────────────────────────────────────────
    print("Berechne Original PCA...")
    pc_orig, pc_s_orig, var_orig, W = embed_pca(sig, M, TAU, SMOOTH_SIGMA)
    days_vecs = days_emb[W:]
    t_norm    = (days_vecs - days_vecs[0]) / (days_vecs[-1] - days_vecs[0])

    print(f"Embedding: {len(days_vecs)} Vektoren  M={M} τ={TAU}d W={W}d")
    print(f"PC1={var_orig[0]*100:.1f}%  PC2={var_orig[1]*100:.1f}%  "
          f"PC3={var_orig[2]*100:.1f}%  kum3={np.cumsum(var_orig)[2]*100:.1f}%")

    # ── IAAFT ─────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument('--surrogate', type=int, default=0,
                        help='Surrogate-Index (= RNG-Seed), default 0')
    args = parser.parse_args()

    rng = np.random.default_rng(args.surrogate)
    print(f"Berechne IAAFT-Surrogat #{args.surrogate} (N_ITER={N_ITER})...")
    surr = iaaft_surrogate(sig, rng)
    pc_surr, pc_s_surr, var_surr, _ = embed_pca(surr, M, TAU, SMOOTH_SIGMA)
    print("Fertig.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    surr_label = f'IAAFT-Surrogat #{args.surrogate}'
    draw_fig(sig,  days_emb, pc_s_orig, var_orig, W, t_norm, 'Original — Ensemble n(t)')
    draw_fig(surr, days_emb, pc_s_surr, var_surr, W, t_norm, f'{surr_label} — Ensemble n(t)')
    draw_pdf_fig(sig, surr)
    draw_polar_fig(pc_orig, pc_surr, t_norm, var_orig, var_surr)

    plt.show()


if __name__ == '__main__':
    main()
