#!/usr/bin/env python3
"""
Attraktor-Analyse – optimierte Parameter aus Halving-Scan:
W=1215d (3.33y)  TAU=135d  M=10
Gleichverteilungs-Optimum: Halvings 2016/2020/2024 bei σ=0.1°
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.widgets import Slider
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── PARAMETER ─────────────────────────────────────────────────────────────────
PERCENTILE        = 0.01
START_IDX         = 1164
TAU               = 135   # W=1215d (3.33y), M=10 → TAU=135d
M                 = 10
NORMALIZE_WINDOWS = False
SMOOTH_SIGMA      = 60         # Gaussian-Glättung der PC-Trajektorie
LABEL_WINDOW      = 30         # ±Tage um Slider-Position hervorheben
SHOW_FIG4         = True       # θ unwrapped (lineare vs. log Periodizität)
CMAP              = plt.cm.coolwarm
HALVINGS          = [datetime(2012, 11, 28), datetime(2016, 7, 9),
                     datetime(2020, 5, 11),  datetime(2024, 4, 20)]
# Harte Realraum-Tops aus /home/hz/Data/gold/cycles.json.
# Residuen-Peaks treffen die historischen Tops nicht immer.
CYCLE_TOPS        = [
    (datetime(2013, 12, 4), "'13-cycle"),
    (datetime(2017, 12, 16), "'17-cycle"),
    (datetime(2021, 11, 9), "'21-cycle"),
    (datetime(2025, 10, 7), "'25-cycle"),
]
# ──────────────────────────────────────────────────────────────────────────────

try:
    plt.style.use('hz.mplstyle')
except Exception:
    plt.rcParams.update({
        'figure.facecolor':  'black',
        'axes.facecolor':    '#1A1A1A',
        'axes.edgecolor':    '#CCCCCC',
        'axes.labelcolor':   '#CCCCCC',
        'text.color':        '#CCCCCC',
        'xtick.color':       '#CCCCCC',
        'ytick.color':       '#CCCCCC',
        'grid.color':        '#666666',
        'legend.facecolor':  '#1A1A1A',
        'legend.edgecolor':  '#CCCCCC',
        'savefig.facecolor': 'black',
        'font.size':         11,
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


def main():
    # ── Daten laden ───────────────────────────────────────────────────────────
    days_all, prices_all, dates_all = read_btc_data('ziel.csv')

    # ── Quantilregression ─────────────────────────────────────────────────────
    log_days_all  = np.log(days_all)
    log_btc_all   = np.log(prices_all)
    X_all         = sm.add_constant(log_days_all)
    qr            = QuantReg(log_btc_all, X_all).fit(q=PERCENTILE)
    log_fit_all   = qr.predict(X_all)
    residuals_all = prices_all / np.exp(log_fit_all)
    log_res_all   = np.log(residuals_all)

    # ── Halving-Tagesindizes ──────────────────────────────────────────────────
    halving_days = []
    for hd in HALVINGS:
        diffs = np.array([abs((d - hd).days) for d in dates_all])
        halving_days.append(days_all[np.argmin(diffs)])
    halving_days = np.array(halving_days)

    # ── Zeitnormierung ────────────────────────────────────────────────────────
    d_min, d_max = days_all[0], days_all[-1]
    t_norm_all   = (days_all - d_min) / (d_max - d_min)

    # ── Historische Cycle-Tops (hard-coded) ──────────────────────────────────
    used_peak_days, used_peak_vals, peak_year_labels = [], [], []
    for dt, lbl in CYCLE_TOPS:
        diffs = np.array([abs((d - dt).days) for d in dates_all])
        idx_ = np.argmin(diffs)
        used_peak_days.append(days_all[idx_])
        used_peak_vals.append(residuals_all[idx_])
        peak_year_labels.append(lbl)
    used_peak_days = np.array(used_peak_days)
    used_peak_vals = np.array(used_peak_vals)

    yr_labels = [dates_all[np.argmin(np.abs(days_all - y))].strftime('%Y')
                 for y in np.linspace(d_min, d_max, 6)]

    # ── Delay Embedding ───────────────────────────────────────────────────────
    mask_emb = days_all >= START_IDX
    log_res  = log_res_all[mask_emb]
    days_emb = days_all[mask_emb]
    N        = len(log_res)
    W        = (M - 1) * TAU

    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = log_res[W - j * TAU : N - j * TAU]
    if NORMALIZE_WINDOWS:
        D -= D.mean(axis=1, keepdims=True)

    D_c      = D - D.mean(axis=0)
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc       = D_c @ Vt.T
    var      = s**2 / (s**2).sum()

    # Geglättete Trajektorie (nur für Visualisierung)
    pc_s = pc.copy()
    for j in range(pc_s.shape[1]):
        pc_s[:, j] = gaussian_filter1d(pc[:, j], sigma=SMOOTH_SIGMA)

    days_vecs  = days_emb[W:]
    t_norm_vec = (days_vecs - d_min) / (d_max - d_min)

    # Phase (raw, nicht geglättet)
    theta    = np.arctan2(pc[:, 1], pc[:, 0])
    r        = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
    theta_uw = np.unwrap(theta)

    cum3 = np.cumsum(var)[2] * 100

    # ══════════════════════════════════════════════════════════════════════════
    # Hauptfenster: Residuen (oben) + 3D Attraktor (unten) + Slider
    # ══════════════════════════════════════════════════════════════════════════
    fig_main = plt.figure(figsize=(22, 10))
    fig_main.patch.set_facecolor('black')

    ax1 = fig_main.add_axes([0.04, 0.30, 0.42, 0.43])   # Residuen links, vertikal zentriert
    ax2 = fig_main.add_axes([0.50, 0.06, 0.50, 0.90],   # 3D rechts, groß
                             projection='3d')
    ax_sl = fig_main.add_axes([0.10, 0.02, 0.80, 0.025])  # Slider unten
    ax_sl.set_facecolor('#2A2A2A')

    # ── ax1: Residuen ─────────────────────────────────────────────────────────
    ax1.set_facecolor('#1A1A1A')
    segs = make_segments(days_all, residuals_all)
    lc1  = LineCollection(segs, cmap=CMAP, linewidth=1.0, alpha=0.9)
    lc1.set_array(t_norm_all)
    ax1.add_collection(lc1)
    ax1.set_yscale('log')
    ax1.set_xlim(0, d_max * 1.05)
    ax1.set_ylim(residuals_all.min() * 0.9, residuals_all.max() * 1.1)
    ax1.axhline(y=1, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)

    if len(used_peak_days):
        ax1.scatter(used_peak_days, used_peak_vals, color='red', s=50,
                    zorder=5, alpha=0.8)
        for px, py, lbl in zip(used_peak_days, used_peak_vals, peak_year_labels):
            ax1.annotate(lbl, (px, py), textcoords='offset points',
                         xytext=(6, 6), color='white', fontsize=9, fontweight='bold')
    ax1.scatter(days_all[-1], residuals_all[-1], color='white', s=40,
                zorder=10, edgecolors='black', linewidth=1.0, label='Latest')
    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_all - hday))
        ax1.scatter(days_all[hidx], residuals_all[hidx], color='lime', s=60,
                    zorder=10, label='Halving' if i == 0 else '_')
        ax1.annotate(str(i + 1), (days_all[hidx], residuals_all[hidx]),
                     textcoords='offset points', xytext=(5, 5),
                     color='lime', fontsize=9, fontweight='bold')

    cb1 = fig_main.colorbar(lc1, ax=ax1, pad=0.01)
    cb1.set_label('Jahr', color='#CCCCCC')
    cb1.set_ticks(np.linspace(0, 1, 6))
    cb1.set_ticklabels(yr_labels)

    # ── Zykluslängen in ax1 ───────────────────────────────────────────────────
    cross_idx = []
    for k in range(1, 30):
        level = k * 2 * np.pi
        idx = np.where((theta_uw[:-1] < level) & (theta_uw[1:] >= level))[0]
        if len(idx) > 0:
            cross_idx.append(idx[0])
    bounds = [0] + cross_idx + [len(theta_uw) - 1]
    if len(cross_idx) > 0:
        ax1.axvline(days_vecs[cross_idx[0]], color='#FFDD44',
                    linewidth=0.8, alpha=0.6, linestyle=':')
    for ci in range(1, len(bounds) - 2):  # nur vollständige Zyklen (Mitte)
        i1, i2 = bounds[ci], bounds[ci + 1]
        dur_yr = (days_vecs[i2] - days_vecs[i1]) / 365.25
        d_cross = days_vecs[i2]
        ax1.axvline(d_cross, color='#FFDD44', linewidth=0.8, alpha=0.6, linestyle=':')
        mid_day = (days_vecs[i1] + days_vecs[i2]) / 2
        ax1.text(mid_day, residuals_all.max() * 0.75,
                 f'{dur_yr:.1f}y', color='#FFDD44', fontsize=8,
                 fontweight='bold', ha='center', va='top')

    ax1.set_ylabel('rel. Value', color='#E0E0E0')
    ax1.set_title('BTC USD Residuals', color='#E0E0E0', fontsize=11)
    ax1.grid(True, alpha=0.5, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=8)

    # ── ax2: PCA 3D ───────────────────────────────────────────────────────────
    segs3d = make_segments3d(pc_s[:, 0], pc_s[:, 1], pc_s[:, 2])
    lc2    = Line3DCollection(segs3d, cmap=CMAP, linewidth=2.5, alpha=0.9)
    lc2.set_array(t_norm_vec)
    ax2.add_collection3d(lc2)

    for arr, setter in [(pc_s[:, 0], ax2.set_xlim),
                        (pc_s[:, 1], ax2.set_ylim),
                        (pc_s[:, 2], ax2.set_zlim)]:
        pad = (arr.max() - arr.min()) * 0.05
        setter(arr.min() - pad, arr.max() + pad)

    for pd_, lbl in zip(used_peak_days, peak_year_labels):
        idx_ = np.argmin(np.abs(days_vecs - pd_))
        if np.abs(days_vecs[idx_] - pd_) < 200:
            ax2.scatter(pc_s[idx_, 0], pc_s[idx_, 1], pc_s[idx_, 2],
                        color='red', s=60, zorder=10)
            ax2.text(pc_s[idx_, 0], pc_s[idx_, 1], pc_s[idx_, 2], f' {lbl}',
                     color='white', fontsize=9, fontweight='bold')

    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_vecs - hday))
        if np.abs(days_vecs[hidx] - hday) < 200:
            ax2.scatter(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                        color='lime', s=60, zorder=10,
                        label='Halving' if i == 0 else '_')
            ax2.text(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2], f' {i+1}',
                     color='lime', fontsize=9, fontweight='bold')

    ax2.scatter(*[pc_s[-1, i] for i in range(3)],
                color='white', s=60, zorder=10, label='Now')


    for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#444444')

    ax2.set_xlabel(f'PC1  ({var[0]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_ylabel(f'PC2  ({var[1]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_zlabel(f'PC3  ({var[2]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_title(f'PCA 3D Attraktor  ·  τ={TAU}d · m={M} · W={W}d  (kum. {cum3:.1f}%)',
                  color='#E0E0E0', fontsize=10)
    ax2.tick_params(colors='#CCCCCC')
    ax2.legend(fontsize=8)


    # ── Slider ────────────────────────────────────────────────────────────────
    slider = Slider(ax_sl, 'Tag', float(days_vecs[0]), float(days_vecs[-1]),
                    valinit=float(days_vecs[len(days_vecs)//2]),
                    color='#666666')
    slider.label.set_color('#CCCCCC')
    slider.valtext.set_color('#CCCCCC')

    # Zustand der beweglichen Highlights
    hl = {'ax1_scat': None, 'ax2_scat': None}

    def update(val):
        d = slider.val
        # Alte Highlights entfernen
        for key in hl:
            if hl[key] is not None:
                try:
                    hl[key].remove()
                except Exception:
                    pass
                hl[key] = None

        # ax1: ±LABEL_WINDOW Tage — neon-gelber Balken
        mask1 = (days_all >= d - LABEL_WINDOW) & (days_all <= d + LABEL_WINDOW)
        if mask1.any():
            hl['ax1_scat'] = ax1.scatter(
                days_all[mask1], residuals_all[mask1],
                color='#FFFF00', s=80, alpha=1.0, zorder=9, linewidths=0)
        # ax2: ±LABEL_WINDOW Tage auf 3D-Trajektorie — neon-gelbe Punkte
        mask2 = (days_vecs >= d - LABEL_WINDOW) & (days_vecs <= d + LABEL_WINDOW)
        if mask2.any():
            hl['ax2_scat'] = ax2.scatter(
                pc_s[mask2, 0], pc_s[mask2, 1], pc_s[mask2, 2],
                color='#FFFF00', s=120, alpha=1.0, zorder=10)

        fig_main.canvas.draw_idle()

    slider.on_changed(update)

    # Initiales Highlight setzen
    update(slider.val)

    # ══════════════════════════════════════════════════════════════════════════
    # Fenster 2: Polarkoordinaten
    # ══════════════════════════════════════════════════════════════════════════
    fig3, _ = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'projection': None})
    fig3.patch.set_facecolor('black')
    ax3a = fig3.add_subplot(1, 2, 1, projection='polar')
    ax3b = fig3.axes[1]
    ax3b.set_facecolor('#1A1A1A')

    segs_pol = make_segments(theta, r)
    lc3a = LineCollection(segs_pol, cmap=CMAP, linewidth=0.7, alpha=0.85)
    lc3a.set_array(t_norm_vec)
    ax3a.add_collection(lc3a)
    ax3a.set_ylim(0, r.max() * 1.1)
    ax3a.scatter(theta[-1], r[-1], color='white', s=80, zorder=10, label='Jetzt')
    for pd_, lbl in zip(used_peak_days, peak_year_labels):
        idx_ = np.argmin(np.abs(days_vecs - pd_))
        if np.abs(days_vecs[idx_] - pd_) < 200:
            ax3a.scatter(theta[idx_], r[idx_], color='red', s=60, zorder=10)
            ax3a.annotate(lbl, (theta[idx_], r[idx_]), color='white',
                          fontsize=9, fontweight='bold')
    ax3a.set_yticklabels([])
    ax3a.set_title('Phase vs Intensity', color='#E0E0E0', fontsize=10, pad=15)
    ax3a.tick_params(colors='#CCCCCC')
    ax3a.set_facecolor('#1A1A1A')

    lc3b = ax3b.scatter(theta, pc[:, 2], c=t_norm_vec, cmap=CMAP,
                        s=2, alpha=0.7, linewidths=0)
    ax3b.scatter(theta[-1], pc[-1, 2], color='white', s=80, zorder=10, label='Jetzt')
    for pd_, lbl in zip(used_peak_days, peak_year_labels):
        idx_ = np.argmin(np.abs(days_vecs - pd_))
        if np.abs(days_vecs[idx_] - pd_) < 200:
            ax3b.scatter(theta[idx_], pc[idx_, 2], color='red', s=60, zorder=10)
            ax3b.annotate(lbl, (theta[idx_], pc[idx_, 2]),
                          textcoords='offset points', xytext=(6, 4),
                          color='white', fontsize=9, fontweight='bold')
    ax3b.set_xlabel('θ = atan2(PC2, PC1)  [Zyklusphase]', color='#E0E0E0')
    ax3b.set_ylabel(f'PC3  ({var[2]*100:.1f}%)', color='#E0E0E0')
    ax3b.set_title('Phase', color='#E0E0E0', fontsize=10)
    ax3b.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_vecs - hday))
        if np.abs(days_vecs[hidx] - hday) < 200:
            lbl_h = 'Halving' if i == 0 else '_'
            ax3a.scatter(theta[hidx], r[hidx], color='lime', s=60,
                         zorder=10, label=lbl_h)
            ax3a.annotate(str(i + 1), (theta[hidx], r[hidx]),
                          color='lime', fontsize=9, fontweight='bold')
            ax3b.scatter(theta[hidx], pc[hidx, 2], color='lime', s=60,
                         zorder=10, label=lbl_h)
            ax3b.annotate(str(i + 1), (theta[hidx], pc[hidx, 2]),
                          textcoords='offset points', xytext=(5, 5),
                          color='lime', fontsize=9, fontweight='bold')

    cb3 = fig3.colorbar(lc3b, ax=ax3b, pad=0.02)
    cb3.set_label('Jahr', color='#CCCCCC')
    cb3.set_ticks(np.linspace(0, 1, 6))
    cb3.set_ticklabels(yr_labels)
    ax3a.legend(loc='lower right', fontsize=8)
    ax3b.legend(fontsize=8)
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.88)

    # ══════════════════════════════════════════════════════════════════════════
    # Fenster 3: θ unwrapped – lineare vs. log(t) Periodizität
    # ══════════════════════════════════════════════════════════════════════════
    if not SHOW_FIG4:
        plt.show()
        print(f"Embedding: {N-W} Vektoren, M={M}, τ={TAU}d, W={W}d, ab idx {START_IDX}")
        print(f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  PC3={var[2]*100:.1f}%  "
              f"kum3={cum3:.1f}%")
        return

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.patch.set_facecolor('black')

    def _halving_on(ax, xvals):
        for i, (hday, xv) in enumerate(zip(halving_days, xvals)):
            hidx = np.argmin(np.abs(days_vecs - hday))
            if np.abs(days_vecs[hidx] - hday) < 200:
                ax.scatter(xv, theta_uw[hidx], color='lime', s=70, zorder=10,
                           label='Halving' if i == 0 else '_')
                ax.annotate(str(i + 1), (xv, theta_uw[hidx]),
                            textcoords='offset points', xytext=(5, 5),
                            color='lime', fontsize=9, fontweight='bold')

    ax4a.set_facecolor('#1A1A1A')
    segs4a = make_segments(days_vecs, theta_uw)
    lc4a   = LineCollection(segs4a, cmap=CMAP, linewidth=0.8, alpha=0.85)
    lc4a.set_array(t_norm_vec)
    ax4a.add_collection(lc4a)
    ax4a.autoscale()
    _halving_on(ax4a, halving_days)
    ax4a.set_xlabel('Day index  (linear)', color='#E0E0E0')
    ax4a.set_ylabel('θ unwrapped  [rad]', color='#E0E0E0')
    ax4a.set_title('Cycle phase vs t  (linear)\n→ straight = constant period',
                   color='#E0E0E0', fontsize=10)
    ax4a.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax4a.legend(fontsize=8)

    ax4b.set_facecolor('#1A1A1A')
    log_days_vecs = np.log(days_vecs)
    segs4b = make_segments(log_days_vecs, theta_uw)
    lc4b   = LineCollection(segs4b, cmap=CMAP, linewidth=0.8, alpha=0.85)
    lc4b.set_array(t_norm_vec)
    ax4b.add_collection(lc4b)
    ax4b.autoscale()
    _halving_on(ax4b, np.log(halving_days))
    ax4b.set_xlabel('log(Day index)', color='#E0E0E0')
    ax4b.set_ylabel('θ unwrapped  [rad]', color='#E0E0E0')
    ax4b.set_title('Cycle phase vs log(t)\n→ straight = log-periodic',
                   color='#E0E0E0', fontsize=10)
    ax4b.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax4b.legend(fontsize=8)
    fig4.tight_layout()

    plt.show()
    print(f"Embedding: {N-W} Vektoren, M={M}, τ={TAU}d, W={W}d, ab idx {START_IDX}")
    print(f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  PC3={var[2]*100:.1f}%  "
          f"kum3={cum3:.1f}%")


if __name__ == '__main__':
    main()
