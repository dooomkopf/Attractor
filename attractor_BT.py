#!/usr/bin/env python3
"""
Attraktor-Analyse via Delay Embedding der BTC-USD-Residuen.
X-Achse: Blockhöhe (statt Tagesindex).

Fenster 1: Residuen (oben) + PCA 3D Attraktor (unten) + Slider.
Fenster 2: Polar-Phase + PC3 vs θ.
Fenster 3: θ unwrapped – lineare vs. log(block) Periodizität.

Datenquelle: ../Blocktime/ziel_BT.csv
Format:      block_height price yyyy.mm.dd  (Leerzeichen-getrennt)
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
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore")

# ── PARAMETER ─────────────────────────────────────────────────────────────────
PERCENTILE        = 0.01
BLOCKS_PER_DAY    = 144            # Durchschnitt; nur für Parameterumrechnung
# Alle Parameter in Blöcken (nicht Tagen)
START_BLOCK       = 170_895        # 2012-03-12 in /home/hz/Data/Blocktime/ziel_BT.csv; entspricht START_IDX=1164
M                 = 35
NORMALIZE_WINDOWS = False
SMOOTH_SIGMA      = 60             # in Zeilen (Tage), da Signal täglich gesampelt
LABEL_WINDOW      = 4_320          # ± Blöcke ≈ ± 30 Tage um Slider-Position
SHOW_FIG4         = True
CMAP              = plt.cm.coolwarm
# Exakte Halving-Blöcke laut /home/hz/Data/gold/cycles.json,
# verifiziert gegen /home/hz/Data/ziel.csv und /home/hz/Data/Blocktime/ziel_BT.csv.
HALVING_BLOCKS    = [210_058, 420_047, 630_023, 840_128]
HALVING_TOL       = 500
# Harte Zyklus-Tops aus cycles.json.
# Residuen/Peak-Suche treffen die Realraum-Tops nicht immer, daher bewusst fix.
CYCLE_TOPS        = [
    (273_104, "'13-Top", '#0000FF'),   # 04.12.2013
    (499_682, "'17-Top", '#90EE90'),   # 16.12.2017
    (708_992, "'21-Top", '#FF69B4'),   # 09.11.2021
    (918_097, "'25-Top", 'orange'),    # 07.10.2025
]

DATA_FILE = Path(__file__).resolve().parent.parent / "Blocktime" / "ziel_BT.csv"
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
    """Liest ziel_BT.csv: block_height price yyyy.mm.dd, überspringt nan-Preise."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                block = int(parts[0])
                if block <= 0:
                    continue                       # Block 0 → log(0) = -Inf
                if parts[1].lower() == 'nan':
                    continue                       # kein Preis
                price = float(parts[1])
                if not np.isfinite(price) or price <= 0:
                    continue
                dt = datetime.strptime(parts[2], "%Y.%m.%d")
                data.append((block, price, dt))
            except ValueError:
                continue
    blocks = np.array([d[0] for d in data])
    prices = np.array([d[1] for d in data])
    dates  = np.array([d[2] for d in data])
    return blocks, prices, dates


def make_segments(x, y):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def make_segments3d(x, y, z):
    pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def main():
    # ── CLI ───────────────────────────────────────────────────────────────────
    ap = argparse.ArgumentParser(description='BTC Attraktor-Analyse (Blockhöhe)')
    ap.add_argument('--W',     type=int,   default=None, help='Fensterbreite in Blöcken')
    ap.add_argument('--years', type=float, default=None, help='Fensterbreite in Jahren')
    ap.add_argument('--M',     type=int,   default=35,   help='Embedding-Dimensionen (default 35)')
    args = ap.parse_args()

    # TAU/W in Blöcken; intern ÷ BLOCKS_PER_DAY → Zeilenindex (tägl. Daten)
    if args.years is not None or args.W is not None:
        W_blocks = round(args.years * 365 * BLOCKS_PER_DAY) if args.years is not None else args.W
        M        = args.M
        TAU      = max(BLOCKS_PER_DAY, round(W_blocks / (M - 1)))  # in Blöcken
    else:
        M   = args.M
        TAU = max(BLOCKS_PER_DAY, round(round(3.77 * 365 * BLOCKS_PER_DAY) / (M - 1)))  # ≈5828 Blöcke
    _tau_rows = TAU // BLOCKS_PER_DAY   # Zeilen für Array-Indexierung

    # ── Daten laden ───────────────────────────────────────────────────────────
    blocks_all, prices_all, dates_all = read_btc_data(DATA_FILE)

    # ── Quantilregression: log(price) ~ log(block_height) ────────────────────
    log_blocks_all = np.log(blocks_all)
    log_btc_all    = np.log(prices_all)
    X_all          = sm.add_constant(log_blocks_all)
    qr             = QuantReg(log_btc_all, X_all).fit(q=PERCENTILE)
    log_fit_all    = qr.predict(X_all)
    residuals_all  = prices_all / np.exp(log_fit_all)
    log_res_all    = np.log(residuals_all)

    # ── Halving-Indizes (nächste verfügbare Blockhöhe) ────────────────────────
    halving_blocks = []
    halving_dates  = []
    for hb in HALVING_BLOCKS:
        idx = np.argmin(np.abs(blocks_all - hb))
        halving_blocks.append(blocks_all[idx])
        halving_dates.append(dates_all[idx])
    halving_blocks = np.array(halving_blocks)

    # ── Zeitnormierung ────────────────────────────────────────────────────────
    b_min, b_max = blocks_all[0], blocks_all[-1]
    t_norm_all   = (blocks_all - b_min) / (b_max - b_min)

    # Jahres-Labels für Colorbar
    yr_labels = [dates_all[np.argmin(np.abs(blocks_all - b))].strftime('%Y')
                 for b in np.linspace(b_min, b_max, 6)]

    # ── Historische Cycle-Tops (hard-coded) ─────────────────────────────────
    used_peak_blocks, peak_labels, peak_colors = [], [], []
    for block, lbl, col in CYCLE_TOPS:
        used_peak_blocks.append(blocks_all[np.argmin(np.abs(blocks_all - block))])
        peak_labels.append(lbl)
        peak_colors.append(col)
    used_peak_blocks = np.array(used_peak_blocks)
    used_peak_vals   = np.array([residuals_all[np.argmin(np.abs(blocks_all - b))]
                                 for b in used_peak_blocks])

    # ── Delay Embedding ───────────────────────────────────────────────────────
    # TAU in Blöcken; Daten täglich → Zeilenindex = TAU // BLOCKS_PER_DAY
    _W_rows    = (M - 1) * _tau_rows   # Anzahl Zeilen im Embedding-Fenster
    W          = (M - 1) * TAU         # in Blöcken (für Anzeige)

    mask_emb   = blocks_all >= START_BLOCK
    log_res    = log_res_all[mask_emb]
    blocks_emb = blocks_all[mask_emb]
    N          = len(log_res)

    D = np.empty((N - _W_rows, M))
    for j in range(M):
        D[:, j] = log_res[_W_rows - j * _tau_rows : N - j * _tau_rows]
    if NORMALIZE_WINDOWS:
        D -= D.mean(axis=1, keepdims=True)

    D_c      = D - D.mean(axis=0)
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc       = D_c @ Vt.T
    var      = s**2 / (s**2).sum()

    pc_s = pc.copy()
    for j in range(pc_s.shape[1]):
        pc_s[:, j] = gaussian_filter1d(pc[:, j], sigma=SMOOTH_SIGMA)

    blocks_vec = blocks_emb[_W_rows:]
    t_norm_vec = (blocks_vec - b_min) / (b_max - b_min)

    # ── Phase ─────────────────────────────────────────────────────────────────
    theta    = np.arctan2(pc[:, 1], pc[:, 0])
    r        = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
    theta_uw = np.unwrap(theta)
    if theta_uw[-1] < theta_uw[0]:
        theta    = -theta
        theta_uw = -theta_uw

    # Anchor: Halving 2 (420k Blöcke) = Phase 0
    # H1 liegt außerhalb der Embedding-Daten → H2 ist erste nutzbare Referenz
    hidx_h2 = np.argmin(np.abs(blocks_vec - halving_blocks[1]))
    if np.abs(blocks_vec[hidx_h2] - halving_blocks[1]) < 500:
        _shift_w  = float(theta[hidx_h2])      # gewrappter Winkel für theta
        _shift_uw = float(theta_uw[hidx_h2])   # kumulierter Winkel für theta_uw
        theta    = (theta    - _shift_w) % (2 * np.pi)
        theta_uw = theta_uw - _shift_uw        # H2 landet jetzt bei exakt 0

    cum3 = np.cumsum(var)[2] * 100

    # ══════════════════════════════════════════════════════════════════════════
    # Fenster 1: Residuen + 3D Attraktor + Slider
    # ══════════════════════════════════════════════════════════════════════════
    fig_main = plt.figure(figsize=(22, 10))
    fig_main.patch.set_facecolor('black')

    ax1   = fig_main.add_axes([0.04, 0.30, 0.42, 0.43])
    ax2   = fig_main.add_axes([0.50, 0.06, 0.50, 0.90], projection='3d')
    ax_sl = fig_main.add_axes([0.10, 0.02, 0.80, 0.025])
    ax_sl.set_facecolor('#2A2A2A')

    # ── ax1: Residuen ─────────────────────────────────────────────────────────
    ax1.set_facecolor('#1A1A1A')
    segs = make_segments(blocks_all, residuals_all)
    lc1  = LineCollection(segs, cmap=CMAP, linewidth=1.0, alpha=0.9)
    lc1.set_array(t_norm_all)
    ax1.add_collection(lc1)
    ax1.set_yscale('log')
    ax1.set_xlim(0, b_max * 1.05)
    ax1.set_ylim(residuals_all.min() * 0.9, residuals_all.max() * 1.1)
    ax1.axhline(y=1, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)

    valid = np.isfinite(used_peak_vals)
    if valid.any():
        ax1.scatter(used_peak_blocks[valid], used_peak_vals[valid],
                    c=[peak_colors[i] for i in np.where(valid)[0]],
                    s=60, zorder=8, alpha=0.9, label='Cycle Top',
                    edgecolors='white', linewidths=1.0)
        for px, py, lbl, col in zip(used_peak_blocks[valid], used_peak_vals[valid],
                                    [peak_labels[i] for i in np.where(valid)[0]],
                                    [peak_colors[i] for i in np.where(valid)[0]]):
            ax1.annotate(lbl, (px, py), textcoords='offset points',
                         xytext=(6, 6), color=col, fontsize=9, fontweight='bold')
    ax1.scatter(blocks_all[-1], residuals_all[-1], color='white', s=40,
                zorder=10, edgecolors='black', linewidth=1.0, label='Latest')
    for i, (hb, hd) in enumerate(zip(halving_blocks, halving_dates)):
        hidx = np.argmin(np.abs(blocks_all - hb))
        ax1.scatter(blocks_all[hidx], residuals_all[hidx], color='lime', s=60,
                    zorder=10, label='Halving' if i == 0 else '_')
        ax1.annotate(str(i + 1), (blocks_all[hidx], residuals_all[hidx]),
                     textcoords='offset points', xytext=(5, 5),
                     color='lime', fontsize=9, fontweight='bold')

    cb1 = fig_main.colorbar(lc1, ax=ax1, pad=0.01)
    cb1.set_label('Jahr', color='#CCCCCC')
    cb1.set_ticks(np.linspace(0, 1, 6))
    cb1.set_ticklabels(yr_labels)

    # ── Zykluslängen ─────────────────────────────────────────────────────────
    cross_idx = []
    for k in range(1, 30):
        level = k * 2 * np.pi
        idx = np.where((theta_uw[:-1] < level) & (theta_uw[1:] >= level))[0]
        if len(idx) > 0:
            cross_idx.append(idx[0])
    bounds = [0] + cross_idx + [len(theta_uw) - 1]
    if len(cross_idx) > 0:
        ax1.axvline(blocks_vec[cross_idx[0]], color='#FFDD44',
                    linewidth=0.8, alpha=0.6, linestyle=':')
    cycle_durs = []
    for ci in range(1, len(bounds) - 2):
        i1, i2    = bounds[ci], bounds[ci + 1]
        dur_blocks = blocks_vec[i2] - blocks_vec[i1]
        dur_yr    = dur_blocks / (144 * 365.25)
        cycle_durs.append(dur_blocks)
        b_cross   = blocks_vec[i2]
        ax1.axvline(b_cross, color='#FFDD44', linewidth=0.8, alpha=0.6, linestyle=':')
        mid_block = (blocks_vec[i1] + blocks_vec[i2]) / 2
        ax1.text(mid_block, residuals_all.max() * 0.75,
                 f'{dur_yr:.1f}y', color='#FFDD44', fontsize=8,
                 fontweight='bold', ha='center', va='top')

    # ── Nächste Grenze (extrapoliert) ─────────────────────────────────────────
    next_cross_block = b_max
    if len(cross_idx) >= 1 and len(cycle_durs) >= 1:
        med_dur          = float(np.median(cycle_durs))
        last_cross_block = blocks_vec[cross_idx[-1]]
        next_cross_block = last_cross_block + med_dur
        pct = (blocks_vec[-1] - last_cross_block) / med_dur * 100
        pct = max(0.0, min(pct, 100.0))
        print(f"Aktueller Zyklus: {pct:.1f}%  "
              f"(Start: Block {int(last_cross_block)}, "
              f"Median-Länge: {med_dur/144/365.25:.2f}y / {int(med_dur)} Blöcke, "
              f"nächste Grenze: Block {int(next_cross_block)})")
        ax1.axvline(next_cross_block, color='#FFDD44', linewidth=0.8,
                    alpha=0.6, linestyle=':',
                    label=f'Next boundary ({med_dur/144/365.25:.1f}y median)')

    ax1.set_xlim(0, max(b_max, next_cross_block) * 1.05)
    ax1.set_xlabel('Blockhöhe', color='#E0E0E0')
    ax1.set_ylabel('rel. Value', color='#E0E0E0')
    ax1.set_title('BTC USD Residuals (vs. Blockhöhe)', color='#E0E0E0', fontsize=11)
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

    for pb, lbl, col in zip(used_peak_blocks, peak_labels, peak_colors):
        idx_ = np.argmin(np.abs(blocks_vec - pb))
        if np.abs(blocks_vec[idx_] - pb) < HALVING_TOL:
            ax2.scatter(pc_s[idx_, 0], pc_s[idx_, 1], pc_s[idx_, 2],
                        color=col, s=60, zorder=10,
                        edgecolors='white', linewidths=1.0,
                        label='Cycle Top' if lbl == peak_labels[0] else '_')
            ax2.text(pc_s[idx_, 0], pc_s[idx_, 1], pc_s[idx_, 2],
                     f' {lbl}', color=col, fontsize=9, fontweight='bold')

    for i, hb in enumerate(halving_blocks):
        hidx = np.argmin(np.abs(blocks_vec - hb))
        if np.abs(blocks_vec[hidx] - hb) < 500:
            ax2.scatter(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                        color='lime', s=60, zorder=10,
                        label='Halving' if i == 0 else '_')
            ax2.text(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                     f' {i+1}', color='lime', fontsize=9, fontweight='bold')

    ax2.scatter(*[pc_s[-1, i] for i in range(3)],
                color='white', s=60, zorder=10, label='Now')

    for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#444444')

    ax2.set_xlabel(f'PC1  ({var[0]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_ylabel(f'PC2  ({var[1]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_zlabel(f'PC3  ({var[2]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_title(f'PCA 3D Attraktor  ·  τ={TAU} · m={M} · W={W} Blöcke  (kum. {cum3:.1f}%)',
                  color='#E0E0E0', fontsize=10)
    ax2.tick_params(colors='#CCCCCC')
    ax2.legend(fontsize=8)

    # ── Slider ────────────────────────────────────────────────────────────────
    slider = Slider(ax_sl, 'Block', float(blocks_vec[0]), float(blocks_vec[-1]),
                    valinit=float(blocks_vec[len(blocks_vec)//2]),
                    color='#666666')
    slider.label.set_color('#CCCCCC')
    slider.valtext.set_color('#CCCCCC')

    hl = {'ax1_scat': None, 'ax2_scat': None}

    def update(val):
        b = slider.val
        for key in hl:
            if hl[key] is not None:
                try:
                    hl[key].remove()
                except Exception:
                    pass
                hl[key] = None
        mask1 = (blocks_all >= b - LABEL_WINDOW) & (blocks_all <= b + LABEL_WINDOW)
        if mask1.any():
            hl['ax1_scat'] = ax1.scatter(
                blocks_all[mask1], residuals_all[mask1],
                color='#FFFF00', s=80, alpha=1.0, zorder=9, linewidths=0)
        mask2 = (blocks_vec >= b - LABEL_WINDOW) & (blocks_vec <= b + LABEL_WINDOW)
        if mask2.any():
            hl['ax2_scat'] = ax2.scatter(
                pc_s[mask2, 0], pc_s[mask2, 1], pc_s[mask2, 2],
                color='#FFFF00', s=120, alpha=1.0, zorder=10)
        fig_main.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)

    # ══════════════════════════════════════════════════════════════════════════
    # Fenster 2: Polarkoordinaten
    # ══════════════════════════════════════════════════════════════════════════
    fig3, _ = plt.subplots(1, 2, figsize=(14, 7))
    fig3.patch.set_facecolor('black')
    _[0].set_visible(False)
    ax3a = fig3.add_subplot(1, 2, 1, projection='polar')
    ax3b = fig3.axes[1]
    ax3b.set_facecolor('#1A1A1A')

    segs_pol = make_segments(theta, r)
    lc3a = LineCollection(segs_pol, cmap=CMAP, linewidth=0.7, alpha=0.85)
    lc3a.set_array(t_norm_vec)
    ax3a.add_collection(lc3a)
    ax3a.set_ylim(0, r.max() * 1.1)
    ax3a.scatter(theta[-1], r[-1], color='white', s=80, zorder=10, label='Jetzt')
    for pb, lbl, col in zip(used_peak_blocks, peak_labels, peak_colors):
        idx_ = np.argmin(np.abs(blocks_vec - pb))
        if np.abs(blocks_vec[idx_] - pb) < HALVING_TOL:
            ax3a.scatter(theta[idx_], r[idx_], color=col, s=60, zorder=10)
            ax3a.annotate(lbl, (theta[idx_], r[idx_]),
                          color=col, fontsize=9, fontweight='bold')
    ax3a.set_yticklabels([])
    ax3a.set_rticks([])
    ax3a.set_title('Phase vs Intensity', color='#E0E0E0', fontsize=10, pad=15)
    ax3a.tick_params(colors='#CCCCCC')
    ax3a.set_facecolor('#1A1A1A')

    lc3b = ax3b.scatter(theta, pc[:, 2], c=t_norm_vec, cmap=CMAP,
                        s=2, alpha=0.7, linewidths=0)
    ax3b.scatter(theta[-1], pc[-1, 2], color='white', s=80, zorder=10, label='Jetzt')
    for pb, lbl, col in zip(used_peak_blocks, peak_labels, peak_colors):
        idx_ = np.argmin(np.abs(blocks_vec - pb))
        if np.abs(blocks_vec[idx_] - pb) < HALVING_TOL:
            ax3b.scatter(theta[idx_], pc[idx_, 2], color=col, s=60, zorder=10)
            ax3b.annotate(lbl, (theta[idx_], pc[idx_, 2]),
                          textcoords='offset points', xytext=(6, 4),
                          color=col, fontsize=9, fontweight='bold')
    ax3b.set_xlabel('Cycle phase  [rad]  (H2=0)', color='#E0E0E0')
    ax3b.set_ylabel(f'PC3  ({var[2]*100:.1f}%)', color='#E0E0E0')
    ax3b.set_title('Phase', color='#E0E0E0', fontsize=10)
    ax3b.set_xlim(0, 2 * np.pi)
    ax3b.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,
                     5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi])
    ax3b.set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{2\pi}{4}$', r'$\frac{3\pi}{4}$',
                           r'$\pi$', r'$\frac{5\pi}{4}$', r'$\frac{6\pi}{4}$',
                           r'$\frac{7\pi}{4}$', r'$2\pi$'])
    ax3b.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    for i, hb in enumerate(halving_blocks):
        hidx = np.argmin(np.abs(blocks_vec - hb))
        if np.abs(blocks_vec[hidx] - hb) < 500:
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
    cb3_ticks = np.linspace(t_norm_vec[0], t_norm_vec[-1], 6)
    cb3_labels = [dates_all[np.argmin(np.abs(blocks_all - (t * (b_max - b_min) + b_min)))].strftime('%Y')
                  for t in cb3_ticks]
    cb3.set_ticks(cb3_ticks)
    cb3.set_ticklabels(cb3_labels)
    ax3a.legend(loc='lower right', fontsize=8)
    ax3b.legend(fontsize=8)
    fig3.tight_layout()
    fig3.subplots_adjust(top=0.88)

    # ══════════════════════════════════════════════════════════════════════════
    # Fenster 3: θ unwrapped – linear vs. log(block)
    # ══════════════════════════════════════════════════════════════════════════
    if not SHOW_FIG4:
        plt.show()
        print(f"Embedding: {N-_W_rows} Vektoren, M={M}, τ={TAU}, W={W} Blöcke, ab Block {START_BLOCK}")
        print(f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  PC3={var[2]*100:.1f}%  kum3={cum3:.1f}%")
        return

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.patch.set_facecolor('black')

    def _halving_on(ax, xvals):
        for i, (hb, xv) in enumerate(zip(halving_blocks, xvals)):
            hidx = np.argmin(np.abs(blocks_vec - hb))
            if np.abs(blocks_vec[hidx] - hb) < 500:
                ax.scatter(xv, theta_uw[hidx], color='lime', s=70, zorder=10,
                           label='Halving' if i == 0 else '_')
                ax.annotate(str(i + 1), (xv, theta_uw[hidx]),
                            textcoords='offset points', xytext=(5, 5),
                            color='lime', fontsize=9, fontweight='bold')

    ax4a.set_facecolor('#1A1A1A')
    segs4a = make_segments(blocks_vec, theta_uw)
    lc4a   = LineCollection(segs4a, cmap=CMAP, linewidth=0.8, alpha=0.85)
    lc4a.set_array(t_norm_vec)
    ax4a.add_collection(lc4a)
    ax4a.autoscale()
    _halving_on(ax4a, halving_blocks)
    for ci in cross_idx:
        ax4a.axvline(blocks_vec[ci], color='#FFDD44', linewidth=0.8, alpha=0.6, linestyle=':')
    ax4a.axvline(next_cross_block, color='#FFDD44', linewidth=0.8, alpha=0.6, linestyle=':')
    ax4a.set_xlabel('Blockhöhe  (linear)', color='#E0E0E0')
    ax4a.set_ylabel('θ unwrapped  [rad]', color='#E0E0E0')
    ax4a.set_title('Cycle phase vs Block  (linear)\n→ gerade = konstante Periode',
                   color='#E0E0E0', fontsize=10)
    ax4a.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    handles4a, _ = ax4a.get_legend_handles_labels()
    if handles4a:
        ax4a.legend(fontsize=8)

    ax4b.set_facecolor('#1A1A1A')
    log_blocks_vec = np.log(blocks_vec)
    segs4b = make_segments(log_blocks_vec, theta_uw)
    lc4b   = LineCollection(segs4b, cmap=CMAP, linewidth=0.8, alpha=0.85)
    lc4b.set_array(t_norm_vec)
    ax4b.add_collection(lc4b)
    ax4b.autoscale()
    _halving_on(ax4b, np.log(halving_blocks))
    for ci in cross_idx:
        ax4b.axvline(np.log(blocks_vec[ci]), color='#FFDD44', linewidth=0.8, alpha=0.6, linestyle=':')
    ax4b.axvline(np.log(next_cross_block), color='#FFDD44', linewidth=0.8, alpha=0.6, linestyle=':')
    ax4b.set_xlabel('log(Blockhöhe)', color='#E0E0E0')
    ax4b.set_ylabel('θ unwrapped  [rad]', color='#E0E0E0')
    ax4b.set_title('Cycle phase vs log(Block)\n→ gerade = log-periodisch',
                   color='#E0E0E0', fontsize=10)
    ax4b.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    handles4b, _ = ax4b.get_legend_handles_labels()
    if handles4b:
        ax4b.legend(fontsize=8)
    fig4.tight_layout()

    plt.show()
    print(f"Embedding: {N-_W_rows} Vektoren, M={M}, τ={TAU}, W={W} Blöcke, ab Block {START_BLOCK}")
    print(f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  PC3={var[2]*100:.1f}%  kum3={cum3:.1f}%")


if __name__ == '__main__':
    main()
