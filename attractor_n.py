#!/usr/bin/env python3
"""
attractor_n.py — Wang-Attraktor-Analyse via Delay Embedding der BTC log-log-Steigung.

Signal:  n(t) = log(p[t+HALF] / p[t-HALF]) / log(day[t+HALF] / day[t-HALF])
         zentriertes 180d-Fenster, kein Buffer, kein Zyklus-Split.

Gleiche Parameter wie attractor.py:
  TAU=30, M=50, SMOOTH_SIGMA=60, START_IDX=1164
  PCA → (PC1, PC2, PC3), Phase θ = atan2(PC2,PC1)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.widgets import Slider, Button
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── PARAMETER ─────────────────────────────────────────────────────────────────
START_IDX         = 1164
WINDOW            = 180          # zentriertes Fenster für n(t)
TAU               = 30
M                 = 50
NORMALIZE_WINDOWS = False
SMOOTH_SIGMA      = 60
LABEL_WINDOW      = 30
SHOW_FIG4         = False
PHASE_OFFSET      = -2.108           # Phasen-Offset in Rad: Halving-2 auf attractor_cy.py ausgerichtet
CMAP              = plt.cm.coolwarm
HALVINGS          = [datetime(2012, 11, 28), datetime(2016, 7, 9),
                     datetime(2020, 5, 11),  datetime(2024, 4, 20)]
CYCLE_TOPS        = [
    (datetime(2013, 12,  4), "'13-Top", '#0000FF'),
    (datetime(2017, 12, 16), "'17-Top", '#90EE90'),
    (datetime(2021, 11,  9), "'21-Top", '#FF69B4'),
    (datetime(2025, 10,  7), "'25-Top", 'orange'),
]
CYCLE_BOTTOMS     = [
    (datetime(2015,  1, 14), "'15-Bottom", 183),    # day 2202, min price between '13-Top and '17-Top
    (datetime(2018, 12, 14), "'18-Bottom", 3282),   # day 3632, min price between '17-Top and '21-Top
    (datetime(2022, 11, 21), "'22-Bottom", 15766),  # day 5070, min price between '21-Top and '25-Top
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

    # ── n(t) berechnen: zentriertes WINDOW-d-Fenster ─────────────────────────
    half = WINDOW // 2
    n_vals  = []
    n_days  = []
    n_dates = []
    for i in range(half, len(days_all) - half):
        t1, t2 = float(days_all[i - half]), float(days_all[i + half])
        p1, p2 = prices_all[i - half], prices_all[i + half]
        if p1 > 0 and p2 > 0 and t2 > t1:
            denom = np.log(t2 / t1)
            n_vals.append(np.log(p2 / p1) / denom if denom != 0 else np.nan)
        else:
            n_vals.append(np.nan)
        n_days.append(days_all[i])
        n_dates.append(dates_all[i])
    daily_n_all = np.array(n_vals)
    days_n_all  = np.array(n_days)
    dates_n_all = np.array(n_dates)

    # ── Historische Cycle-Tops (hard-coded) ───────────────────────────────────
    used_peak_days, peak_labels, peak_colors = [], [], []
    for dt, lbl, col in CYCLE_TOPS:
        diffs = np.array([abs((d - dt).days) for d in dates_all])
        used_peak_days.append(days_all[np.argmin(diffs)])
        peak_labels.append(lbl)
        peak_colors.append(col)
    used_peak_days = np.array(used_peak_days)
    peak_n_vals = np.array([
        daily_n_all[np.argmin(np.abs(days_n_all - d))]
        for d in used_peak_days
    ])

    # ── Historische Cycle-Bottoms (hard-coded) ────────────────────────────────
    used_bottom_days, bottom_labels = [], []
    for dt, lbl, _ in CYCLE_BOTTOMS:
        diffs = np.array([abs((d - dt).days) for d in dates_all])
        used_bottom_days.append(days_all[np.argmin(diffs)])
        bottom_labels.append(lbl)
    used_bottom_days = np.array(used_bottom_days)
    bottom_n_vals = np.array([
        daily_n_all[np.argmin(np.abs(days_n_all - d))]
        for d in used_bottom_days
    ])

    # ── Halving-Indizes ───────────────────────────────────────────────────────
    halving_days = []
    for hd in HALVINGS:
        diffs = np.array([abs((d - hd).days) for d in dates_all])
        halving_days.append(days_all[np.argmin(diffs)])
    halving_days = np.array(halving_days)

    # ── Zeitnormierung ────────────────────────────────────────────────────────
    d_min, d_max = days_all[0], days_all[-1]
    t_norm_n_all = (days_n_all - d_min) / (d_max - d_min)

    yr_labels = [dates_all[np.argmin(np.abs(days_all - y))].strftime('%Y')
                 for y in np.linspace(d_min, d_max, 6)]

    # ── Delay Embedding ───────────────────────────────────────────────────────
    mask_emb  = (days_n_all >= START_IDX - WINDOW//2) & np.isfinite(daily_n_all)
    sig       = daily_n_all[mask_emb]
    days_emb  = days_n_all[mask_emb]
    dates_emb = dates_n_all[mask_emb]
    N = len(sig)
    W = (M - 1) * TAU

    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = sig[W - j * TAU : N - j * TAU]
    if NORMALIZE_WINDOWS:
        D -= D.mean(axis=1, keepdims=True)

    D_c      = D - D.mean(axis=0)
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc       = D_c @ Vt.T
    var      = s**2 / (s**2).sum()

    pc_s = pc.copy()
    for j in range(pc_s.shape[1]):
        pc_s[:, j] = gaussian_filter1d(pc[:, j], sigma=SMOOTH_SIGMA)

    days_vecs  = days_emb[W:]
    dates_vecs = dates_emb[W:]
    t_norm_vec = (days_vecs - d_min) / (d_max - d_min)

    # Phasen-Rotation auf PC1/PC2 anwenden → wirkt in ALLEN Plots
    _cos, _sin = np.cos(PHASE_OFFSET), np.sin(PHASE_OFFSET)
    for _arr in (pc, pc_s):
        _pc1 = _arr[:, 0] * _cos - _arr[:, 1] * _sin
        _pc2 = _arr[:, 0] * _sin + _arr[:, 1] * _cos
        _arr[:, 0], _arr[:, 1] = _pc1, _pc2

    # PC3 Vorzeichen an attractor_cy.py angleichen
    pc[:, 2]   *= -1
    pc_s[:, 2] *= -1

    theta    = np.arctan2(pc[:, 1], pc[:, 0])
    r        = np.sqrt(pc[:, 0]**2 + pc[:, 1]**2)
    theta_uw = np.unwrap(theta)
    theta_uw = np.unwrap(theta)

    # letzte 90d abschneiden (Vorlauf-Kompensation)
    _cut       = WINDOW // 2
    pc         = pc[:-_cut]
    pc_s       = pc_s[:-_cut]
    days_vecs  = days_vecs[:-_cut]
    dates_vecs = dates_vecs[:-_cut]
    t_norm_vec = t_norm_vec[:-_cut]
    theta      = theta[:-_cut]
    r          = r[:-_cut]
    theta_uw   = theta_uw[:-_cut]

    cum3 = np.cumsum(var)[2] * 100

    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_vecs - hday))
        if np.abs(days_vecs[hidx] - hday) < 200:
            print(f"Halving {i+1}: theta = {theta[hidx]:.4f} rad  ({np.degrees(theta[hidx]):.1f}°)")

    print(f"daily_n: {N} Werte  (ab idx {START_IDX})")
    print(f"Embedding: {N-W} Vektoren  M={M} τ={TAU}d W={W}d")
    print(f"PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  PC3={var[2]*100:.1f}%  kum3={cum3:.1f}%")
    print(f"daily_n: min={np.nanmin(daily_n_all[mask_emb]):.2f}  "
          f"max={np.nanmax(daily_n_all[mask_emb]):.2f}  "
          f"mean={np.nanmean(daily_n_all[mask_emb]):.2f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Fenster 1: daily_n (links) + 3D Attraktor (rechts)
    # ══════════════════════════════════════════════════════════════════════════
    fig_main = plt.figure(figsize=(22, 10))
    fig_main.patch.set_facecolor('black')

    ax1 = fig_main.add_axes([0.04, 0.30, 0.42, 0.43])
    ax2 = fig_main.add_axes([0.50, 0.06, 0.50, 0.90], projection='3d', elev=6, azim=-153)
    ax_sl = fig_main.add_axes([0.10, 0.02, 0.80, 0.025])
    ax_sl.set_facecolor('#2A2A2A')

    # ── ax1: daily_n ─────────────────────────────────────────────────────────
    ax1.set_facecolor('#1A1A1A')

    # Roh
    mask_plot = days_n_all >= START_IDX
    segs_raw = make_segments(days_n_all[mask_plot], daily_n_all[mask_plot])
    lc_raw   = LineCollection(segs_raw, cmap=CMAP, linewidth=0.8, alpha=0.9)
    lc_raw.set_array(t_norm_n_all[mask_plot])
    ax1.add_collection(lc_raw)

    ax1.autoscale()
    ax1.axhline(y=0, color='#808080', linestyle='--', alpha=0.5, linewidth=1.0)

    # Halving-Vertikallinien
    for hday in halving_days:
        ax1.axvline(x=hday, color='grey', linestyle='--', alpha=0.4, linewidth=0.8, zorder=1)

    # Mittelwert (zartrote Linie)
    _mean_n = np.nanmean(daily_n_all[mask_plot])
    ax1.axhline(y=_mean_n, color='#FF9999', linestyle='-', alpha=0.8, linewidth=1.0, zorder=2, label='Mittelwert')

    # Halvings
    for i, hday in enumerate(halving_days):
        hidx_n = np.argmin(np.abs(days_n_all - hday))
        ax1.scatter(days_n_all[hidx_n], daily_n_all[hidx_n],
                    color='grey', s=60, zorder=10,
                    edgecolors='white', linewidths=1.0,
                    label='Halving' if i == 0 else '_')
        ax1.annotate(str(i+1), (days_n_all[hidx_n], daily_n_all[hidx_n]),
                     textcoords='offset points', xytext=(5, 5),
                     color='white', fontsize=9, fontweight='bold')

    # Peaks (Tops)
    valid = np.isfinite(peak_n_vals)
    if valid.any():
        ax1.scatter(used_peak_days[valid], peak_n_vals[valid],
                    c=[peak_colors[i] for i in np.where(valid)[0]],
                    s=60, zorder=8, alpha=0.9, label='Cycle Top',
                    edgecolors='white', linewidths=1.0)
        for px, py, lbl, col in zip(used_peak_days[valid], peak_n_vals[valid],
                                    [peak_labels[i] for i in np.where(valid)[0]],
                                    [peak_colors[i] for i in np.where(valid)[0]]):
            ax1.annotate(lbl, (px, py), textcoords='offset points',
                         xytext=(6, 6), color=col, fontsize=9, fontweight='bold')

    # Bottoms
    valid_b = np.isfinite(bottom_n_vals)
    if valid_b.any():
        ax1.scatter(used_bottom_days[valid_b], bottom_n_vals[valid_b],
                    color='grey', marker='s', s=60, zorder=8, alpha=0.9,
                    edgecolors='white', linewidths=1.0, label='Cycle Bottom')
        for bx, by, lbl in zip(used_bottom_days[valid_b], bottom_n_vals[valid_b],
                                [bottom_labels[i] for i in np.where(valid_b)[0]]):
            ax1.annotate(lbl, (bx, by), textcoords='offset points',
                         xytext=(6, -12), color='#AAAAAA', fontsize=9, fontweight='bold')

    ax1.scatter(days_n_all[-1], daily_n_all[-1], color='white', s=40,
                zorder=10, edgecolors='black', linewidth=1.0, label='Latest Data')

    ax1.set_ylim(-60, 60)

    # Geglättete Zyklinien in Zyklusfarben (zwischen Halvings)
    _cyc_colors = ['grey', '#0000FF', '#90EE90', '#FF69B4', 'orange']
    _bounds = np.concatenate([[days_n_all[0]], halving_days, [days_n_all[-1]]])
    for k in range(len(_bounds) - 1):
        _d0, _d1 = _bounds[k], _bounds[k + 1]
        _col = _cyc_colors[k] if k < len(_cyc_colors) else 'white'
        _mc = (days_n_all >= max(_d0, START_IDX)) & (days_n_all <= _d1) & np.isfinite(daily_n_all)
        _dc, _vc = days_n_all[_mc], daily_n_all[_mc]
        if len(_vc) > 121:
            _sm = gaussian_filter1d(_vc, sigma=60)
            ax1.plot(_dc, _sm, color=_col, linewidth=1.5, alpha=0.7, zorder=5)

    cb1 = fig_main.colorbar(lc_raw, ax=ax1, pad=0.01)
    cb1.set_label('Jahr', color='#CCCCCC')
    cb1.set_ticks(np.linspace(0, 1, 6))
    cb1.set_ticklabels(yr_labels)

    ax1.set_ylabel('daily_n  =  log(p₂/p₁) / log(t₂/t₁)', color='#E0E0E0')
    ax1.set_title(f'BTC Power-Law-Exponent  (zentriertes {WINDOW}d-Fenster)',
                  color='#E0E0E0', fontsize=11)
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
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

    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_vecs - hday))
        if np.abs(days_vecs[hidx] - hday) < 200:
            ax2.scatter(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                        color='grey', s=60, zorder=10,
                        edgecolors='white', linewidths=1.0,
                        label='Halving' if i == 0 else '_')
            ax2.text(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2], f' {i+1}',
                     color='white', fontsize=9, fontweight='bold')

    # Peaks in 3D
    for pd_, lbl, col in zip(used_peak_days, peak_labels, peak_colors):
        hidx = np.argmin(np.abs(days_vecs - pd_))
        if np.abs(days_vecs[hidx] - pd_) < 200:
            ax2.scatter(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                        color=col, s=60, zorder=10,
                        edgecolors='white', linewidths=1.0,
                        label='Cycle Top' if lbl == peak_labels[0] else '_')
            ax2.text(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2], f' {lbl}',
                     color=col, fontsize=9, fontweight='bold')

    # Bottoms in 3D
    for i, (bd_, lbl) in enumerate(zip(used_bottom_days, bottom_labels)):
        hidx = np.argmin(np.abs(days_vecs - bd_))
        if np.abs(days_vecs[hidx] - bd_) < 200:
            ax2.scatter(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                        color='grey', marker='s', s=60, zorder=10,
                        edgecolors='white', linewidths=1.0,
                        label='Cycle Bottom' if i == 0 else '_')
            ax2.text(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2], f' {lbl}',
                     color='#AAAAAA', fontsize=9, fontweight='bold')

    ax2.scatter(*[pc_s[-1, i] for i in range(3)],
                color='white', s=60, zorder=10, label='Latest Data')

    for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#444444')

    ax2.set_xlabel(f'PC1  ({var[0]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_ylabel(f'PC2  ({var[1]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_zlabel(f'PC3  ({var[2]*100:.1f}%)', color='#CCCCCC', labelpad=8)
    ax2.set_title(f'PCA 3D Attraktor (daily_n)  ·  τ={TAU}d · m={M} · W={W}d  (kum. {cum3:.1f}%)',
                  color='#E0E0E0', fontsize=10)
    ax2.tick_params(colors='#CCCCCC')
    ax2.legend(fontsize=8)

    # ── Cycle-Colors Button ───────────────────────────────────────────────────
    _cyc_colors = ['grey', '#0000FF', '#90EE90', '#FF69B4', 'orange']
    _bounds = np.concatenate([[days_vecs[0]], halving_days, [days_vecs[-1]]])
    _cyc_lines = []
    for k in range(len(_bounds) - 1):
        _d0, _d1 = _bounds[k], _bounds[k + 1]
        _col = _cyc_colors[k] if k < len(_cyc_colors) else 'white'
        _mc = (days_vecs >= _d0) & (days_vecs <= _d1)
        if _mc.sum() > 1:
            _ln, = ax2.plot(pc_s[_mc, 0], pc_s[_mc, 1], pc_s[_mc, 2],
                            color=_col, linewidth=2.5, alpha=0.9, visible=False)
            _cyc_lines.append(_ln)

    ax_btn = fig_main.add_axes([0.01, 0.02, 0.08, 0.025])
    btn_cyc = Button(ax_btn, 'Cycle-Colors', color='#333333', hovercolor='#555555')
    btn_cyc.label.set_color('#CCCCCC')
    _cyc_active = [False]

    def _toggle_cyc(event):
        _cyc_active[0] = not _cyc_active[0]
        lc2.set_visible(not _cyc_active[0])
        for _l in _cyc_lines:
            _l.set_visible(_cyc_active[0])
        fig_main.canvas.draw_idle()

    btn_cyc.on_clicked(_toggle_cyc)

    # ── Slider ────────────────────────────────────────────────────────────────
    slider = Slider(ax_sl, 'Tag', float(days_n_all[mask_plot][0]), float(days_vecs[-1]),
                    valinit=float(days_vecs[len(days_vecs)//2]),
                    color='#666666')
    slider.label.set_color('#CCCCCC')
    slider.valtext.set_color('#CCCCCC')

    hl = {'ax1_scat': None, 'ax2_scat': None}

    def update(val):
        d = slider.val
        for key in hl:
            if hl[key] is not None:
                try:
                    hl[key].remove()
                except Exception:
                    pass
                hl[key] = None
        mask1 = (days_n_all >= d - LABEL_WINDOW) & (days_n_all <= d + LABEL_WINDOW)
        if mask1.any():
            hl['ax1_scat'] = ax1.scatter(
                days_n_all[mask1], daily_n_all[mask1],
                color='#FFFF00', s=80, alpha=1.0, zorder=9, linewidths=0)
        mask2 = (days_vecs >= d - LABEL_WINDOW) & (days_vecs <= d + LABEL_WINDOW)
        if mask2.any():
            hl['ax2_scat'] = ax2.scatter(
                pc_s[mask2, 0], pc_s[mask2, 1], pc_s[mask2, 2],
                color='#FFFF00', s=120, alpha=1.0, zorder=10)
        fig_main.canvas.draw_idle()

    slider.on_changed(update)
    update(slider.val)

    # ══════════════════════════════════════════════════════════════════════════
    # Fenster 2: Polar-Phase + PC3 vs θ
    # ══════════════════════════════════════════════════════════════════════════
    fig3, _ = plt.subplots(1, 2, figsize=(14, 7))
    fig3.patch.set_facecolor('black')
    ax3a = fig3.add_subplot(1, 2, 1, projection='polar')
    ax3b = fig3.axes[1]
    ax3b.set_facecolor('#1A1A1A')

    segs_pol = make_segments(theta, r)
    lc3a = LineCollection(segs_pol, cmap=CMAP, linewidth=0.7, alpha=0.85)
    lc3a.set_array(t_norm_vec)
    ax3a.add_collection(lc3a)
    ax3a.set_ylim(0, r.max() * 1.1)
    ax3a.scatter(theta[-1], r[-1], color='white', s=80, zorder=10, label='Latest Data')
    ax3a.set_yticklabels([])
    ax3a.set_title('Phase vs Intensität (daily_n)', color='#E0E0E0', fontsize=10, pad=15)
    ax3a.tick_params(colors='#CCCCCC')
    ax3a.set_facecolor('#1A1A1A')

    for j, (pd_, lbl, col) in enumerate(zip(used_peak_days, peak_labels, peak_colors)):
        idx_ = np.argmin(np.abs(days_vecs - pd_))
        if np.abs(days_vecs[idx_] - pd_) < 200:
            ax3a.scatter(theta[idx_], r[idx_], color=col, s=60, zorder=10,
                         edgecolors='white', linewidths=1.0,
                         label='Cycle Top' if j == 0 else '_')
            ax3a.annotate(lbl, (theta[idx_], r[idx_]),
                          textcoords='offset points', xytext=(6, 4),
                          color=col, fontsize=9, fontweight='bold')

    for i_b, (bd_, lbl) in enumerate(zip(used_bottom_days, bottom_labels)):
        idx_ = np.argmin(np.abs(days_vecs - bd_))
        if np.abs(days_vecs[idx_] - bd_) < 200:
            ax3a.scatter(theta[idx_], r[idx_], color='grey', marker='s', s=60, zorder=10,
                         edgecolors='white', linewidths=1.0,
                         label='Cycle Bottom' if i_b == 0 else '_')
            ax3a.annotate(lbl, (theta[idx_], r[idx_]),
                          textcoords='offset points', xytext=(6, 4),
                          color='#AAAAAA', fontsize=9, fontweight='bold')

    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_vecs - hday))
        if np.abs(days_vecs[hidx] - hday) < 200:
            ax3a.scatter(theta[hidx], r[hidx], color='grey', s=60, zorder=10,
                         edgecolors='white', linewidths=1.0,
                         label='Halving' if i == 0 else '_')
            ax3a.annotate(str(i+1), (theta[hidx], r[hidx]),
                          textcoords='offset points', xytext=(8, 6),
                          color='white', fontsize=9, fontweight='bold')

    lc3b = ax3b.scatter(theta, pc[:, 2], c=t_norm_vec, cmap=CMAP,
                        s=2, alpha=0.7, linewidths=0)
    ax3b.scatter(theta[-1], pc[-1, 2], color='white', s=80, zorder=10, label='Latest Data')
    for j3, (pd_, lbl, col) in enumerate(zip(used_peak_days, peak_labels, peak_colors)):
        idx_ = np.argmin(np.abs(days_vecs - pd_))
        if np.abs(days_vecs[idx_] - pd_) < 200:
            ax3b.scatter(theta[idx_], pc[idx_, 2], color=col, s=60, zorder=10,
                         edgecolors='white', linewidths=1.0,
                         label='Cycle Top' if j3 == 0 else '_')
            ax3b.annotate(lbl, (theta[idx_], pc[idx_, 2]),
                          textcoords='offset points', xytext=(6, 4),
                          color=col, fontsize=9, fontweight='bold')
    for i_b, (bd_, lbl) in enumerate(zip(used_bottom_days, bottom_labels)):
        idx_ = np.argmin(np.abs(days_vecs - bd_))
        if np.abs(days_vecs[idx_] - bd_) < 200:
            ax3b.scatter(theta[idx_], pc[idx_, 2], color='grey', marker='s', s=60, zorder=10,
                         edgecolors='white', linewidths=1.0,
                         label='Cycle Bottom' if i_b == 0 else '_')
            ax3b.annotate(lbl, (theta[idx_], pc[idx_, 2]),
                          textcoords='offset points', xytext=(6, 4),
                          color='#AAAAAA', fontsize=9, fontweight='bold')
    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_vecs - hday))
        if np.abs(days_vecs[hidx] - hday) < 200:
            ax3b.scatter(theta[hidx], pc[hidx, 2], color='grey', s=60, zorder=10,
                         edgecolors='white', linewidths=1.0,
                         label='Halving' if i == 0 else '_')
            ax3b.annotate(str(i+1), (theta[hidx], pc[hidx, 2]),
                          textcoords='offset points', xytext=(6, 4),
                          color='white', fontsize=9, fontweight='bold')
    ax3b.set_xlabel('θ = atan2(PC2, PC1)  [Zyklusphase]', color='#E0E0E0')
    ax3b.set_ylabel(f'PC3  ({var[2]*100:.1f}%)', color='#E0E0E0')
    ax3b.set_title('PC3 vs Phase (daily_n)', color='#E0E0E0', fontsize=10)
    ax3b.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    cb3 = fig3.colorbar(lc3b, ax=ax3b, pad=0.02)
    cb3.set_label('Jahr', color='#CCCCCC')
    cb3.set_ticks(np.linspace(0, 1, 6))
    cb3.set_ticklabels(yr_labels)
    ax3a.legend(loc='lower right', fontsize=8)
    ax3b.legend(fontsize=8)
    fig3.tight_layout()

    # ══════════════════════════════════════════════════════════════════════════
    # Fenster 3: θ unwrapped – linear vs log(t)
    # ══════════════════════════════════════════════════════════════════════════
    if not SHOW_FIG4:
        plt.show()
        return

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.patch.set_facecolor('black')

    def _halving_on(ax, xvals):
        for i, (hday, xv) in enumerate(zip(halving_days, xvals)):
            hidx = np.argmin(np.abs(days_vecs - hday))
            if np.abs(days_vecs[hidx] - hday) < 200:
                ax.scatter(xv, theta_uw[hidx], color='grey', s=70, zorder=10,
                           edgecolors='white', linewidths=1.0,
                           label='Halving' if i == 0 else '_')
                ax.annotate(str(i+1), (xv, theta_uw[hidx]),
                            textcoords='offset points', xytext=(5, 5),
                            color='white', fontsize=9, fontweight='bold')

    ax4a.set_facecolor('#1A1A1A')
    segs4a = make_segments(days_vecs, theta_uw)
    lc4a   = LineCollection(segs4a, cmap=CMAP, linewidth=0.8, alpha=0.85)
    lc4a.set_array(t_norm_vec)
    ax4a.add_collection(lc4a)
    ax4a.autoscale()
    _halving_on(ax4a, halving_days)
    ax4a.set_xlabel('Day index  (linear)', color='#E0E0E0')
    ax4a.set_ylabel('θ unwrapped  [rad]', color='#E0E0E0')
    ax4a.set_title('Cycle phase vs t  (linear)', color='#E0E0E0', fontsize=10)
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
    ax4b.set_title('Cycle phase vs log(t)', color='#E0E0E0', fontsize=10)
    ax4b.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax4b.legend(fontsize=8)
    fig4.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
