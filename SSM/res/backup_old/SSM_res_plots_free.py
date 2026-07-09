"""SSM/res/plots_free.py — Plot-Funktionen für die modellfreie Pipeline.

Drei Plots ohne SSM-Annahmen:

Fig 1:  plot_pure_tde            — 3D der ROHEN Lag-Koordinaten
                                   X(t), X(t-τ), X(t-2τ) ohne PCA, ohne Fit
Fig 2:  plot_intrinsic_dim       — TWO-NN d_intr (Histogramm + Linear-Fit-Plot)
Fig 3:  plot_local_geometry      — Trajektorie farbcodiert mit lokaler 2D-ness
                                   aus Local PCA  (keine globale Polynom-Form)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib  # noqa: F401 — 3D-Backend


# ── Fig 1: Pure TDE 3D ───────────────────────────────────────────────────────
def plot_pure_tde(D, days_vecs, dates_vecs, halving_days,
                  cycle_top_days, cycle_top_labels, M, TAU,
                  fig_num=11, sigma_smooth=60):
    """3D-Plot der ROHEN Lag-Koordinaten ohne PCA.

    WICHTIG zur Achsen-Wahl:
        Wir nehmen NICHT die ersten drei Lag-Spalten D[:, 0..2], weil bei
        einer Halving-Periode ~1400d und τ=41d die Lags 0, τ, 2τ nur 0, 41,
        82d auseinander liegen. Auf der Cycle-Skala sind sie quasi identisch,
        die drei Achsen kollabieren auf die Diagonale → "lang-gestreckte
        Wolke" statt Limit-Cycle.

        Stattdessen: drei Lag-Indizes ÜBER DAS VOLLE FENSTER verteilt:
            j_x = 0           → log_res(t)
            j_y = M//2        → log_res(t - (M/2)·τ)   ≈ halbe Periode
            j_z = M - 1       → log_res(t - (M-1)·τ)   ≈ volle Periode

        Damit spannt das 3D-System genau ein Halving-Fenster auf und der
        Limit-Cycle wird visuell sichtbar.

    sigma_smooth:  optionale Glättung der drei Achsen-Signale für bessere
                   Visualisierung (sigma=60 wie SSM_res default). Setze 0
                   für die ungefilterte Variante.
    """
    from scipy.ndimage import gaussian_filter1d

    j_x, j_y, j_z = 0, M // 2, M - 1
    x = D[:, j_x]
    y = D[:, j_y]
    z = D[:, j_z]
    if sigma_smooth and sigma_smooth > 0:
        x = gaussian_filter1d(x, sigma=sigma_smooth)
        y = gaussian_filter1d(y, sigma=sigma_smooth)
        z = gaussian_filter1d(z, sigma=sigma_smooth)

    fig = plt.figure(figsize=(13.5, 8), num=fig_num)
    ax = fig.add_axes([-0.10, 0.03, 1.03, 0.96], projection='3d')
    ax.set_facecolor('#1A1A1A')
    ax.set_box_aspect((1.45, 1.0, 0.85))
    try:
        ax.dist = 7
    except Exception:
        pass

    # Coolwarm-cmap nach Zeit
    pts = np.array([x, y, z]).T
    seg = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
    t_norm = (days_vecs.astype(float) - float(days_vecs[0])) \
             / max(1.0, float(days_vecs[-1] - days_vecs[0]))
    line = Line3DCollection(seg, cmap='coolwarm', linewidth=1.6, alpha=0.85)
    line.set_array(t_norm[:-1])
    ax.add_collection3d(line)

    # Halving-Marker (lime)
    for i, hday in enumerate(halving_days):
        if hday < days_vecs[0] or hday > days_vecs[-1]:
            continue
        idx = int(np.argmin(np.abs(days_vecs - hday)))
        ax.scatter(x[idx], y[idx], z[idx], color='lime', s=55, zorder=12,
                   depthshade=False, edgecolors='black', linewidth=0.6)
        ax.text(x[idx], y[idx], z[idx], f'  H{i+1}',
                color='lime', fontsize=9, fontweight='bold')

    # Cycle-Tops
    for tday, tlabel in zip(cycle_top_days, cycle_top_labels):
        if tday < days_vecs[0] or tday > days_vecs[-1]:
            continue
        idx = int(np.argmin(np.abs(days_vecs - tday)))
        ax.scatter(x[idx], y[idx], z[idx], color='#FFAA33', s=90, zorder=15,
                   depthshade=False, edgecolors='white', linewidth=0.8)
        ax.text(x[idx], y[idx], z[idx], f'  {tlabel}',
                color='white', fontsize=10, fontweight='bold')

    ax.set_xlabel(rf'$X(t)$', fontsize=11, labelpad=8)
    ax.set_ylabel(rf'$X(t - {j_y}\tau)$', fontsize=11, labelpad=8)
    ax.set_zlabel(rf'$X(t - {j_z}\tau)$', fontsize=11, labelpad=8)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#444444')
    ax.tick_params(colors='#CCCCCC', labelsize=8)

    # Jahres-Colorbar
    cbar_ax = fig.add_axes([0.94, 0.20, 0.014, 0.62])
    cbar = fig.colorbar(line, cax=cbar_ax)
    n_ticks = 6
    tick_pos = np.linspace(0, 1, n_ticks)
    tick_labels = []
    for t in tick_pos:
        idx = int(np.argmin(np.abs(t_norm - t)))
        tick_labels.append(dates_vecs[idx].strftime('%Y'))
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels(tick_labels)

    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(
            f'Residual SSM (free) — pure TDE  '
            f'(no PCA, no fit, M={M}, τ={TAU}d)',
            color='#CCCCCC', fontsize=13, y=0.985,
            fontname='Comfortaa', fontweight='bold')
    return fig


# ── Fig 2: TWO-NN intrinsische Dimension ─────────────────────────────────────
def plot_intrinsic_dim(twonn_result, fig_num=12):
    """Zwei Panels:
    (a) Histogramm der μ-Werte (= r2/r1)
    (b) Linear-Fit log(1-F̂) vs log(μ) → Steigung -d
    """
    res = twonn_result
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), num=fig_num)

    # (a) Histogramm der μ
    ax1.hist(res.mu, bins=60, color='#4488FF', alpha=0.85, edgecolor='#2266CC')
    ax1.set_xlabel(r'$\mu_i = r_2 / r_1$', fontsize=11)
    ax1.set_ylabel('count', fontsize=11)
    ax1.set_title(rf'TWO-NN: $\mu$ distribution  ($N={res.n_used}$)',
                  fontsize=11, color='#E0E0E0')
    ax1.grid(True, alpha=0.4, linewidth=0.5)

    # (b) Linear-Fit log(1-F̂) vs log(μ)
    ax2.plot(res.cdf_x, res.cdf_y, color='#44FF88', linewidth=1.6,
             label=r'$-\log(1 - \hat F(\mu))$')
    # Fit-Linie
    x_fit = np.array([res.cdf_x.min(), res.cdf_x.max()])
    y_fit = res.d_intr * x_fit
    ax2.plot(x_fit, y_fit, color='#FFDD44', linestyle='--', linewidth=1.4,
             label=rf'fit slope $= d = {res.d_intr:.3f}$')
    ax2.set_xlabel(r'$\log \mu$', fontsize=11)
    ax2.set_ylabel(r'$-\log(1 - \hat F)$', fontsize=11)
    ax2.set_title(rf'TWO-NN linear fit  $\to d_{{\mathrm{{intr}}}} = '
                  rf'{res.d_intr:.3f} \pm {res.d_intr_std:.3f}$',
                  fontsize=11, color='#E0E0E0')
    ax2.grid(True, alpha=0.4, linewidth=0.5)
    ax2.legend(fontsize=9, facecolor='#1A1A1A', edgecolor='#808080',
               labelcolor='#E0E0E0', loc='upper left')

    with plt.rc_context({'text.usetex': False}):
        fig.suptitle('Residual SSM (free) — intrinsic dimension (TWO-NN, Facco et al. 2017)',
                     color='#CCCCCC', fontsize=13, y=0.985,
                     fontname='Comfortaa', fontweight='bold')
    plt.subplots_adjust(top=0.86, wspace=0.30, left=0.07, right=0.97, bottom=0.13)
    return fig


# ── Fig 3: Lokale Geometrie aus Local PCA ────────────────────────────────────
def plot_local_geometry(pc_for_traj, local_result, days_vecs, dates_vecs,
                        halving_days, cycle_top_days, cycle_top_labels,
                        master_idx=(0, 1), slaved_idx=2, fig_num=13):
    """Zwei Panels:
    (a) 3D-Trajektorie (PC1, PC2, PC3) farbcodiert mit lokaler 2D-ness
    (b) Histogramm der lokalen 2D-ness-Werte über alle Punkte

    Local 2D-ness = (λ1 + λ2) / Σλ aus der lokalen k-NN-PCA.
    Werte nahe 1.0 = Mannigfaltigkeit lokal flach (echtes 2D).
    Werte deutlich kleiner = lokal höherdimensional oder krümmung.

    pc_for_traj wird nur für die VISUELLE 3D-Lage der Trajektorie verwendet
    — die Local-PCA-Berechnung passiert auf dem rohen Embedding D.
    """
    i_u = master_idx[0]
    i_v = master_idx[1]
    i_w = slaved_idx
    two_d = local_result.two_d_ness

    fig = plt.figure(figsize=(15, 7), num=fig_num)

    # (a) 3D-Trajektorie + Local-2Dness als Farbe
    ax3d = fig.add_axes([-0.04, 0.05, 0.62, 0.92], projection='3d')
    ax3d.set_facecolor('#1A1A1A')
    ax3d.set_box_aspect((1.4, 1.0, 0.85))
    try:
        ax3d.dist = 7
    except Exception:
        pass

    pts = np.array([pc_for_traj[:, i_u],
                    pc_for_traj[:, i_v],
                    pc_for_traj[:, i_w]]).T
    seg = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
    line = Line3DCollection(seg, cmap='viridis', linewidth=2.5, alpha=0.9)
    line.set_array(two_d[:-1])
    line.set_clim(np.nanpercentile(two_d, 2), np.nanpercentile(two_d, 98))
    ax3d.add_collection3d(line)

    # Halving-Marker
    for i, hday in enumerate(halving_days):
        if hday < days_vecs[0] or hday > days_vecs[-1]:
            continue
        idx = int(np.argmin(np.abs(days_vecs - hday)))
        ax3d.scatter(pc_for_traj[idx, i_u], pc_for_traj[idx, i_v],
                     pc_for_traj[idx, i_w],
                     color='lime', s=55, zorder=12, depthshade=False,
                     edgecolors='black', linewidth=0.6)
        ax3d.text(pc_for_traj[idx, i_u], pc_for_traj[idx, i_v],
                  pc_for_traj[idx, i_w], f'  H{i+1}',
                  color='lime', fontsize=9, fontweight='bold')

    for tday, tlabel in zip(cycle_top_days, cycle_top_labels):
        if tday < days_vecs[0] or tday > days_vecs[-1]:
            continue
        idx = int(np.argmin(np.abs(days_vecs - tday)))
        ax3d.scatter(pc_for_traj[idx, i_u], pc_for_traj[idx, i_v],
                     pc_for_traj[idx, i_w],
                     color='#FFAA33', s=90, zorder=15, depthshade=False,
                     edgecolors='white', linewidth=0.8)
        ax3d.text(pc_for_traj[idx, i_u], pc_for_traj[idx, i_v],
                  pc_for_traj[idx, i_w], f'  {tlabel}',
                  color='white', fontsize=10, fontweight='bold')

    ax3d.set_xlabel(f'PC{i_u+1}', fontsize=10, labelpad=8)
    ax3d.set_ylabel(f'PC{i_v+1}', fontsize=10, labelpad=8)
    ax3d.set_zlabel(f'PC{i_w+1}', fontsize=10, labelpad=8)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#444444')
    ax3d.tick_params(colors='#CCCCCC', labelsize=8)

    # Colorbar
    cbar_ax = fig.add_axes([0.58, 0.20, 0.014, 0.60])
    cbar = fig.colorbar(line, cax=cbar_ax)
    cbar.set_label('local 2D-ness', fontsize=9)

    # (b) Histogramm der 2D-ness
    ax_h = fig.add_axes([0.66, 0.13, 0.31, 0.78])
    ax_h.hist(two_d[np.isfinite(two_d)], bins=40,
              color='#44FF88', alpha=0.85, edgecolor='#22AA55')
    mu_2d = float(np.nanmean(two_d))
    sd_2d = float(np.nanstd(two_d))
    ax_h.axvline(mu_2d, color='#FFDD44', linestyle='--', linewidth=1.4,
                 label=rf'mean $= {mu_2d:.3f}$')
    ax_h.set_xlabel('local 2D-ness  $(\\lambda_1 + \\lambda_2)/\\Sigma\\lambda$', fontsize=11)
    ax_h.set_ylabel('count', fontsize=11)
    ax_h.set_title(rf'Local PCA  ($k={local_result.k}$)  '
                   rf'mean $= {mu_2d:.3f} \pm {sd_2d:.3f}$',
                   fontsize=11, color='#E0E0E0')
    ax_h.grid(True, alpha=0.4, linewidth=0.5)
    ax_h.legend(fontsize=9, facecolor='#1A1A1A', edgecolor='#808080',
                labelcolor='#E0E0E0')
    ax_h.set_xlim(0, 1.02)

    with plt.rc_context({'text.usetex': False}):
        fig.suptitle('Residual SSM (free) — local geometry from k-NN PCA',
                     color='#CCCCCC', fontsize=13, y=0.985,
                     fontname='Comfortaa', fontweight='bold')
    return fig
