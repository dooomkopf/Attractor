"""SSM/res/plots_manifold.py — 3D-Manifold-Plot mit Cycle-Colors-Button.

Fig 3: Datenwolke (PC1, PC2, PC3) als graue Default-Trajektorie,
       überlagert mit der gefitteten Polynom-Surface PC3 = W₃(PC1, PC2)
       als halbtransparente Fläche.

Interaktivität:
       Button "Cycle-Colors" toggelt die Trajektorie zwischen
         (a) grauer Default-Linie  und
         (b) intrinsisch via θ_unwrap aus den Master-Koordinaten gefärbten
             Cycles (KEINE Halving-Bound-Vorgabe).
       Halvings + Cycle-Tops werden als externe Marker auf der Trajektorie
       eingezeichnet (lime Halving-Punkte, T1/T2/T3 in den Cycle-Farben).

Best-Practices (Codex Iter. 1):
    - shade=False, linewidth=0, edgecolor='none', alpha≈0.3 für Surface
    - depthshade=False für Scatter
    - 2-98%-Perzentile als Surface-Grid-Range (kein min/max-Overshoot)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.widgets import Button
import matplotlib  # noqa: F401 (für 3D-Backend)

from SSM_res_geometry import evaluate_surface


# Cycle-Farben (analog attractor_n.py:377)
_CYCLE_COLORS = ['grey', '#0000FF', '#90EE90', '#FF69B4', 'orange']
_CYCLE_LABELS = ['Cycle 0', 'Cycle 1', 'Cycle 2', 'Cycle 3', 'Cycle 4']


def _cycle_index_for(idx, bounds):
    """Welcher Cycle (0..n-1) enthält den Datenindex idx?"""
    for k in range(len(bounds) - 1):
        if bounds[k] <= idx <= bounds[k + 1]:
            return k
    return len(bounds) - 2


def plot_manifold(pca_res, pc_s, fit, days_vecs, dates_vecs,
                  halving_days, cycle_top_days, cycle_top_labels,
                  phase_result, master_idx=(0, 1), slaved_pos=0,
                  signal_label='log res', fig_num=3):
    """Fig 3 — 3D-Manifold-Plot.

    Args:
        pca_res:           PCAResult
        pc_s:              (N, M) geglättete PC-Trajektorie
        fit:               GeometryFit aus geometry.fit_geometry/scan_orders
        days_vecs:         Tag-Achse der Embedding-Vektoren
        dates_vecs:        datetime-Array für Jahres-Colorbar
        halving_days:      int-Array, Halving-Tage (volle Reihe)
        cycle_top_days:    int-Array der Cycle-Top-Tage (T0..T3)
        cycle_top_labels:  list der Top-Labels ('T0', 'T1', ...)
        phase_result:      PhaseResult aus phase.compute_phase_full
                           — liefert die intrinsischen Cycle-Bounds
        master_idx:        2-Tupel der master-PC-Spalten (default (0, 1))
        slaved_pos:        Index der slaved PC innerhalb fit.slaved_idx
        signal_label:      Beschreibung des Signals (für Titel)
    """
    pc = pca_res.pc
    var = pca_res.var

    i_u = master_idx[0]
    i_v = master_idx[1]
    i_w = int(fit.slaved_idx[slaved_pos])

    # ── Surface Grid in skalierten Koords (2-98 %ile, kein min/max) ─────────
    u_data = pc[:, i_u] / fit.u_std
    v_data = pc[:, i_v] / fit.v_std
    u_lo, u_hi = np.percentile(u_data, [2, 98])
    v_lo, v_hi = np.percentile(v_data, [2, 98])
    n_grid = 50
    u_grid = np.linspace(u_lo, u_hi, n_grid)
    v_grid = np.linspace(v_lo, v_hi, n_grid)
    Z = evaluate_surface(fit, u_grid, v_grid, slaved_pos=slaved_pos)
    UU, VV = np.meshgrid(u_grid * fit.u_std, v_grid * fit.v_std)

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13.5, 8), num=fig_num)
    # 3D-Achse mit negativem left: schiebt die Bounding-Box über den linken
    # Figure-Rand hinaus, damit der gerenderte 3D-Inhalt (der von mplot3d
    # mit internem Padding versehen ist) wirklich am linken Rand beginnt.
    ax = fig.add_axes([-0.10, 0.03, 1.03, 0.96], projection='3d')
    ax.set_facecolor('#1A1A1A')
    ax.set_box_aspect((1.45, 1.0, 0.85))
    try:
        ax.dist = 7  # Kamera näher (Default 10) → Plot füllt mehr Fläche
    except Exception:
        pass

    # 1) Default-Trajektorie: coolwarm-cmap, von alt (blau) zu neu (rot).
    #    Wie attractor.py:282-285. Die geglätteten pc_s werden verwendet.
    pts = np.array([pc_s[:, i_u], pc_s[:, i_v], pc_s[:, i_w]]).T
    seg = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
    t_norm = (days_vecs.astype(float) - float(days_vecs[0])) \
             / max(1.0, float(days_vecs[-1] - days_vecs[0]))
    line_default = Line3DCollection(seg, cmap='coolwarm', linewidth=2.5,
                                    alpha=0.92)
    line_default.set_array(t_norm[:-1])
    ax.add_collection3d(line_default)

    # 2) Cycle-Linien (intrinsisch via phase_result.bounds, initial unsichtbar)
    bounds = phase_result.bounds
    n_segs = max(0, len(bounds) - 1)
    cycle_lines = []
    for k in range(n_segs):
        s0 = int(bounds[k])
        s1 = min(int(bounds[k + 1]) + 1, len(pc_s))
        if s1 - s0 < 2:
            continue
        col = _CYCLE_COLORS[k] if k < len(_CYCLE_COLORS) else 'white'
        lbl = _CYCLE_LABELS[k] if k < len(_CYCLE_LABELS) else f'Cycle {k}'
        ln, = ax.plot(
            pc_s[s0:s1, i_u], pc_s[s0:s1, i_v], pc_s[s0:s1, i_w],
            color=col, linewidth=2.5, alpha=0.95,
            visible=False, label=lbl,
        )
        cycle_lines.append(ln)

    # 3) Polynom-Surface (transparent, helle Facecolor)
    ax.plot_surface(
        UU, VV, Z,
        color='#FFEEDD', alpha=0.18,
        shade=False, linewidth=0, edgecolor='none',
        rcount=n_grid, ccount=n_grid,
    )

    # 4) Externe Marker — Halvings (lime) + Cycle-Tops (T1..T3) ──────────────
    halving_handle = None
    for i, hday in enumerate(halving_days):
        # Halving im Embedding-Bereich vorhanden?
        if hday < days_vecs[0] or hday > days_vecs[-1]:
            continue
        idx = int(np.argmin(np.abs(days_vecs - hday)))
        sc = ax.scatter(pc_s[idx, i_u], pc_s[idx, i_v], pc_s[idx, i_w],
                        color='lime', s=60, zorder=12, depthshade=False,
                        edgecolors='black', linewidth=0.6,
                        label='Halving' if halving_handle is None else None)
        if halving_handle is None:
            halving_handle = sc
        ax.text(pc_s[idx, i_u], pc_s[idx, i_v], pc_s[idx, i_w], f'  H{i+1}',
                color='lime', fontsize=9, fontweight='bold')

    # T-Marker (T0..T3) — Farbe ist die des intrinsischen Cycles, in dem T liegt
    for tday, tlabel in zip(cycle_top_days, cycle_top_labels):
        if tday < days_vecs[0] or tday > days_vecs[-1]:
            continue
        idx = int(np.argmin(np.abs(days_vecs - tday)))
        cyc_k = _cycle_index_for(idx, bounds)
        col = _CYCLE_COLORS[cyc_k] if 0 <= cyc_k < len(_CYCLE_COLORS) else 'red'
        ax.scatter(pc_s[idx, i_u], pc_s[idx, i_v], pc_s[idx, i_w],
                   color=col, s=110, zorder=15, depthshade=False,
                   edgecolors='white', linewidth=1.0)
        ax.text(pc_s[idx, i_u], pc_s[idx, i_v], pc_s[idx, i_w], f'  {tlabel}',
                color='white', fontsize=10, fontweight='bold')

    # ── Achsen / Title ──────────────────────────────────────────────────────
    ax.set_xlabel(f'PC{i_u+1}  ({var[i_u]*100:.1f}\\%)', fontsize=10, labelpad=8)
    ax.set_ylabel(f'PC{i_v+1}  ({var[i_v]*100:.1f}\\%)', fontsize=10, labelpad=8)
    ax.set_zlabel(f'PC{i_w+1}  ({var[i_w]*100:.1f}\\%)', fontsize=10, labelpad=8)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#444444')
    ax.tick_params(colors='#CCCCCC', labelsize=8)

    # Parameter-Subtitle (order, R², cond, rank, intrinsic cycles) bewusst
    # weggelassen — auf User-Wunsch keine Achsen-Tags. Bei Bedarf später
    # wieder aktivieren:
    # cond_tag = f'cond={fit.cond:.1e}' if np.isfinite(fit.cond) else 'cond=∞'
    # n_intrinsic_cycles = max(0, len(bounds) - 1)
    # ax.set_title(
    #     f'order {fit.order} $\\;|\\;$ $R^2={fit.R2_total:.3f}$ $\\;|\\;$ '
    #     f'{cond_tag} $\\;|\\;$ rank={fit.rank}/{fit.B.shape[0]} $\\;|\\;$ '
    #     f'{n_intrinsic_cycles} intrinsic cycles',
    #     fontsize=10, color='#E0E0E0', pad=14)

    # Jahres-Colorbar in eigener Achse rechts (kein Label, Ticks=Jahre)
    cbar_ax = fig.add_axes([0.94, 0.20, 0.014, 0.62])
    cbar = fig.colorbar(line_default, cax=cbar_ax)
    n_ticks = 6
    tick_pos = np.linspace(0, 1, n_ticks)
    tick_labels = []
    for t in tick_pos:
        idx = int(np.argmin(np.abs(t_norm - t)))
        tick_labels.append(dates_vecs[idx].strftime('%Y'))
    cbar.set_ticks(tick_pos)
    cbar.set_ticklabels(tick_labels)

    # ── Cycle-Colors-Button — mittig unter der 3D-Achse ───────────────────
    # 3D-Achse hat x-Range [-0.02, 0.91], Mitte = 0.445.
    # Button-Breite 0.14 → x_left = 0.445 - 0.07 = 0.375.
    ax_btn = fig.add_axes([0.375, 0.015, 0.14, 0.038])
    btn_cyc = Button(ax_btn, 'Cycle-Colors',
                     color='#333333', hovercolor='#555555')
    btn_cyc.label.set_color('#CCCCCC')
    cyc_active = [False]

    def _toggle_cyc(event):
        cyc_active[0] = not cyc_active[0]
        line_default.set_visible(not cyc_active[0])
        for ln in cycle_lines:
            ln.set_visible(cyc_active[0])
        fig.canvas.draw_idle()
    btn_cyc.on_clicked(_toggle_cyc)
    # Persistente Referenz, sonst GC
    fig._btn_cyc = btn_cyc  # noqa: SLF001

    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(
            f'Residual SSM — manifold geometry  '
            f'(slaved PC{i_w+1} = W(PC{i_u+1}, PC{i_v+1}))',
            color='#CCCCCC', fontsize=13, y=0.985,
            fontname='Comfortaa', fontweight='bold')
    # Layout: ax und ax_btn werden manuell via add_axes positioniert,
    # subplots_adjust würde damit kollidieren → bewusst weggelassen.
    return fig
