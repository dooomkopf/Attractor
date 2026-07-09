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
from matplotlib.lines import Line2D
import matplotlib  # noqa: F401 (für 3D-Backend)

from SSM_res_geometry import evaluate_surface


# Halving-Cycle-Farben aus gold/cycles.json halving_periods (Konvention wie in
# attractor_cy.py:111-126). Index k = Halving-Cycle:
#   0 = pre-1st (grey), 1 = '13, 2 = '17, 3 = '21, 4 = '25
# Verwendung: searchsorted(halving_days, day, side='right') liefert k.
_HALVING_CYCLE_COLORS = ['grey', '#0000FF', '#90EE90', '#FF69B4', 'orange']
_HALVING_CYCLE_LABELS = ['pre-Halving', "'13 Cycle", "'17 Cycle", "'21 Cycle", "'25 Cycle"]


def _cycle_index_for(idx, bounds):
    """Welcher Cycle (0..n-1) enthält den Datenindex idx?"""
    for k in range(len(bounds) - 1):
        if bounds[k] <= idx <= bounds[k + 1]:
            return k
    return len(bounds) - 2


def plot_manifold(pca_res, pc_s, fit, days_vecs, dates_vecs,
                  halving_days, cycle_top_days, cycle_top_labels,
                  phase_result, master_idx=(0, 1), slaved_pos=0,
                  signal_label='log res', fig_num=3,
                  ssm_dim=None, time_mode=None):
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
    fig.patch.set_facecolor('#1A1A1A')   # gleicher Hintergrund wie Achse → kein Rand
    # 3D-Achse mit negativem left: schiebt die Bounding-Box über den linken
    # Figure-Rand hinaus, damit der gerenderte 3D-Inhalt (der von mplot3d
    # mit internem Padding versehen ist) wirklich am linken Rand beginnt.
    # top = 0.91 → Platz für Suptitle (bei y≈0.96), keine Überlappung mehr.
    ax = fig.add_axes([-0.10, 0.05, 1.03, 0.86], projection='3d')
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

    # 2) Cycle-Linien — Tag-basiert nach Halving-Cycle (Konvention aus
    #    gold/cycles.json, analog attractor_cy.py:111-126). Pro durchgängigem
    #    Halving-Cycle-Segment eine Linie mit zugehöriger Farbe + Label.
    hv_idx_per_pt = np.searchsorted(halving_days, days_vecs, side='right')
    cycle_lines = []
    seen_labels = set()
    i_start = 0
    n_pts = len(days_vecs)
    for i in range(1, n_pts + 1):
        if i == n_pts or hv_idx_per_pt[i] != hv_idx_per_pt[i_start]:
            hv_k = int(hv_idx_per_pt[i_start])
            col = _HALVING_CYCLE_COLORS[hv_k] if 0 <= hv_k < len(_HALVING_CYCLE_COLORS) else 'white'
            lbl_raw = _HALVING_CYCLE_LABELS[hv_k] if 0 <= hv_k < len(_HALVING_CYCLE_LABELS) else f'Cycle {hv_k}'
            lbl = lbl_raw if lbl_raw not in seen_labels else None
            seen_labels.add(lbl_raw)
            s1 = min(i + 1, n_pts)
            ln, = ax.plot(
                pc_s[i_start:s1, i_u], pc_s[i_start:s1, i_v], pc_s[i_start:s1, i_w],
                color=col, linewidth=2.5, alpha=0.95,
                visible=False, label=lbl,
            )
            cycle_lines.append(ln)
            i_start = i

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

    # T-Marker (T1..T4) — Farbe ist die des Halving-Cycles (aus gold/cycles.json),
    # in dem T liegt. searchsorted(halving_days, tday, side='right'):
    #   0 = vor H1 (pre-1st), 1 = H1..H2 ('13), 2 = H2..H3 ('17), ...
    for tday, tlabel in zip(cycle_top_days, cycle_top_labels):
        if tday < days_vecs[0] or tday > days_vecs[-1]:
            continue
        idx = int(np.argmin(np.abs(days_vecs - tday)))
        hv_k = int(np.searchsorted(halving_days, tday, side='right'))
        col = _HALVING_CYCLE_COLORS[hv_k] if 0 <= hv_k < len(_HALVING_CYCLE_COLORS) else 'red'
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
    cbar_ax = fig.add_axes([0.86, 0.20, 0.014, 0.62])
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

    # Legende mit Line2D-Proxies (3D-Axes-Legende zeigt sonst keine Farbsymbole).
    # Nur die Halving-Cycles aufnehmen, die tatsächlich im Plot vorkommen.
    _unique_hv = sorted({int(x) for x in hv_idx_per_pt})
    _legend_handles = [
        Line2D([0], [0], color=_HALVING_CYCLE_COLORS[k], lw=3,
               label=_HALVING_CYCLE_LABELS[k])
        for k in _unique_hv
        if 0 <= k < len(_HALVING_CYCLE_COLORS)
    ]
    cyc_legend = ax.legend(handles=_legend_handles, loc='upper left',
                           facecolor='#1A1A1A', edgecolor='#444444',
                           labelcolor='#CCCCCC', fontsize=9, framealpha=0.85)
    cyc_legend.set_visible(False)

    def _toggle_cyc(event):
        cyc_active[0] = not cyc_active[0]
        line_default.set_visible(not cyc_active[0])
        for ln in cycle_lines:
            ln.set_visible(cyc_active[0])
        cyc_legend.set_visible(cyc_active[0])
        fig.canvas.draw_idle()
    btn_cyc.on_clicked(_toggle_cyc)
    # Persistente Referenz, sonst GC
    fig._btn_cyc = btn_cyc  # noqa: SLF001

    _title_parts = ['Residual SSM — manifold geometry']
    if ssm_dim is not None:
        _title_parts.append(f'ssm_dim={ssm_dim}')
    if time_mode is not None:
        _title_parts.append(f'time_mode={time_mode}')
    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(
            '  |  '.join(_title_parts),
            color='#CCCCCC', fontsize=13, y=0.96,
            fontname='Comfortaa', fontweight='bold')
    # Layout: ax und ax_btn werden manuell via add_axes positioniert,
    # subplots_adjust würde damit kollidieren → bewusst weggelassen.
    return fig
