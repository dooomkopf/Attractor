"""attractor_controls.py — Button/Slider-Widgets für attractor_n_ens.py"""

import numpy as np
from matplotlib.widgets import Slider, Button


# ── Hilfsfunktion ─────────────────────────────────────────────────────────────
def _make_btn(fig, pos, label, color='#333333', hover='#555555', fontsize=8):
    ax = fig.add_axes(pos)
    btn = Button(ax, label, color=color, hovercolor=hover)
    btn.label.set_color('#CCCCCC')
    btn.label.set_fontsize(fontsize)
    return btn


# ── fig_main: Cycle-Colors ────────────────────────────────────────────────────
def add_cycle_button_fig1(fig, pcyc_lines, cyc_lines):
    """Cycle-Colors Toggle für fig_main (3D + ax1).

    pcyc_lines : sichtbar wenn NICHT aktiv  (p-Zyklus RGB)
    cyc_lines  : sichtbar wenn aktiv         (Halving-Farben)
    """
    btn = _make_btn(fig, [0.01, 0.02, 0.08, 0.025], 'Cycle-Colors')
    _active = [False]

    def _toggle(event):
        _active[0] = not _active[0]
        for _l in pcyc_lines:
            _l.set_visible(not _active[0])
        for _l in cyc_lines:
            _l.set_visible(_active[0])
        fig.canvas.draw_idle()

    btn.on_clicked(_toggle)
    return btn


# ── fig_main: Preset View-Buttons ─────────────────────────────────────────────
def add_view_buttons(fig, ax2):
    """PC1-PC2 / PC1-PC3 / PC2-PC3 Preset-Views."""
    positions = [
        [0.01, 0.055, 0.08, 0.022],
        [0.01, 0.082, 0.08, 0.022],
        [0.01, 0.109, 0.08, 0.022],
    ]
    params = [
        ('PC1-PC2', 90,  0),
        ('PC1-PC3',  0,  0),
        ('PC2-PC3',  0, 90),
    ]
    btns = []
    for pos, (lbl, elev, azim) in zip(positions, params):
        btn = _make_btn(fig, pos, lbl)

        def _mk(el, az):
            def _cb(event):
                ax2.view_init(elev=el, azim=az)
                fig.canvas.draw_idle()
            return _cb

        btn.on_clicked(_mk(elev, azim))
        btns.append(btn)
    return btns


# ── fig_main: Custom View-Buttons ─────────────────────────────────────────────
def add_custom_view_buttons(fig, ax2):
    """Custom1/2/3: Rechtsklick=speichern, Linksklick=abrufen."""
    positions = [
        [0.10, 0.055, 0.08, 0.022],
        [0.10, 0.082, 0.08, 0.022],
        [0.10, 0.109, 0.08, 0.022],
    ]
    _views = [None, None, None]
    _axes  = []
    btns   = []
    for i, pos in enumerate(positions):
        btn = _make_btn(fig, pos, f'Custom{i+1}', color='#2a2a4a', hover='#3a3a6a')
        btns.append(btn)
        _axes.append(btn.ax)

    def _click(event):
        for i, _ax in enumerate(_axes):
            if event.inaxes == _ax:
                if event.button == 3:                        # Rechtsklick → speichern
                    _views[i] = (ax2.elev, ax2.azim)
                elif event.button == 1 and _views[i] is not None:  # Linksklick → abrufen
                    ax2.view_init(elev=_views[i][0], azim=_views[i][1])
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', _click)
    return btns


# ── fig_main: Rotate + Elevation-Slider ───────────────────────────────────────
def add_rotate_controls(fig, ax2):
    """Rotate-Toggle + Elevation-Slider (0–40°), Label oben mittig."""
    btn_rot = _make_btn(fig, [0.195, 0.082, 0.055, 0.022], 'Rotate',
                        color='#1a3a1a', hover='#2a5a2a')

    ax_elev = fig.add_axes([0.26, 0.075, 0.16, 0.022])
    slider_elev = Slider(ax_elev, 'Elev', 0, 40, valinit=30, color='#444444')
    slider_elev.valtext.set_color('#CCCCCC')
    # Label oben mittig
    slider_elev.label.set_text('Elevation')
    slider_elev.label.set_position((0.5, 1.15))
    slider_elev.label.set_transform(ax_elev.transAxes)
    slider_elev.label.set_ha('center')
    slider_elev.label.set_va('bottom')
    slider_elev.label.set_color('#CCCCCC')
    slider_elev.label.set_fontsize(8)

    _timer   = fig.canvas.new_timer(interval=16)   # ~60 fps
    _running = [False]

    def _step():
        ax2.azim = (ax2.azim + 0.33) % 360
        fig.canvas.draw_idle()

    _timer.add_callback(_step)

    def _toggle(event):
        if _running[0]:
            _timer.stop()
            _running[0] = False
        else:
            _timer.start()
            _running[0] = True

    btn_rot.on_clicked(_toggle)

    def _elev_update(val):
        ax2.view_init(elev=slider_elev.val, azim=ax2.azim)
        fig.canvas.draw_idle()

    slider_elev.on_changed(_elev_update)
    return btn_rot, slider_elev


# ── fig_main: Tag-Slider ──────────────────────────────────────────────────────
def add_tag_slider(fig, ax_sl, days_start, days_end, days_mid,
                   ax1, ax2, days_n_all, daily_n_all, pc_s, days_vecs, label_window):
    """Tag-Slider mit gelbem Highlight in ax1 und ax2."""
    slider = Slider(ax_sl, 'Tag', days_start, days_end, valinit=days_mid, color='#666666')
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
        mask1 = (days_n_all >= d - label_window) & (days_n_all <= d + label_window)
        if mask1.any():
            hl['ax1_scat'] = ax1.scatter(days_n_all[mask1], daily_n_all[mask1],
                                          color='#FFFF00', s=80, alpha=1.0, zorder=9, linewidths=0)
        mask2 = (days_vecs >= d - label_window) & (days_vecs <= d + label_window)
        if mask2.any():
            hl['ax2_scat'] = ax2.scatter(pc_s[mask2, 0], pc_s[mask2, 1], pc_s[mask2, 2],
                                          color='#FFFF00', s=120, alpha=1.0, zorder=10)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(days_mid)
    return slider


# ── fig3: Cycle-Colors ────────────────────────────────────────────────────────
def add_cycle_button_fig3(fig, pcyc_lines_pol, pcyc_lines_phs,
                           halv_lines_pol, halv_lines_phs):
    """Cycle-Colors Toggle für fig3 (Polar + Phase)."""
    btn = _make_btn(fig, [0.01, 0.01, 0.12, 0.03], 'Cycle-Colors')
    _active = [False]

    def _toggle(event):
        _active[0] = not _active[0]
        for _l in pcyc_lines_pol + pcyc_lines_phs:
            _l.set_visible(not _active[0])
        for _l in halv_lines_pol + halv_lines_phs:
            _l.set_visible(_active[0])
        fig.canvas.draw_idle()

    btn.on_clicked(_toggle)
    return btn
