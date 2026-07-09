"""attractor_n_ens/plot1.py — Figure 1: Ensemble-n (ax1) + 3D Phase Space (ax2)"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter1d

from .attractor_n_ens_utils import make_segments, make_segments3d
from .attractor_n_ens_controls import (add_cycle_button_fig1, add_view_buttons,
                        add_custom_view_buttons, add_rotate_controls,
                        add_tag_slider, add_slider_marker_button,
                        _make_cycle_legend_handles)

_CYC_LEG_KW = dict(fontsize=8, framealpha=0.8, facecolor=(0.102, 0.102, 0.102, 0.8),
                   edgecolor='#808080', labelcolor='#E0E0E0')


def plot_fig1(data, params):
    START_IDX    = params['START_IDX']
    CMAP         = params['CMAP']
    LABEL_WINDOW = params['LABEL_WINDOW']

    days_n_all     = data['days_n_all']
    daily_n_all    = data['daily_n_all']
    t_norm_n_all   = data['t_norm_n_all']
    yr_labels      = data['yr_labels']
    halving_days   = data['halving_days']
    _x_max_fixed   = data['_x_max_fixed']
    used_peak_days = data['used_peak_days']
    peak_labels    = data['peak_labels']
    peak_colors    = data['peak_colors']
    peak_n_vals    = data['peak_n_vals']
    used_bottom_days = data['used_bottom_days']
    bottom_labels  = data['bottom_labels']
    bottom_colors  = data['bottom_colors']
    bottom_n_vals  = data['bottom_n_vals']
    pc_s           = data['pc_s']
    var            = data['var']
    cum3           = data['cum3']
    days_vecs      = data['days_vecs']
    _pcyc_idxs     = data['_pcyc_idxs']

    fig_main = plt.figure(figsize=(22, 10))
    fig_main.patch.set_facecolor('black')

    ax1   = fig_main.add_axes([0.04, 0.30, 0.42, 0.43])
    ax2   = fig_main.add_axes([0.50, 0.06, 0.50, 0.90], projection='3d', elev=30, azim=156)
    ax_sl = fig_main.add_axes([0.10, 0.02, 0.80, 0.025])
    ax_sl.set_facecolor('#2A2A2A')

    # ── ax1: Ensemble-n ───────────────────────────────────────────────────────
    ax1.set_facecolor('#1A1A1A')
    mask_plot = days_n_all >= START_IDX
    ax1.autoscale()
    ax1.axhline(y=0, color='#808080', linestyle='--', alpha=0.5, linewidth=1.0)

    for hday in halving_days:
        ax1.axvline(x=hday, color='grey', linestyle='--', alpha=0.4, linewidth=0.8, zorder=1)

    _pcyc_day_starts_ax1 = [days_vecs[_pidx] for _pidx in _pcyc_idxs]
    for _pnum, _pday in enumerate(_pcyc_day_starts_ax1):
        ax1.axvline(x=_pday, color='red', linestyle='-', alpha=0.6, linewidth=0.8, zorder=1)
        ax1.text(_pday, 51, f'PCA{_pnum+1}', color='red', fontsize=7,
                 ha='right', va='top', fontweight='bold', alpha=0.85, zorder=10)

    _mean_n = np.nanmean(daily_n_all[mask_plot])
    ax1.axhline(y=_mean_n, color='#FF9999', linestyle='-', alpha=0.8, linewidth=1.0,
                zorder=2, label='Mean')

    for i, hday in enumerate(halving_days):
        hidx_n = np.argmin(np.abs(days_n_all - hday))
        ax1.scatter(days_n_all[hidx_n], daily_n_all[hidx_n],
                    color='grey', s=60, zorder=10, edgecolors='white', linewidths=1.0,
                    label='Halving' if i == 0 else '_')
        ax1.annotate(f'H{i+1}', (days_n_all[hidx_n], daily_n_all[hidx_n]),
                     textcoords='offset points', xytext=(6, 5),
                     color='white', fontsize=9, fontweight='bold')
        if START_IDX <= hday <= _x_max_fixed:
            ax1.text(hday, -56, f'H{i+1}', color='grey', fontsize=8,
                     ha='left', va='bottom', fontweight='bold', alpha=0.85, zorder=10)
    _h5_est = halving_days[-1] + 1460
    if _h5_est <= _x_max_fixed:
        ax1.axvline(x=_h5_est, color='grey', linestyle='--', alpha=0.4, linewidth=0.8, zorder=1)
        ax1.text(_h5_est, 56, 'H5 (est.)', color='grey', fontsize=8,
                 ha='right', va='top', fontweight='bold', alpha=0.85, zorder=10)

    # markers are placed on _sm_all (smooth curve) — computed below after _sm_all

    ax1.set_ylim(-60, 60)
    ax1.set_xlim(START_IDX, _x_max_fixed)

    _cyc_colors       = ['grey', '#0000FF', '#90EE90', '#FF69B4', 'orange']
    _pcyc_colors_ax1  = ['#FF4444', '#44FF44', '#4488FF', '#FFD700', '#FF44FF']
    _bounds           = np.concatenate([[days_n_all[0]], halving_days, [days_n_all[-1]]])
    _mask_all         = np.isfinite(daily_n_all) & (days_n_all >= START_IDX)
    _sm_all           = np.full_like(daily_n_all, np.nan)
    _sm_all[_mask_all]= gaussian_filter1d(daily_n_all[_mask_all], sigma=60)

    _last_fin_sm = np.where(np.isfinite(_sm_all))[0][-1]
    ax1.scatter(days_n_all[_last_fin_sm], _sm_all[_last_fin_sm], facecolors='white', s=30,
                zorder=10, edgecolors='black', linewidths=1.5, label='Latest Data')

    _tb_artists_ax1 = []
    for px, lbl, col in zip(used_peak_days, peak_labels, peak_colors):
        pidx = np.argmin(np.abs(days_n_all - px))
        py = _sm_all[pidx]
        if not np.isfinite(py):
            continue
        _sc = ax1.scatter(px, py, color=col, s=60, zorder=8, alpha=0.9,
                    edgecolors='white', linewidths=1.0, label='_')
        _sc.set_visible(False)
        _disp_lbl = f'{lbl} (prelim.)' if lbl == 'T4' else lbl
        _ann = ax1.annotate(_disp_lbl, (px, py), textcoords='offset points',
                     xytext=(6, 6), color=col, fontsize=9, fontweight='bold')
        _ann.set_visible(False)
        _tb_artists_ax1.extend([_sc, _ann])

    for bx, lbl, col in zip(used_bottom_days, bottom_labels, bottom_colors):
        bidx = np.argmin(np.abs(days_n_all - bx))
        by = _sm_all[bidx]
        if not np.isfinite(by):
            continue
        _sc = ax1.scatter(bx, by, color=col, marker='s', s=60, zorder=8, alpha=0.9,
                    edgecolors='white', linewidths=1.0, label='_')
        _sc.set_visible(False)
        _ann = ax1.annotate(lbl, (bx, by), textcoords='offset points',
                     xytext=(6, -12), color=col, fontsize=9, fontweight='bold')
        _ann.set_visible(False)
        _tb_artists_ax1.extend([_sc, _ann])

    _halv_lines_ax1 = []
    for k in range(len(_bounds) - 1):
        _d0, _d1 = _bounds[k], _bounds[k + 1]
        _col = _cyc_colors[k] if k < len(_cyc_colors) else 'white'
        _mc  = (days_n_all >= max(_d0, START_IDX)) & (days_n_all <= _d1)
        _ln, = ax1.plot(days_n_all[_mc], _sm_all[_mc], color=_col,
                        linewidth=2.5, alpha=0.7, zorder=5, visible=False)
        _halv_lines_ax1.append(_ln)

    _pcyc_day_starts  = [days_vecs[_pidx] for _pidx in _pcyc_idxs]
    _pre_h2_day = _pcyc_day_starts[0] if _pcyc_day_starts else days_n_all[-1]
    _mc_pre = (days_n_all >= START_IDX) & (days_n_all <= _pre_h2_day) & np.isfinite(_sm_all)
    if _mc_pre.sum() > 1:
        ax1.plot(days_n_all[_mc_pre], _sm_all[_mc_pre],
                 color='grey', linewidth=2.5, alpha=0.7, zorder=5)

    _pcyc_bounds_ax1  = list(_pcyc_day_starts) + [days_n_all[-1]]
    _pcyc_lines_ax1   = []
    for _ci in range(len(_pcyc_idxs)):
        _d0, _d1 = _pcyc_bounds_ax1[_ci], _pcyc_bounds_ax1[_ci + 1]
        _mc = (days_n_all >= max(_d0, START_IDX)) & (days_n_all <= _d1) & np.isfinite(_sm_all)
        if _mc.sum() > 1:
            _ln, = ax1.plot(days_n_all[_mc], _sm_all[_mc],
                            color=_pcyc_colors_ax1[_ci % len(_pcyc_colors_ax1)],
                            linewidth=2.5, alpha=0.7, zorder=5)
            _pcyc_lines_ax1.append(_ln)

    _cb_dummy = plt.cm.ScalarMappable(cmap=CMAP)
    _cb_dummy.set_clim(0, 1)
    _cb_dummy.set_array([])
    cb1 = fig_main.colorbar(_cb_dummy, ax=ax1, pad=0.01)
    cb1.set_label('Jahr', color='#CCCCCC')
    cb1.set_ticks(np.linspace(0, 1, 6))
    cb1.set_ticklabels(yr_labels)

    with plt.rc_context({'text.usetex': False}):
        ax1.set_ylabel('Ensemble n', color='#E0E0E0')
        ax1.set_title('BTC Ensemble Power-Law-Exponent', color='#CCCCCC',
                      fontsize=12, fontname='Comfortaa', fontweight='bold')
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.5)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.8,
               facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#E0E0E0')
    _leg_ax1_main = ax1.get_legend()
    _leg_ax1_main.get_frame().set_alpha(0.8)
    for _h in _leg_ax1_main.legendHandles: _h.set_alpha(1.0)
    _leg_ax1_cyc  = ax1.legend(handles=_make_cycle_legend_handles(), loc='lower left', **_CYC_LEG_KW)
    _leg_ax1_cyc.get_frame().set_alpha(0.8)
    _leg_ax1_cyc.set_visible(False)
    ax1.add_artist(_leg_ax1_main)

    # ── ax2: 3D Phase Space ───────────────────────────────────────────────────
    _cyc_col3d     = ['#FF4444', '#44FF44', '#4488FF', '#FFD700', '#FF44FF']
    _pcyc_lines_3d = []
    _h2_i3d = _pcyc_idxs[0] if _pcyc_idxs else 0
    if _h2_i3d > 1:
        _segs3d_pre = make_segments3d(pc_s[0:_h2_i3d+1, 0], pc_s[0:_h2_i3d+1, 1], pc_s[0:_h2_i3d+1, 2])
        ax2.add_collection3d(Line3DCollection(_segs3d_pre, color='grey', linewidth=2.5, alpha=0.5))
    _cyc_bounds_3d = list(_pcyc_idxs) + [len(pc_s)]
    for _ci in range(len(_pcyc_idxs)):
        _s, _e = _cyc_bounds_3d[_ci], _cyc_bounds_3d[_ci + 1]
        if _e > _s + 1:
            _segs3d = make_segments3d(pc_s[_s:_e+1, 0], pc_s[_s:_e+1, 1], pc_s[_s:_e+1, 2])
            _lc3d   = Line3DCollection(_segs3d, color=_cyc_col3d[_ci % len(_cyc_col3d)],
                                       linewidth=2.5, alpha=0.9)
            ax2.add_collection3d(_lc3d)
            _pcyc_lines_3d.append(_lc3d)

    for arr, setter in [(pc_s[:, 0], ax2.set_xlim),
                        (pc_s[:, 1], ax2.set_ylim),
                        (pc_s[:, 2], ax2.set_zlim)]:
        pad = (arr.max() - arr.min()) * 0.05
        setter(arr.min() - pad, arr.max() + pad)

    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_vecs - hday))
        if np.abs(days_vecs[hidx] - hday) < 200:
            ax2.scatter(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                        color='grey', s=60, zorder=10, edgecolors='white', linewidths=1.0,
                        label='Halving' if i == 0 else '_')
            ax2.text(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                     f'H{i+1}', color='white', fontsize=9, fontweight='bold',
                     ha='right', va='bottom')

    _tb_artists_ax2 = []
    for pd_, lbl, col in zip(used_peak_days, peak_labels, peak_colors):
        hidx = np.argmin(np.abs(days_vecs - pd_))
        if np.abs(days_vecs[hidx] - pd_) < 200:
            _sc = ax2.scatter(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                        color=col, s=60, zorder=10, edgecolors='white', linewidths=1.0,
                        label='_')
            _sc.set_visible(False)
            _disp_lbl = f' {lbl} (prelim.)' if lbl == 'T4' else f' {lbl}'
            _txt = ax2.text(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                     _disp_lbl, color=col, fontsize=9, fontweight='bold')
            _txt.set_visible(False)
            _tb_artists_ax2.extend([_sc, _txt])

    for i, (bd_, lbl, col) in enumerate(zip(used_bottom_days, bottom_labels, bottom_colors)):
        if not np.isfinite(bottom_n_vals[i]):
            continue
        hidx = np.argmin(np.abs(days_vecs - bd_))
        if np.abs(days_vecs[hidx] - bd_) < 200:
            _sc = ax2.scatter(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                        color=col, marker='s', s=60, zorder=10,
                        edgecolors='white', linewidths=1.0,
                        label='_')
            _sc.set_visible(False)
            _txt = ax2.text(pc_s[hidx, 0], pc_s[hidx, 1], pc_s[hidx, 2],
                     f' {lbl}', color=col, fontsize=9, fontweight='bold')
            _txt.set_visible(False)
            _tb_artists_ax2.extend([_sc, _txt])

    ax2.scatter(*[pc_s[-1, i] for i in range(3)],
                facecolors='white', s=30, zorder=10, edgecolors='black', linewidths=1.5, label='Latest Data')
    for _pn, _pidx in enumerate(_pcyc_idxs):
        ax2.scatter(pc_s[_pidx, 0], pc_s[_pidx, 1], pc_s[_pidx, 2],
                    color='red', s=80, zorder=11, marker='^',
                    edgecolors='white', linewidths=1.2,
                    label='PCA-Cycle-Start' if _pn == 0 else '_')
        ax2.text(pc_s[_pidx, 0], pc_s[_pidx, 1], pc_s[_pidx, 2],
                 f'{_pn+1}', color='red', fontsize=9, fontweight='bold',
                 ha='left', va='top')

    for pane in [ax2.xaxis.pane, ax2.yaxis.pane, ax2.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#444444')

    with plt.rc_context({'text.usetex': False}):
        ax2.set_xlabel(f'PC1  ({var[0]*100:.1f}%)', color='#CCCCCC', labelpad=8)
        ax2.set_ylabel(f'PC2  ({var[1]*100:.1f}%)', color='#CCCCCC', labelpad=8)
        ax2.set_zlabel(f'PC3  ({var[2]*100:.1f}%)', color='#CCCCCC', labelpad=8)
        ax2.set_title('Phase Space Reconstruction', color='#CCCCCC',
                      fontsize=12, fontname='Comfortaa', fontweight='bold')
    ax2.tick_params(colors='#CCCCCC')
    ax2.legend(fontsize=8, framealpha=0.8,
               facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#E0E0E0')
    _leg_ax2_main = ax2.get_legend()
    _leg_ax2_main.get_frame().set_alpha(0.8)
    for _h in _leg_ax2_main.legendHandles: _h.set_alpha(1.0)
    _leg_ax2_cyc  = ax2.legend(handles=_make_cycle_legend_handles(), loc='lower left', **_CYC_LEG_KW)
    _leg_ax2_cyc.get_frame().set_alpha(0.8)
    _leg_ax2_cyc.set_visible(False)
    ax2.add_artist(_leg_ax2_main)

    # Halving-Farben für Button (hidden)
    _cyc_colors_3d = ['grey', '#0000FF', '#90EE90', '#FF69B4', 'orange']
    _bounds_3d     = np.concatenate([[days_vecs[0]], halving_days, [days_vecs[-1]]])
    _cyc_lines     = []
    for k in range(len(_bounds_3d) - 1):
        _d0, _d1 = _bounds_3d[k], _bounds_3d[k + 1]
        _col = _cyc_colors_3d[k] if k < len(_cyc_colors_3d) else 'white'
        _mc  = (days_vecs >= _d0) & (days_vecs <= _d1)
        if _mc.sum() > 1:
            _ln, = ax2.plot(pc_s[_mc, 0], pc_s[_mc, 1], pc_s[_mc, 2],
                            color=_col, linewidth=2.5, alpha=0.9, visible=False)
            _cyc_lines.append(_ln)

    # ── Controls ──────────────────────────────────────────────────────────────
    # Matplotlib widgets need live references; otherwise callbacks can disappear
    # once this function returns.
    fig_main._widget_refs = []
    fig_main._widget_refs.append(
        add_cycle_button_fig1(
            fig_main,
            _pcyc_lines_3d + _pcyc_lines_ax1,
            _cyc_lines + _halv_lines_ax1,
            cycle_legends=[_leg_ax1_cyc, _leg_ax2_cyc],
            marker_artists=_tb_artists_ax1 + _tb_artists_ax2,
        )
    )
    fig_main._widget_refs.extend(add_view_buttons(fig_main, ax2))
    fig_main._widget_refs.extend(add_custom_view_buttons(fig_main, ax2))
    fig_main._widget_refs.extend(add_rotate_controls(fig_main, ax2))
    _slider, _toggle_marker = add_tag_slider(
        fig_main,
        ax_sl,
        float(days_n_all[mask_plot][0]),
        float(days_vecs[-1]),
        2675.0,
        ax1,
        ax2,
        days_n_all,
        _sm_all,
        pc_s,
        days_vecs,
        LABEL_WINDOW,
    )
    fig_main._widget_refs.append(_slider)
    fig_main._widget_refs.append(
        add_slider_marker_button(fig_main, _toggle_marker)
    )

    return fig_main
