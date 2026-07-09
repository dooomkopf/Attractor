"""attractor_n_ens/attractor_n_ens_plot2.py — Figure 2: Polar + PC3, Figure 3: θ unwrapped"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .attractor_n_ens_utils import make_segments
from .attractor_n_ens_controls import (add_cycle_button_fig1, add_cycle_button_fig3,
                                       _make_cycle_legend_handles)

_CYC_LEG_KW = dict(fontsize=8, framealpha=0.8, facecolor=(0.102, 0.102, 0.102, 0.8),
                   edgecolor='#808080', labelcolor='#E0E0E0')


def plot_fig2(data, params):
    CMAP = params['CMAP']

    theta          = data['theta']
    r              = data['r']
    theta_uw       = data['theta_uw']
    pc             = data['pc']
    var            = data['var']
    days_vecs      = data['days_vecs']
    dates_vecs     = data['dates_vecs']
    halving_days   = data['halving_days']
    used_peak_days = data['used_peak_days']
    peak_labels    = data['peak_labels']
    peak_colors    = data['peak_colors']
    used_bottom_days = data['used_bottom_days']
    bottom_labels  = data['bottom_labels']
    bottom_colors  = data['bottom_colors']
    bottom_n_vals  = data['bottom_n_vals']
    _pcyc_idxs     = data['_pcyc_idxs']

    _b_mask = np.isfinite(bottom_n_vals)
    used_bottom_days_f = used_bottom_days[_b_mask]
    bottom_labels_f  = [bottom_labels[i] for i in np.where(_b_mask)[0]]
    bottom_colors_f  = [bottom_colors[i] for i in np.where(_b_mask)[0]]

    fig3, _ = plt.subplots(1, 2, figsize=(14, 7))
    fig3.patch.set_facecolor('black')
    fig3.subplots_adjust(top=0.78, bottom=0.08, left=0.06, right=0.96)
    _[0].set_visible(False)
    ax3a = fig3.add_subplot(1, 2, 1, projection='polar')
    ax3b = fig3.axes[1]
    ax3b.set_facecolor('#1A1A1A')

    _cyc_colors_pol = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF']
    _cyc_bounds     = list(_pcyc_idxs) + [len(theta)]

    # ── ax3a: Polar ───────────────────────────────────────────────────────────
    _n_cyc = len(_pcyc_idxs)
    _h2_i = _pcyc_idxs[0] if _pcyc_idxs else 0
    if _h2_i > 0:
        ax3a.scatter(theta[0:_h2_i], r[0:_h2_i], color='grey', s=3, alpha=0.6, linewidths=0)
    _pcyc_lines_pol = []
    for _ci in range(len(_pcyc_idxs)):
        _s, _e = _cyc_bounds[_ci], _cyc_bounds[_ci + 1]
        if _e > _s + 1:
            _sc = ax3a.scatter(theta[_s:_e], r[_s:_e],
                               color=_cyc_colors_pol[_ci % len(_cyc_colors_pol)],
                               s=3, alpha=0.8, linewidths=0)
            _pcyc_lines_pol.append(_sc)

    ax3a.set_ylim(0, r.max() * 1.1)
    ax3a.scatter(theta[-1], r[-1], color='white', s=30, zorder=20, edgecolors='black', linewidths=1.5, label='Latest Data')
    ax3a.set_yticklabels([])
    ax3a.set_rticks([])
    with plt.rc_context({'text.usetex': False}):
        ax3a.set_title('Cycle Phase in PC1-PC2 Plane', color='#CCCCCC',
                       fontsize=12, fontname='Comfortaa', fontweight='bold', y=1.15)
    ax3a.tick_params(colors='#CCCCCC')
    ax3a.set_facecolor('#1A1A1A')

    def _scatter_markers(ax, xdata, ydata, events, label_key, marker='o'):
        days_arr, labels, colors = events
        _artists = []
        for j, (pd_, lbl, col) in enumerate(zip(days_arr, labels, colors)):
            idx_ = np.argmin(np.abs(days_vecs - pd_))
            if np.abs(days_vecs[idx_] - pd_) < 200:
                _sc = ax.scatter(xdata[idx_], ydata[idx_], color=col, marker=marker,
                           s=60, zorder=10, edgecolors='white', linewidths=1.0,
                           label='_')
                _sc.set_visible(False)
                _artists.append(_sc)
                _xt = (-15, -12) if (ax is ax3a and lbl == 'B4') else (6, 4)
                _disp_lbl = f'{lbl} (prelim.)' if lbl == 'T4' else lbl
                _ann = ax.annotate(_disp_lbl, (xdata[idx_], ydata[idx_]),
                            textcoords='offset points', xytext=_xt,
                            color=col, fontsize=9, fontweight='bold')
                _ann.set_visible(False)
                _artists.append(_ann)
        return _artists

    _tb_ax3a  = _scatter_markers(ax3a, theta, r,
                     (used_peak_days, peak_labels, peak_colors), 'Cycle Top')
    _tb_ax3a += _scatter_markers(ax3a, theta, r,
                     (used_bottom_days_f, bottom_labels_f, bottom_colors_f), 'Cycle Bottom', marker='s')

    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_vecs - hday))
        if np.abs(days_vecs[hidx] - hday) < 200:
            ax3a.scatter(theta[hidx], r[hidx], color='grey', s=60, zorder=10,
                         edgecolors='white', linewidths=1.0,
                         label='Halving' if i == 0 else '_')
            ax3a.annotate(f'H{i+1}', (theta[hidx], r[hidx]),
                          textcoords='offset points', xytext=(-8, 6),
                          color='white', fontsize=9, fontweight='bold')

    _h_idxs = [np.argmin(np.abs(days_vecs - hd)) for hd in halving_days
               if np.min(np.abs(days_vecs - hd)) < 200]
    _y_offsets = [4, -14, 4, -14, 4]
    for _pn, _pidx in enumerate(_pcyc_idxs):
        ax3a.scatter(theta[_pidx], r[_pidx], color='red', s=80, zorder=11,
                     marker='^', edgecolors='white', linewidths=1.2,
                     label='PCA-Cycle-Start' if _pn == 0 else '_')
        ax3a.annotate(str(_pn + 1), (theta[_pidx], r[_pidx]),
                      textcoords='offset points', xytext=(-6, -12) if _pn == 0 else (6, -12),
                      color='red', fontsize=9, fontweight='bold')
        print(f"p-Zyklus {_pn+1}: idx={_pidx}  theta_uw={theta_uw[_pidx]:.3f}"
              f"  r={r[_pidx]:.3f}  datum={dates_vecs[_pidx].strftime('%d.%m.%Y')}")
    print(f"p-Zyklen gefunden: {len(_pcyc_idxs)}")

    if len(_h_idxs) >= 2:
        _theta_cycle = (theta_uw[_h_idxs[-1]] - theta_uw[_h_idxs[0]]) / (len(_h_idxs) - 1)
    else:
        _theta_cycle = 2 * np.pi
    _cyc_pct = (theta_uw[-1] - theta_uw[_h_idxs[-1]]) / _theta_cycle * 100

    # ── ax3b: Phase vs PC3 ────────────────────────────────────────────────────
    if _h2_i > 0:
        ax3b.scatter(theta[0:_h2_i], pc[0:_h2_i, 2], color='grey', s=3, alpha=0.6, linewidths=0)
    _pcyc_lines_phs = []
    for _ci in range(len(_pcyc_idxs)):
        _s, _e = _cyc_bounds[_ci], _cyc_bounds[_ci + 1]
        _sc = ax3b.scatter(theta[_s:_e], pc[_s:_e, 2],
                           color=_cyc_colors_pol[_ci % len(_cyc_colors_pol)],
                           s=3, alpha=0.8, linewidths=0)
        _pcyc_lines_phs.append(_sc)

    ax3b.scatter(theta[-1], pc[-1, 2], color='white', s=30, zorder=20,
                 edgecolors='black', linewidths=1.5, label='Latest Data')
    _tb_ax3b  = _scatter_markers(ax3b, theta, pc[:, 2],
                     (used_peak_days, peak_labels, peak_colors), 'Cycle Top')
    _tb_ax3b += _scatter_markers(ax3b, theta, pc[:, 2],
                     (used_bottom_days_f, bottom_labels_f, bottom_colors_f), 'Cycle Bottom', marker='s')

    for i, hday in enumerate(halving_days):
        hidx = np.argmin(np.abs(days_vecs - hday))
        if np.abs(days_vecs[hidx] - hday) < 200:
            ax3b.scatter(theta[hidx], pc[hidx, 2], color='grey', s=60, zorder=10,
                         edgecolors='white', linewidths=1.0,
                         label='Halving' if i == 0 else '_')
            ax3b.annotate(f'H{i+1}', (theta[hidx], pc[hidx, 2]),
                          textcoords='offset points', xytext=(-6, 4),
                          color='white', fontsize=9, fontweight='bold')

    with plt.rc_context({'text.usetex': False}):
        ax3b.set_xlabel('Cycle Phase [rad]', color='#E0E0E0')
        ax3b.set_ylabel(f'PC3  ({var[2]*100:.1f}%)', color='#E0E0E0')
        ax3b.set_title('Cycle Phase along PC3', color='#CCCCCC',
                       fontsize=12, fontname='Comfortaa', fontweight='bold', y=1.15)
    ax3b.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3b.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi,
                     5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 2*np.pi])
    ax3b.set_xticklabels([r'$0$', r'$\frac{\pi}{4}$', r'$\frac{2\pi}{4}$', r'$\frac{3\pi}{4}$',
                           r'$\frac{4\pi}{4}$', r'$\frac{5\pi}{4}$', r'$\frac{6\pi}{4}$',
                           r'$\frac{7\pi}{4}$', r'$\frac{8\pi}{4}$'])

    # Halving-Farben (hidden)
    _halv_colors_f2 = ['grey', '#0000FF', '#90EE90', '#FF69B4', 'orange']
    _halv_bounds_f2 = np.concatenate([[days_vecs[0]], halving_days, [days_vecs[-1]]])
    _halv_lines_pol, _halv_lines_phs = [], []
    for _k in range(len(_halv_bounds_f2) - 1):
        _d0, _d1 = _halv_bounds_f2[_k], _halv_bounds_f2[_k + 1]
        _col = _halv_colors_f2[_k % len(_halv_colors_f2)]
        _mc  = (days_vecs >= _d0) & (days_vecs <= _d1)
        if _mc.sum() > 1:
            _lp = ax3a.scatter(theta[_mc], r[_mc], color=_col, s=3, alpha=0.8, linewidths=0)
            _lp.set_visible(False)
            _halv_lines_pol.append(_lp)
            _lph = ax3b.scatter(theta[_mc], pc[_mc, 2], color=_col, s=3, alpha=0.8, linewidths=0)
            _lph.set_visible(False)
            _halv_lines_phs.append(_lph)

    for _pn, _pidx in enumerate(_pcyc_idxs):
        ax3b.scatter(theta[_pidx], pc[_pidx, 2], color='red', s=80, zorder=12,
                     marker='^', edgecolors='white', linewidths=1.2, alpha=1.0,
                     label='PCA-Cycle-Start' if _pn == 0 else '_')
        ax3b.annotate(str(_pn + 1), (theta[_pidx], pc[_pidx, 2]),
                      textcoords='offset points', xytext=(6, -12),
                      color='red', fontsize=9, fontweight='bold', zorder=13,
                      bbox=dict(boxstyle='square,pad=0.15', facecolor='#1A1A1A',
                                edgecolor='none', alpha=0.85))

    ax3b.legend(loc='upper right', fontsize=8, framealpha=0.8,
                facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#E0E0E0')
    _leg_ax3b_main = ax3b.get_legend()
    _leg_ax3b_main.get_frame().set_alpha(0.8)
    for _h in _leg_ax3b_main.legendHandles: _h.set_alpha(1.0)
    _leg_ax3b_cyc  = ax3b.legend(handles=_make_cycle_legend_handles(), loc='upper left', **_CYC_LEG_KW)
    _leg_ax3b_cyc.get_frame().set_alpha(0.8)
    _leg_ax3b_cyc.set_visible(False)
    ax3b.add_artist(_leg_ax3b_main)

    fig3._widget_refs = []
    fig3._widget_refs.append(
        add_cycle_button_fig3(fig3, _pcyc_lines_pol, _pcyc_lines_phs,
                               _halv_lines_pol, _halv_lines_phs,
                               cycle_legends=[_leg_ax3b_cyc],
                               marker_artists=_tb_ax3a + _tb_ax3b)
    )

    fig3.tight_layout()
    return fig3


def plot_fig3(data, params):
    CMAP = params['CMAP']

    theta_uw         = data['theta_uw']
    t_norm_vec       = data['t_norm_vec']
    days_vecs        = data['days_vecs']
    dates_vecs       = data['dates_vecs']
    halving_days     = data['halving_days']
    _pcyc_idxs       = data['_pcyc_idxs']
    used_peak_days   = data['used_peak_days']
    peak_labels      = data['peak_labels']
    peak_colors      = data['peak_colors']
    used_bottom_days = data['used_bottom_days']
    bottom_labels    = data['bottom_labels']
    bottom_colors    = data['bottom_colors']
    bottom_n_vals    = data['bottom_n_vals']

    # Zeit-Normierung ab frühesten Embedding-Daten (nicht 2010)
    _t_scatter = (days_vecs - days_vecs[0]) / (days_vecs[-1] - days_vecs[0])
    _yr_ticks  = np.linspace(0, 1, 6)
    _yr_labels = [dates_vecs[int(t * (len(dates_vecs) - 1))].strftime('%Y')
                  for t in _yr_ticks]

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    fig4.patch.set_facecolor('black')

    def _halving_on(ax, xvals):
        _first = [True]
        for i, (hday, xv) in enumerate(zip(halving_days, xvals)):
            hidx = np.argmin(np.abs(days_vecs - hday))
            if np.abs(days_vecs[hidx] - hday) < 200:
                ax.scatter(xv, theta_uw[hidx], color='grey', s=70, zorder=10,
                           edgecolors='white', linewidths=1.0,
                           label='Halving' if _first[0] else '_')
                _first[0] = False
                ax.annotate(f'H{i+1}', (xv, theta_uw[hidx]),
                            textcoords='offset points', xytext=(-15, 5),
                            color='white', fontsize=9, fontweight='bold')

    # PCA-Cycle-farbige Linien (sichtbar by default)
    _cyc_col_f4 = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF']
    _pcyc_bnd   = list(_pcyc_idxs) + [len(theta_uw)]
    _pcyc_ln_a, _pcyc_ln_b = [], []
    # grau vor H2 (vor erstem Cycle-Start)
    _pre = _pcyc_bnd[0]
    if _pre > 1:
        _la_pre, = ax4a.plot(days_vecs[:_pre+1], theta_uw[:_pre+1],
                             color='grey', lw=2.5, alpha=1.0, zorder=5)
        _lb_pre, = ax4b.plot(np.log(days_vecs[:_pre+1]), theta_uw[:_pre+1],
                             color='grey', lw=2.5, alpha=1.0, zorder=5)
        _pcyc_ln_a.append(_la_pre)
        _pcyc_ln_b.append(_lb_pre)
    for _ci in range(len(_pcyc_idxs)):
        _s, _e = _pcyc_bnd[_ci], _pcyc_bnd[_ci + 1]
        if _e > _s + 1:
            _col = _cyc_col_f4[_ci % len(_cyc_col_f4)]
            _la, = ax4a.plot(days_vecs[_s:_e], theta_uw[_s:_e],
                             color=_col, lw=2.5, alpha=1.0, zorder=5)
            _lb, = ax4b.plot(np.log(days_vecs[_s:_e]), theta_uw[_s:_e],
                             color=_col, lw=2.5, alpha=1.0, zorder=5)
            _pcyc_ln_a.append(_la)
            _pcyc_ln_b.append(_lb)

    # Halving-farbige Linien (versteckt)
    _halv_col_f4 = ['grey', '#0000FF', '#90EE90', '#FF69B4', 'orange']
    _halv_bnd    = np.concatenate([[days_vecs[0]], halving_days, [days_vecs[-1]]])
    _halv_ln_a, _halv_ln_b = [], []
    for _k in range(len(_halv_bnd) - 1):
        _d0, _d1 = _halv_bnd[_k], _halv_bnd[_k + 1]
        _mc = (days_vecs >= _d0) & (days_vecs <= _d1)
        if _mc.sum() > 1:
            _col = _halv_col_f4[_k % len(_halv_col_f4)]
            _la, = ax4a.plot(days_vecs[_mc], theta_uw[_mc],
                             color=_col, lw=2.0, alpha=0.8, zorder=3, visible=False)
            _lb, = ax4b.plot(np.log(days_vecs[_mc]), theta_uw[_mc],
                             color=_col, lw=2.0, alpha=0.8, zorder=3, visible=False)
            _halv_ln_a.append(_la)
            _halv_ln_b.append(_lb)

    _sc_b = None
    for _ax, _xv, _xl, _ttl in [
        (ax4a, days_vecs,         'Day index  (linear)',  'Cycle phase vs t  (linear)'),
        (ax4b, np.log(days_vecs), 'log(Day index)',        'Cycle phase vs log(t)'),
    ]:
        _ax.set_facecolor('#1A1A1A')
        _sc = _ax.scatter(_xv, theta_uw, c=_t_scatter, cmap=CMAP,
                          s=8, alpha=0.15, linewidths=0, zorder=1)
        if _ax is ax4b:
            _sc_b = _sc
        _ax.autoscale()
        _halving_on(_ax, _xv[[np.argmin(np.abs(days_vecs - h)) for h in halving_days]])
        for _pn, _pidx in enumerate(_pcyc_idxs):
            _ax.scatter(_xv[_pidx], theta_uw[_pidx], color='red', s=80, zorder=11,
                        marker='^', edgecolors='white', linewidths=1.2,
                        label='PCA-Cycle-Start' if _pn == 0 else '_')
            _ax.annotate(str(_pn + 1), (_xv[_pidx], theta_uw[_pidx]),
                         textcoords='offset points', xytext=(6, -12),
                         color='red', fontsize=9, fontweight='bold')
        with plt.rc_context({'text.usetex': False}):
            _ax.set_xlabel(_xl, color='#E0E0E0')
            _ax.set_ylabel('θ unwrapped  [rad]', color='#E0E0E0')
            _ax.set_title(_ttl, color='#CCCCCC', fontsize=12,
                          fontname='Comfortaa', fontweight='bold')
        _ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        _ax.legend(fontsize=8, framealpha=0.8,
                   facecolor='#1A1A1A', edgecolor='#808080', labelcolor='#E0E0E0')

    # ── Top/Bottom-Marker für Fig4 (hidden, togglebar) ───────────────────────
    _tb_artists_fig4 = []
    for _ax, _xv in [(ax4a, days_vecs), (ax4b, np.log(days_vecs))]:
        for _pd, _lbl, _col in zip(used_peak_days, peak_labels, peak_colors):
            _hidx = np.argmin(np.abs(days_vecs - _pd))
            if _pd <= days_vecs[-1] and np.abs(days_vecs[_hidx] - _pd) < 30:
                _sc = _ax.scatter(_xv[_hidx], theta_uw[_hidx], color=_col,
                                  marker='o', s=60, zorder=12,
                                  edgecolors='white', linewidths=1.0, label='_')
                _sc.set_visible(False)
                _disp = f'{_lbl} (prelim.)' if _lbl == 'T4' else _lbl
                _ann = _ax.annotate(_disp, (_xv[_hidx], theta_uw[_hidx]),
                                    textcoords='offset points',
                                    xytext=(-70, 4) if _lbl == 'T4' else (6, 4),
                                    color=_col, fontsize=9, fontweight='bold')
                _ann.set_visible(False)
                _tb_artists_fig4.extend([_sc, _ann])
        for _pd, _lbl, _col in zip(used_bottom_days, bottom_labels, bottom_colors):
            _hidx = np.argmin(np.abs(days_vecs - _pd))
            if _pd <= days_vecs[-1] and np.abs(days_vecs[_hidx] - _pd) < 30:
                _sc = _ax.scatter(_xv[_hidx], theta_uw[_hidx], color=_col,
                                  marker='s', s=60, zorder=12,
                                  edgecolors='white', linewidths=1.0, label='_')
                _sc.set_visible(False)
                _ann = _ax.annotate(_lbl, (_xv[_hidx], theta_uw[_hidx]),
                                    textcoords='offset points', xytext=(6, 4),
                                    color=_col, fontsize=9, fontweight='bold')
                _ann.set_visible(False)
                _tb_artists_fig4.extend([_sc, _ann])

    _cyc_legs_fig4 = []
    for _ax in (ax4a, ax4b):
        _leg_main = _ax.get_legend()
        _leg_main.get_frame().set_alpha(0.8)
        for _h in _leg_main.legendHandles: _h.set_alpha(1.0)
        _leg_cyc  = _ax.legend(handles=_make_cycle_legend_handles(), loc='lower right', **_CYC_LEG_KW)
        _leg_cyc.get_frame().set_alpha(0.8)
        _leg_cyc.set_visible(False)
        _ax.add_artist(_leg_main)
        _cyc_legs_fig4.append(_leg_cyc)

    fig4._widget_refs = []
    fig4._widget_refs.append(
        add_cycle_button_fig1(fig4, _pcyc_ln_a + _pcyc_ln_b,
                               _halv_ln_a + _halv_ln_b,
                               cycle_legends=_cyc_legs_fig4,
                               marker_artists=_tb_artists_fig4)
    )

    fig4.tight_layout()
    return fig4
