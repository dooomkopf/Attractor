#!/usr/bin/env python3
"""Two complete SSM analyses on the BTC ensemble-n signal: linear vs log10 clock.

Analog zu analyze_residuals/13_cli_lambda2_visualize.py, aber das Eingangs-
Signal ist die Ensemble-gemittelte LPPL-Exponentenkurve n(t) (aus
attractor_n_ens.compute_ensemble_n_signal mit Fenstern [90,...,180]) statt
der log-Residuen der Preise.

Top    -- linear-time SSM (PC1, PC2 vs day)
Mitte  -- log10-time SSM   (PC1, PC2 vs log10(day))
Unten  -- Eigenwerte beider linearer M-Matrizen in komplexer Ebene
"""

import argparse
import os
import sys
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

HZ_MPLSTYLE = '/home/hz/Data/hz.mplstyle'
if os.path.exists(HZ_MPLSTYLE):
    plt.style.use(HZ_MPLSTYLE)
    mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']

from analyze_n_ens.constants import (  # noqa: E402
    DEFAULT_FILENAME, DEFAULT_M, DEFAULT_YEARS, START_IDX,
)
from analyze_n_ens.data import (  # noqa: E402
    build_n_ens_context, build_n_log_uniform_context,
)
from analyze_residuals.common import build_time_vector  # noqa: E402
from analyze_residuals.ssm_learn import run_slave_test  # noqa: E402


HALVING_DAYS = [1425, 2744, 4146, 5586]
HALVING_LABELS = ['H1', 'H2', 'H3', 'H4']
COLOR_PC1 = '#FFB04A'
COLOR_PC2 = '#B197FC'
COLOR_HALV = '#A0A0A0'
LEGEND_KW = dict(loc='upper left', fontsize=10, facecolor='#1A1A1A',
                 edgecolor='#808080', labelcolor='#E0E0E0', framealpha=0.85)


def build_one_analysis(filename, M, years, start_idx, ssm_dim, poly, time_mode,
                        window_sizes=None, log_uniform_mean=None,
                        log_uniform_apply_to_linear=False):
    """Build embedding+PCA+SSMLearn for one clock; return packaged result."""
    use_log_uniform = (
        log_uniform_mean is not None
        and (time_mode == 'log' or log_uniform_apply_to_linear)
    )
    if use_log_uniform:
        if hasattr(log_uniform_mean, '__iter__') and len(log_uniform_mean) == 1:
            target_mean = float(log_uniform_mean[0])
        else:
            target_mean = log_uniform_mean
        payload = build_n_log_uniform_context(
            filename, M, years, start_idx,
            target_mean_w_days=target_mean, time_mode=time_mode,
        )
    else:
        payload = build_n_ens_context(filename, M, years, start_idx, time_mode=time_mode,
                                       window_sizes=window_sizes)
    ctx = payload['ctx']
    days = ctx['days_vecs'].astype(float)
    pc1 = ctx['pc'][:, 0]
    pc2 = ctx['pc'][:, 1]
    time_vec = build_time_vector(days, time_mode)
    res = run_slave_test(ctx, ssm_dim, poly, max_slave_pc=8,
                         time_vec=time_vec, time_mode=time_mode)
    eigs = res.get('eigvals')
    lam_master = None
    T_master = None
    lam_fit = None
    if eigs is not None and len(eigs) >= 2:
        pos_pairs = [e for e in eigs if e.imag > 1e-12]
        complex_pairs = [e for e in eigs if abs(e.imag) > 1e-12]
        candidates = pos_pairs if pos_pairs else complex_pairs
        if candidates:
            lam_master = min(candidates, key=lambda e: abs(e.imag))
            T_master = 2.0 * np.pi / abs(lam_master.imag)
            if time_mode == 'log':
                lam_fit = 10.0 ** T_master
    fit_err = float(res.get('fit_err', float('nan')))
    var = ctx.get('var')
    cum_var_2 = float(np.sum(var[:2]) * 100.0) if var is not None else float('nan')
    return {
        'time_mode': time_mode,
        'days': days,
        'log10_days': np.log10(days),
        'pc1': pc1, 'pc2': pc2,
        'eigvals': eigs,
        'lam_master': lam_master,
        'T_master': T_master,
        'lambda_fit': lam_fit,
        'fit_err': fit_err,
        'cum_var_2': cum_var_2,
    }


def plot_eigenvalues(ax, ana, title, legend_loc='upper right',
                     text_corner='lower right', sci_xaxis=False):
    eigs = ana['eigvals']
    if eigs is None:
        ax.text(0.5, 0.5, 'no eigenvalues', transform=ax.transAxes,
                ha='center', va='center', color='#E0E0E0')
        ax.set_title(title)
        return
    re = np.array([e.real for e in eigs])
    im = np.array([e.imag for e in eigs])
    ax.axhline(0, color='#808080', lw=0.5, alpha=0.6)
    ax.axvline(0, color='#808080', lw=0.5, alpha=0.6)
    ax.scatter(re, im, s=45, c=COLOR_PC1, edgecolors='#FFFFFF', linewidth=1.0,
               zorder=10, label=r'eigenvalues $\mu_k$')
    if ana['lam_master'] is not None:
        alpha = ana['lam_master'].real
        omega = ana['lam_master'].imag
        T = ana['T_master']
        fit_err = ana.get('fit_err', float('nan'))
        cum_var = ana.get('cum_var_2', float('nan'))
        Q = abs(omega) / (2.0 * abs(alpha)) if abs(alpha) > 1e-30 else float('inf')
        if ana['time_mode'] == 'log':
            text_phys = (r'$\alpha = ' + f'{alpha:+.3e}' + r'$' + '\n' +
                         r'$\omega = ' + f'{omega:+.3e}' + r'$' + '\n' +
                         r'$T = ' + f'{T:.3f}' + r'$' + '\n' +
                         r'$\lambda = 10^T = ' + f'{ana["lambda_fit"]:.3f}' + r'$')
        else:
            T_y = T / 365.25
            text_phys = (r'$\alpha = ' + f'{alpha:+.3e}' + r'$' + '\n' +
                         r'$\omega = ' + f'{omega:+.3e}' + r'$' + '\n' +
                         r'$T = ' + f'{T:.0f}\,\mathrm{{d}} = ' + f'{T_y:.2f}\,\mathrm{{y}}' + r'$')
        text_err = (r'decoder\_err $= ' + f'{fit_err:.3f}' + r'$, ' +
                    r'cum\_var(2) $= ' + f'{cum_var:.1f}\,\%' + r'$, ' +
                    r'Q $= ' + f'{Q:.2f}' + r'$')
        if text_corner == 'lower left':
            tx, ty, tha, tva = 0.02, 0.02, 'left', 'bottom'
        elif text_corner == 'upper left':
            tx, ty, tha, tva = 0.02, 0.98, 'left', 'top'
        elif text_corner == 'upper right':
            tx, ty, tha, tva = 0.98, 0.98, 'right', 'top'
        else:
            tx, ty, tha, tva = 0.98, 0.02, 'right', 'bottom'
        ax.text(tx, ty, text_phys, transform=ax.transAxes, va=tva, ha=tha,
                fontsize=9, color='#E0E0E0', zorder=50,
                bbox=dict(facecolor='#1A1A1A', edgecolor='#808080', alpha=0.85))
        ax.text(0.5, -0.18, text_err, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, color='#CCCCCC')
    ax.set_xlabel(r'$\mathrm{Re}(\mu) = \alpha$', fontsize=12)
    ax.set_ylabel(r'$\mathrm{Im}(\mu) = \omega$', fontsize=12)
    ax.set_title(title)
    ax.tick_params(labelsize=11)
    if sci_xaxis:
        ax.ticklabel_format(style='sci', scilimits=(-3, 3), axis='x', useMathText=True)
        ax.xaxis.get_offset_text().set_visible(False)
        # Custom offset text via ax.text — will respect the requested position
        # Read after ticks computed: tick_format gives the exponent
        ax.figure.canvas.draw()  # force layout to compute exponent
        offset_str = ax.xaxis.get_major_formatter().get_offset()
        if offset_str:
            ax.annotate(offset_str, xy=(1.08, -0.02), xycoords='axes fraction',
                        xytext=(15, 2), textcoords='offset points',
                        color='white', fontsize=12, ha='right', va='top')
    ax.legend(loc=legend_loc, fontsize=9, facecolor='#1A1A1A',
              edgecolor='#808080', labelcolor='#E0E0E0', framealpha=0.85)
    if len(re) > 0:
        max_x = max(np.max(np.abs(re)), 1e-30) * 1.5
        max_y = max(np.max(np.abs(im)), 1e-30) * 1.5
        ax.set_xlim(-max_x, max_x)
        ax.set_ylim(-max_y, max_y)


def plot_time_series(ax, ana, axis_clock):
    if axis_clock == 'log10':
        x = ana['log10_days']
        x_label = r'$\log_{10}(\mathrm{day})$ since Genesis'
    else:
        x = ana['days']
        x_label = r'day since Genesis'

    ax.plot(x, ana['pc1'], color=COLOR_PC1, lw=0.9, label='PC1')
    ax.plot(x, ana['pc2'], color=COLOR_PC2, lw=0.7, alpha=0.7, label='PC2')

    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    first_halv = True
    for h_day, h_lab in zip(HALVING_DAYS, HALVING_LABELS):
        if h_day < ana['days'][0] or h_day > ana['days'][-1]:
            continue
        x_h = np.log10(h_day) if axis_clock == 'log10' else h_day
        ax.axvline(x_h, color=COLOR_HALV, lw=1.0, alpha=0.85, ls='--',
                   label='Halving' if first_halv else None)
        ax.text(x_h, 0.78, h_lab, transform=trans, color=COLOR_HALV,
                fontsize=10, fontweight='bold', ha='center', va='top',
                bbox=dict(facecolor='#1A1A1A', edgecolor='none', alpha=0.7, pad=1.5))
        first_halv = False
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('PC amplitude', fontsize=12)
    ax.tick_params(labelsize=13)
    ax.legend(**LEGEND_KW)


def main():
    ap = argparse.ArgumentParser(
        description='Two SSM analyses on BTC ensemble-n signal: linear vs log10 clock')
    ap.add_argument('--filename', default=DEFAULT_FILENAME)
    ap.add_argument('--M', type=int, default=DEFAULT_M)
    ap.add_argument('--years', type=float, default=DEFAULT_YEARS)
    ap.add_argument('--start_idx', type=int, default=START_IDX)
    ap.add_argument('--ssm_dim', type=int, default=3)
    ap.add_argument('--poly', type=int, default=2)
    ap.add_argument('--windows', type=int, nargs='+', default=None,
                    help='Window sizes (days) for n_ens; default = [90,100,...,180]')
    ap.add_argument('--log_uniform_mean', type=float, nargs='+', default=None,
                    help='If set, SSM B uses log-uniform window (W(t)=c*t) with mean=N days. '
                         'Multiple values create ensemble. SSM A unaffected unless --log_uniform_both.')
    ap.add_argument('--log_uniform_both', action='store_true',
                    help='Apply log-uniform-window n(t) to BOTH SSMs.')
    ap.add_argument('--no-show', action='store_true')
    ap.add_argument('--save', type=str, default=None)
    args = ap.parse_args()

    print('Building linear-mode SSM analysis...')
    ana_lin = build_one_analysis(args.filename, args.M, args.years, args.start_idx,
                                 args.ssm_dim, args.poly, time_mode='linear',
                                 window_sizes=args.windows,
                                 log_uniform_mean=args.log_uniform_mean,
                                 log_uniform_apply_to_linear=args.log_uniform_both)
    print('Building log10-mode SSM analysis...')
    ana_log = build_one_analysis(args.filename, args.M, args.years, args.start_idx,
                                 args.ssm_dim, args.poly, time_mode='log',
                                 window_sizes=args.windows,
                                 log_uniform_mean=args.log_uniform_mean)

    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.9])
    ax_lin_ts = fig.add_subplot(gs[0, :])
    ax_log_ts = fig.add_subplot(gs[1, :])
    ax_lin_eig = fig.add_subplot(gs[2, 0])
    ax_log_eig = fig.add_subplot(gs[2, 1])

    LINEAR_FRAME = '#69DB7C'
    LOG10_FRAME = '#4DABF7'

    def _frame(ax, color):
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)

    _frame(ax_lin_ts, LINEAR_FRAME)
    _frame(ax_lin_eig, LINEAR_FRAME)
    _frame(ax_log_ts, LOG10_FRAME)
    _frame(ax_log_eig, LOG10_FRAME)

    plot_time_series(ax_lin_ts, ana_lin, axis_clock='linear')

    from scipy.signal import find_peaks as _find_peaks
    _idx_lin, _ = _find_peaks(ana_lin['pc1'],
                              prominence=float(np.std(ana_lin['pc1']) * 0.5))
    h5_est = HALVING_DAYS[-1] + 3.9 * 365.25
    halv_ext = HALVING_DAYS + [h5_est]

    def _mark_peaks(ax_ts, ana, idx_arr, pc_key, color_face, color_edge,
                    label_str, num_color, arrow_y=None, use_nearest=False):
        """Mark peaks (dots + numbers); arrows/percent labels removed."""
        if len(idx_arr) == 0:
            return
        days = ana['days']
        pc = ana[pc_key]
        ax_ts.scatter(days[idx_arr], pc[idx_arr],
                      s=70, c=color_face, edgecolors=color_edge, linewidth=1.5,
                      zorder=12, label=label_str)
        for n, idx in enumerate(idx_arr, start=1):
            d = days[idx]
            ax_ts.annotate(str(n), (d, pc[idx]),
                           xytext=(6, 6), textcoords='offset points',
                           color=num_color, fontsize=11, fontweight='bold')

    _mark_peaks(ax_lin_ts, ana_lin, _idx_lin, 'pc1',
                color_face='#FFEB3B', color_edge='#69DB7C',
                label_str='PC1 maxima', num_color='#FFEB3B',
                use_nearest=True)

    _idx_pc2_lin, _ = _find_peaks(ana_lin['pc2'],
                                  prominence=float(np.std(ana_lin['pc2']) * 0.5))
    _mark_peaks(ax_lin_ts, ana_lin, _idx_pc2_lin, 'pc2',
                color_face='#4DABF7', color_edge='#69DB7C',
                label_str='PC2 maxima', num_color='#4DABF7',
                arrow_y=0.0, use_nearest=False)

    ax_lin_ts.legend(**LEGEND_KW)
    if ana_lin['T_master'] is not None:
        T_y = ana_lin['T_master'] / 365.25
        ax_lin_ts.set_title(
            r'SSM A: linear-time sampling.  '
            r'Master $T = ' + f'{ana_lin["T_master"]:.0f}' +
            r'\,\mathrm{d} = ' + f'{T_y:.2f}' + r'\,\mathrm{y}$')
    else:
        ax_lin_ts.set_title('SSM A: linear-time sampling')

    plot_time_series(ax_log_ts, ana_log, axis_clock='log10')

    def _mark_log10_from_lin(ax_log, ana_log_inner, idx_arr_lin, ana_lin_inner,
                             pc_key, color_face, color_edge, label_str, num_color):
        if len(idx_arr_lin) == 0:
            return
        first = True
        for n, idx in enumerate(idx_arr_lin, start=1):
            d = ana_lin_inner['days'][idx]
            x_log = np.log10(d)
            y_log = float(np.interp(d, ana_log_inner['days'],
                                    ana_log_inner[pc_key]))
            ax_log.scatter(x_log, y_log, s=70, c=color_face,
                           edgecolors=color_edge, linewidth=1.5, zorder=12,
                           label=label_str if first else None)
            ax_log.annotate(str(n), (x_log, y_log),
                            xytext=(8, 8), textcoords='offset points',
                            color=num_color, fontsize=11, fontweight='bold')
            first = False

    _mark_log10_from_lin(ax_log_ts, ana_log, _idx_lin, ana_lin, 'pc1',
                         color_face='#FFEB3B', color_edge='#69DB7C',
                         label_str='linear PC1 maxima', num_color='#FFEB3B')
    _mark_log10_from_lin(ax_log_ts, ana_log, _idx_pc2_lin, ana_lin, 'pc2',
                         color_face='#4DABF7', color_edge='#69DB7C',
                         label_str='linear PC2 maxima', num_color='#4DABF7')
    ax_log_ts.legend(**LEGEND_KW)
    if ana_log['T_master'] is not None and ana_log['lambda_fit'] is not None:
        ax_log_ts.set_title(
            r'SSM B: log10-time sampling.  '
            r'Master $T = ' + f'{ana_log["T_master"]:.3f}' +
            r'\;\log_{10}(\mathrm{d})$' +
            r',  $\lambda = 10^T = ' +
            f'{ana_log["lambda_fit"]:.3f}' + r'$')
    else:
        ax_log_ts.set_title('SSM B: log10-time sampling')

    if args.windows is None:
        win_title = 'windows=[90,..,180]'
    elif len(args.windows) == 1:
        win_title = f'windows={args.windows[0]}'
    else:
        win_title = f'windows={list(args.windows)}'
    if args.log_uniform_mean is not None:
        if len(args.log_uniform_mean) == 1:
            win_title += f', log_uniform_mean={int(args.log_uniform_mean[0])}'
        else:
            win_title += (f', log_uniform_mean='
                          f'{[int(m) for m in args.log_uniform_mean]}')
        if args.log_uniform_both:
            win_title += ', log_uniform_both'
    with plt.rc_context({'text.usetex': False}):
        plt.suptitle(f'BTC Ensemble n: two independent SSM analyses with different time samplings  '
                     f'[ssm_dim={args.ssm_dim}, poly={args.poly}]',
                     color='#CCCCCC', fontsize=13, y=0.985,
                     fontname='Comfortaa', fontweight='bold')
        fig.text(0.5, 0.962, win_title,
                 ha='center', va='top', color='#AAAAAA', fontsize=11)

    plot_eigenvalues(ax_lin_eig, ana_lin,
                     title=r'Linear-clock eigenvalues of $M$',
                     legend_loc='upper left', text_corner='lower left',
                     sci_xaxis=True)
    plot_eigenvalues(ax_log_eig, ana_log,
                     title=r'Log10-clock eigenvalues of $M$',
                     legend_loc='upper right', text_corner='lower right')

    plt.subplots_adjust(top=0.92, bottom=0.07, hspace=0.38, wspace=0.28)

    if args.save is None:
        if args.windows is None:
            wtag = 'wens'
        elif len(args.windows) == 1:
            wtag = f'w{args.windows[0]}'
        else:
            wtag = f'w{min(args.windows)}-{max(args.windows)}'
        if args.log_uniform_mean is not None:
            lu_suffix = 'both' if args.log_uniform_both else ''
            if len(args.log_uniform_mean) == 1:
                lu_str = f'{int(args.log_uniform_mean[0])}'
            else:
                lu_str = '-'.join(str(int(m)) for m in args.log_uniform_mean)
            wtag = f'{wtag}_lu{lu_str}{lu_suffix}'
        args.save = os.path.join(
            HERE,
            f'13_lambda2_visualize_dim{args.ssm_dim}_poly{args.poly}_{wtag}.png',
        )
    fig.savefig(args.save, dpi=300, facecolor='#0a0a0a')
    print(f'plot saved to {args.save}')
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
