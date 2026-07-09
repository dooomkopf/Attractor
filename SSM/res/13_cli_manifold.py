#!/usr/bin/env python3
"""04: Manifold — polynomial decoder W(PC1,PC2)->PC3.. fit + 3D manifold plot."""

import argparse
import os
import sys
import importlib.util
# Pre-load pip-mpl_toolkits.mplot3d BEVOR matplotlib importiert wird —
# sonst greift matplotlib's Auto-Register auf das alte System-mpl_toolkits
# (/usr/lib/python3/dist-packages/) zu und scheitert an entferntem
# matplotlib.docstring/rcParams/_preprocess_data in matplotlib 3.6+.
_PIP_SP = '/home/hz/.local/lib/python3.10/site-packages'
for _name, _file in [('mpl_toolkits.mplot3d', f'{_PIP_SP}/mpl_toolkits/mplot3d/__init__.py'),
                     ('mpl_toolkits.mplot3d.axes3d', f'{_PIP_SP}/mpl_toolkits/mplot3d/axes3d.py'),
                     ('mpl_toolkits.mplot3d.art3d', f'{_PIP_SP}/mpl_toolkits/mplot3d/art3d.py')]:
    _spec = importlib.util.spec_from_file_location(_name, _file)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ATTRACTOR = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _ATTRACTOR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt
import numpy as np
from SSM_res_data import load_data
from SSM_res_embedding import build_embedding, pca, smooth_pcs
from SSM_res_geometry import fit_geometry, scan_orders
from SSM_res_phase import compute_phase_full
from SSM_res_plots_manifold import plot_manifold

try:
    plt.style.use(os.path.join(_ATTRACTOR, 'hz.mplstyle'))
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description='04: SSM manifold fit + 3D plot')
    ap.add_argument('--M', type=int, default=35)
    ap.add_argument('--tau', type=int, default=41)
    ap.add_argument('--ssm_dim', type=int, default=7,
                    help='Gesamtdimension des reduzierten SSM-Modells '
                         '(master_dim=2 hardcoded + slaved = ssm_dim - 2). '
                         'Default 7 = 2 master + 5 slaved PCs.')
    ap.add_argument('--sigma', type=float, default=60)
    ap.add_argument('--order', type=int, default=3)
    ap.add_argument('--scan-orders', action='store_true')
    ap.add_argument('--start-idx', type=int, default=1164)
    ap.add_argument('--filename', type=str, default=os.path.join(_ATTRACTOR, 'ziel.csv'))
    ap.add_argument('--time_mode', choices=['linear', 'log'], default='linear',
                    help='linear: Embedding auf täglich uniformem Gitter. '
                         'log: vor Embedding auf uniform log10(t)-Gitter resamplen.')
    ap.add_argument('--fracdiff', nargs='?', const=0.98, default=None, type=float,
                    metavar='d',
                    help='Periodic signal filter mit Parameter d in (0,1). '
                         'Cross-Design: bei time_mode=linear → log-time-fracdiff '
                         '(unterdrückt log-periodisch); bei time_mode=log → '
                         'linear-time-fracdiff (unterdrückt periodisch). '
                         'Beispiel: --fracdiff 0.15. Weggelassen: aus.')
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    data = load_data(args.filename, args.start_idx)
    if args.fracdiff is not None:
        # Cross-Design periodic signal filter (Hosking (1-B)^d):
        # - time_mode=linear → log-time fracdiff (unterdrückt log-periodisch),
        #                      Resultat zurückprojiziert auf daily-Gitter
        # - time_mode=log    → linear-time fracdiff (unterdrückt periodisch),
        #                      anschließend log-Resampling im nächsten Block
        _d = args.fracdiff
        if '/home/hz/Data' not in sys.path:
            sys.path.insert(0, '/home/hz/Data')
        from fracdiff_filter import FractionalDifferencing
        fd = FractionalDifferencing(d=_d)
        sig = data['signal']
        days_lin = data['days_emb'].astype(float)
        if args.time_mode == 'linear':
            tau_raw = np.log10(np.maximum(days_lin, 1.0))
            tau_grid = np.linspace(tau_raw[0], tau_raw[-1], len(tau_raw))
            sig_tau = np.interp(tau_grid, tau_raw, sig)
            sig_tau_fd = fd.filter(sig_tau)
            days_tau = 10.0 ** tau_grid
            sig_back = np.interp(days_lin, days_tau, sig_tau_fd)
            data = dict(data)
            data['signal'] = sig_back
            print(f'Periodic signal filter ({_d}) applied (log-time → linear-clock)')
        else:
            sig_fd = fd.filter(sig)
            data = dict(data)
            data['signal'] = sig_fd
            print(f'Periodic signal filter ({_d}) applied (linear-time → log-clock)')
    if args.time_mode == 'log':
        # Uniform log10(t)-Resampling vor dem Embedding (Analogie zu
        # analyze_residuals/15_cli prewhiten_in_log_clock-Vorlage)
        sig = data['signal']
        days_lin = data['days_emb'].astype(float)
        if np.any(days_lin <= 0):
            raise ValueError('log-time mode requires strictly positive days')
        tau_raw = np.log10(days_lin)
        tau_grid = np.linspace(tau_raw[0], tau_raw[-1], len(tau_raw))
        sig_tau = np.interp(tau_grid, tau_raw, sig)
        data = dict(data)
        data['signal'] = sig_tau
        data['days_emb'] = (10.0 ** tau_grid).astype(float)
        print(f'log-time resampling: uniform τ-grid, N={len(tau_grid)}, '
              f'τ ∈ [{tau_grid[0]:.3f}, {tau_grid[-1]:.3f}] (days '
              f'[{data["days_emb"][0]:.0f}, {data["days_emb"][-1]:.0f}])')
    D, W = build_embedding(data['signal'], args.M, args.tau)
    pca_res = pca(D)
    pc_s = smooth_pcs(pca_res.pc, args.sigma)
    days_vecs = data['days_emb'][W:]

    phase_result = compute_phase_full(
        pca_res.pc, days_vecs, data['halving_days'],
        master_idx=(0, 1), anchor_idx=1)

    if args.scan_orders:
        fits = scan_orders(pca_res.pc, orders=(2, 3, 4, 5),
                           master_idx=(0, 1), K=args.ssm_dim - 2, start_deg=2)
        print("=" * 72)
        print("SSM/res MANIFOLD ORDER SCAN")
        print("=" * 72)
        for o, f in fits.items():
            cond_str = f"{f.cond:.2e}" if np.isfinite(f.cond) else "inf"
            pc_r2 = '  '.join(f'{r:.3f}' for r in f.R2_per_pc)
            print(f"  order={o}  R2_total={f.R2_total:.4f}  cond={cond_str}  [{pc_r2}]")
        print("=" * 72)
        fit_for_plot = fits.get(args.order, fit_geometry(
            pca_res.pc, master_idx=(0, 1), K=args.ssm_dim - 2, order=args.order, start_deg=2))
    else:
        fit_for_plot = fit_geometry(
            pca_res.pc, master_idx=(0, 1), K=args.ssm_dim - 2, order=args.order, start_deg=2)
        cond_str = f"{fit_for_plot.cond:.2e}" if np.isfinite(fit_for_plot.cond) else "inf"
        pc_r2 = '  '.join(f'{r:.3f}' for r in fit_for_plot.R2_per_pc)
        print("=" * 72)
        print("SSM/res MANIFOLD FIT")
        print("=" * 72)
        print(f"  order={args.order}  ssm_dim={args.ssm_dim} (K={args.ssm_dim-2})  R2_total={fit_for_plot.R2_total:.4f}  cond={cond_str}")
        print(f"  R2 per slaved PC: [{pc_r2}]")
        print("=" * 72)

    if args.show:
        dates_vecs = data['dates_emb'][W:]
        plot_manifold(pca_res, pc_s, fit_for_plot,
                      days_vecs, dates_vecs, data['halving_days'],
                      data['cycle_top_days'], data['cycle_top_labels'],
                      phase_result, master_idx=(0, 1), slaved_pos=0,
                      signal_label='log-res', fig_num=3,
                      ssm_dim=args.ssm_dim, time_mode=args.time_mode)
        plt.show()


if __name__ == '__main__':
    main()
