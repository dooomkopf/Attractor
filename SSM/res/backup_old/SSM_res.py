#!/usr/bin/env python3
"""
SSM_res.py — Spectral-Submanifold-Analyse auf BTC-QuantReg-Residuen.

Phase 1, alle Schritte 1-3:
    Schritt 1: Daten laden, QuantReg-Detrending → log-Residuen
    Schritt 2: Time Delay Embedding → PCA → Mode-Plot + Variance-Plot
    Schritt 3: Geometrie-Fit Slaved-PCs = W(PC1, PC2),
               Polynom-Order-Scan {2,3,4,5}, Manifold-3D-Plot

Modulares Layout (alles im selben Ordner SSM/res/):
    data.py             — read_btc_data, load_data (mit QuantReg, --norm Option)
    embedding.py        — build_embedding, pca, smooth_pcs, PCAResult
    geometry.py         — poly_features_2d, fit_geometry, scan_orders, GeometryFit
    plots_input.py      — plot_residuals  (Sanity-Check, Fig 0)
    plots_modes.py      — plot_modes, plot_variance  (Fig 1, 2)
    plots_manifold.py   — plot_manifold  (3D Surface, Fig 3)
    SSM_res.py          — Entry-Point: CLI + main()

Aufruf:
    python3 SSM_res.py [--M 35] [--tau 41] [--K 5] [--sigma 60]
                       [--order 3] [--scan-orders] [--norm]
                       [--start-idx 1164] [--filename /abs/ziel.csv]
                       [--no-show] [--save-prefix /tmp/ssm_res]
"""

import argparse
import os
import sys

# Modul-Pfad: alle Geschwister im selben Ordner verfügbar machen
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from SSM_res_data            import load_data
from SSM_res_embedding       import build_embedding, pca, smooth_pcs
from SSM_res_geometry        import fit_geometry, scan_orders
from SSM_res_phase           import compute_phase_full
from SSM_res_plots_input     import plot_residuals
from SSM_res_plots_modes     import plot_modes, plot_variance
from SSM_res_plots_manifold  import plot_manifold


# ── Style ─────────────────────────────────────────────────────────────────────
_STYLEFILE = '/home/hz/Data/Attractor/hz.mplstyle'
_ZIEL_DEF  = '/home/hz/Data/Attractor/ziel.csv'

if os.path.exists(_STYLEFILE):
    plt.style.use(_STYLEFILE)
mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']


def _print_fit_summary(fit, label=''):
    """Kompakte Ausgabe der Fit-Diagnostik."""
    cond_str = f"{fit.cond:.2e}" if np.isfinite(fit.cond) else "∞"
    cond_warn = ''
    if fit.cond > 1e10:
        cond_warn = '  ← FRAGIL'
    elif fit.cond > 1e8:
        cond_warn = '  ← warnung'
    pc_r2 = '  '.join(f'{r:.3f}' for r in fit.R2_per_pc)
    print(f"  {label}order={fit.order}  R²={fit.R2_total:.4f}  "
          f"cond={cond_str}{cond_warn}  rank={fit.rank}/{fit.B.shape[0]}")
    print(f"      R² per slaved PC: [{pc_r2}]")


def main():
    ap = argparse.ArgumentParser(
        description='SSM_res.py — SSM-Phase 1 Schritte 1-3 auf BTC-Residuen')

    ap.add_argument('--M',         type=int,   default=35,   help='Embedding dim (default 35)')
    ap.add_argument('--tau',       type=int,   default=41,   help='Lag in Tagen (default 41)')
    ap.add_argument('--K',         type=int,   default=5,    help='Anzahl slaved PCs (default 5)')
    ap.add_argument('--sigma',     type=float, default=60,   help='Smoothing σ für 3D-Viz (default 60)')
    ap.add_argument('--order',     type=int,   default=3,    help='Polynom-Grad für Single-Fit (default 3)')
    ap.add_argument('--scan-orders', action='store_true',    help='Scan {2,3,4,5} statt single')
    ap.add_argument('--norm',      action='store_true',      help='Cycle-Amplitude normieren (exp-Decay raus)')
    ap.add_argument('--percentile', type=float, default=0.01, help='QuantReg-Quantil (default 0.01)')
    ap.add_argument('--start-idx', type=int,   default=1164, help='Start-day (default 1164)')
    ap.add_argument('--filename',  type=str,   default=_ZIEL_DEF, help='ziel.csv Pfad')
    ap.add_argument('--no-show',   action='store_true',      help='Kein plt.show() (headless)')
    ap.add_argument('--save-prefix', type=str, default=None,
                    help='Wenn gesetzt, speichere PNGs mit diesem Pfad-Präfix')
    args = ap.parse_args()

    norm_tag = '+norm' if args.norm else ''
    print(f"SSM_res  M={args.M}  τ={args.tau}d  K={args.K}  σ={args.sigma}  "
          f"start_idx={args.start_idx}  {norm_tag}")
    print(f"  filename: {args.filename}")

    # 1) Daten laden + QuantReg
    data = load_data(args.filename, args.start_idx,
                     percentile=args.percentile, norm=args.norm)
    signal    = data['signal']
    days_emb  = data['days_emb']
    dates_emb = data['dates_emb']
    print(f"  loaded N_full={len(data['days_all'])}  "
          f"N_emb={len(signal)}  range [day {days_emb[0]} .. {days_emb[-1]}]"
          f"  = {dates_emb[0].strftime('%Y-%m-%d')} .. {dates_emb[-1].strftime('%Y-%m-%d')}")

    # 2) Embedding + PCA
    D, W = build_embedding(signal, args.M, args.tau)
    print(f"  embedding D shape={D.shape}  W={W}d")
    pca_res = pca(D)
    cum2 = np.cumsum(pca_res.var)[1] * 100
    cumK = np.cumsum(pca_res.var)[args.K + 1] * 100
    print(f"  PCA  cum-var(K=2): {cum2:.2f}%  cum-var(K={args.K+2}): {cumK:.2f}%")

    # Smoothing für die 3D-Trajektorie
    pc_s = smooth_pcs(pca_res.pc, args.sigma)
    days_vecs = days_emb[W:]   # Embedding hat (N-W) Vektoren

    # 2.5) Intrinsische Phase + Cycle-Bounds aus Master-Koordinaten
    phase_result = compute_phase_full(
        pca_res.pc, days_vecs, data['halving_days'],
        master_idx=(0, 1), anchor_idx=1)
    n_cyc = max(0, len(phase_result.bounds) - 1)
    print(f"  phase  direction={phase_result.direction:+d}  "
          f"anchor_shift={phase_result.anchor_shift:.3f} rad  "
          f"intrinsic cycles in embedding range = {n_cyc}")

    # 3) Geometrie-Fit — Manifold-Plot benutzt IMMER args.order (Default 3).
    #    --scan-orders zeigt zusätzlich die Diagnostik der höheren Orders,
    #    wählt aber nicht automatisch die "beste" Order — wir bleiben bei
    #    der vom User gewählten (default 3).
    if args.scan_orders:
        fits = scan_orders(pca_res.pc, orders=(2, 3, 4, 5),
                           master_idx=(0, 1), K=args.K, start_deg=2)
        print("  Order-Scan (Diagnostik):")
        for o, f in fits.items():
            _print_fit_summary(f, label='  ')
        if args.order in fits:
            fit_for_plot = fits[args.order]
        else:
            fit_for_plot = fit_geometry(
                pca_res.pc, master_idx=(0, 1),
                K=args.K, order=args.order, start_deg=2)
        print(f"  → Manifold-Plot mit Order {args.order}")
    else:
        fit_for_plot = fit_geometry(
            pca_res.pc, master_idx=(0, 1),
            K=args.K, order=args.order, start_deg=2)
        _print_fit_summary(fit_for_plot)

    # ── Plots erzeugen (alle gleichzeitig offen) ─────────────────────────────
    sig_label = data['signal_label']

    fig0 = plot_residuals(data, percentile=args.percentile, fig_num=0)
    fig1 = plot_modes(pca_res, args.M, args.tau, K_show=6,
                      mode_label=sig_label, fig_num=1)
    fig2 = plot_variance(pca_res, args.K, signal_label=sig_label, fig_num=2)
    dates_vecs = data['dates_emb'][W:]
    fig3 = plot_manifold(pca_res, pc_s, fit_for_plot,
                         days_vecs, dates_vecs, data['halving_days'],
                         data['cycle_top_days'], data['cycle_top_labels'],
                         phase_result,
                         master_idx=(0, 1), slaved_pos=0,
                         signal_label=sig_label, fig_num=3)

    # Save vor show
    if args.save_prefix:
        out0 = f"{args.save_prefix}_residuals.png"
        out1 = f"{args.save_prefix}_modes.png"
        out2 = f"{args.save_prefix}_variance.png"
        out3 = f"{args.save_prefix}_manifold.png"
        fig0.savefig(out0, dpi=150, facecolor='black')
        fig1.savefig(out1, dpi=150, facecolor='black')
        fig2.savefig(out2, dpi=150, facecolor='black')
        fig3.savefig(out3, dpi=150, facecolor='black')
        print(f"  saved: {out0}")
        print(f"  saved: {out1}")
        print(f"  saved: {out2}")
        print(f"  saved: {out3}")

    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
