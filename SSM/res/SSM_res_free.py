#!/usr/bin/env python3
"""
SSM_res_free.py — Modellfreie Geometrie-Analyse parallel zu SSM_res.

Dieselben Daten, dasselbe TDE (gleiche M, τ, W, start_idx wie SSM_res.py),
aber KEIN Polynom-Fit, KEINE globale Funktionsform, KEINE vorgegebene Dim.

Methoden (alle modellfrei, nur Distanzen / lokale Hauptachsen):

    Fig 0  — Eingangs-Sanity (kopiert von SSM_res.py)
    Fig 1  — Pure TDE 3D der ROHEN Lag-Koordinaten X(t), X(t-τ), X(t-2τ)
             keine PCA, keine Glättung, keine Marker-Ausrichtung
    Fig 2  — TWO-NN intrinsische Dimension d_intr (Facco et al. 2017)
             Histogramm der μ-Verteilung + Linear-Fit log(1-F̂) vs log(μ)
    Fig 3  — Lokale Geometrie aus k-NN PCA pro Punkt
             3D-Trajektorie farbcodiert mit lokaler 2D-ness +
             Histogramm der 2D-ness-Verteilung

Aufruf (verwendet dieselben Defaults wie SSM_res.py):
    python3 SSM_res_free.py [--M 35] [--tau 41] [--sigma 60]
                            [--k_local 50]
                            [--start-idx 1164] [--filename /abs/ziel.csv]
                            [--no-show] [--save-prefix /tmp/ssm_res_free]
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

from SSM_res_data           import load_data
from SSM_res_embedding      import build_embedding, pca, smooth_pcs
from SSM_res_intrinsic_dim  import two_nn
from SSM_res_local_pca      import local_pca, LocalPCAResult
from SSM_res_plots_input    import plot_residuals
from SSM_res_plots_free     import (plot_pure_tde, plot_intrinsic_dim,
                                    plot_local_geometry)


# ── Style ─────────────────────────────────────────────────────────────────────
_STYLEFILE = '/home/hz/Data/Attractor/hz.mplstyle'
_ZIEL_DEF  = '/home/hz/Data/Attractor/ziel.csv'

if os.path.exists(_STYLEFILE):
    plt.style.use(_STYLEFILE)
mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']


def main():
    ap = argparse.ArgumentParser(
        description='SSM_res_free.py — modellfreie Geometrie-Analyse '
                    'parallel zu SSM_res')

    # Embedding-Parameter — bewusst dieselben Defaults wie SSM_res.py
    ap.add_argument('--M',         type=int,   default=35,   help='Embedding dim (default 35)')
    ap.add_argument('--tau',       type=int,   default=41,   help='Lag in Tagen (default 41)')
    ap.add_argument('--sigma',     type=float, default=60,   help='Smoothing σ NUR für Trajektorien-Viz (default 60)')
    ap.add_argument('--start-idx', type=int,   default=1164, help='Start-day (default 1164)')
    ap.add_argument('--percentile', type=float, default=0.01, help='QuantReg-Quantil (default 0.01)')
    ap.add_argument('--filename',  type=str,   default=_ZIEL_DEF, help='ziel.csv Pfad')

    # Modellfreie Methoden-Parameter
    ap.add_argument('--k_local',   type=int,   default=50,   help='k für Local PCA (default 50)')
    ap.add_argument('--n_boot',    type=int,   default=20,   help='Bootstrap-Replikate für TWO-NN std (default 20)')
    ap.add_argument('--no-norm',   dest='norm', action='store_false', default=True,
                    help='Normierung deaktivieren (Default: Cycle-Amplitude normiert)')
    ap.add_argument('--full-range', action='store_true',
                    help='Visualisierung über volle Reihe (überschreibt --cycle)')
    ap.add_argument('--cycle',     type=str,   default='3,4',
                    help='Visualisierungs-Range als "i,j" mit i,j ∈ {1..4} '
                         '→ zwischen Halving i und Halving j (Default "3,4")')

    ap.add_argument('--no-show',   action='store_true',      help='Kein plt.show() (headless)')
    ap.add_argument('--save-prefix', type=str, default=None,
                    help='Wenn gesetzt, speichere PNGs mit diesem Pfad-Präfix')
    args = ap.parse_args()

    norm_tag = '+norm' if args.norm else 'NOnorm'
    range_tag = 'full-range' if args.full_range else 'cycle H3..H4'
    print(f"SSM_res_free  M={args.M}  τ={args.tau}d  σ={args.sigma}  "
          f"start_idx={args.start_idx}  k_local={args.k_local}  "
          f"{norm_tag}  {range_tag}")
    print(f"  filename: {args.filename}")

    # ── 1) Daten laden + QuantReg ─────────────────────────────────────────
    data = load_data(args.filename, args.start_idx,
                     percentile=args.percentile, norm=args.norm)
    signal    = data['signal']
    days_emb  = data['days_emb']
    dates_emb = data['dates_emb']
    print(f"  loaded N_full={len(data['days_all'])}  "
          f"N_emb={len(signal)}  range [day {days_emb[0]} .. {days_emb[-1]}]"
          f"  = {dates_emb[0].strftime('%Y-%m-%d')} .. {dates_emb[-1].strftime('%Y-%m-%d')}")

    # ── 2) Embedding (gleiche M, τ, W wie SSM_res!) ───────────────────────
    D, W = build_embedding(signal, args.M, args.tau)
    days_vecs  = days_emb[W:]
    dates_vecs = data['dates_emb'][W:]
    print(f"  embedding D shape={D.shape}  W={W}d")

    # PCA NUR für die 3D-Visualisierung in Fig 3 (kein Fit, keine Geometrie-
    # Annahme — PCA ist hier reine Achsen-Rotation für die Anschauung).
    pca_res = pca(D)
    pc_for_traj = smooth_pcs(pca_res.pc, args.sigma)

    # ── 3) Modellfreie Methode A: TWO-NN intrinsische Dimension ──────────
    print("  Computing TWO-NN intrinsic dimension ...")
    twonn_res = two_nn(D, n_boot=args.n_boot)
    print(f"    d_intr = {twonn_res.d_intr:.3f} ± {twonn_res.d_intr_std:.3f}  "
          f"(N_used={twonn_res.n_used}/{twonn_res.n_total})")

    # ── 4) Modellfreie Methode B: Local PCA ──────────────────────────────
    print(f"  Computing Local PCA (k={args.k_local}) ...")
    local_res = local_pca(D, k=args.k_local)
    finite = np.isfinite(local_res.two_d_ness)
    print(f"    local 2D-ness  mean = {local_res.two_d_ness[finite].mean():.3f}  "
          f"std = {local_res.two_d_ness[finite].std():.3f}  "
          f"min = {local_res.two_d_ness[finite].min():.3f}  "
          f"max = {local_res.two_d_ness[finite].max():.3f}")

    # ── Range-Filter für Visualisierung: standardmäßig nur Cycle H3..H4 ──
    halving_days = data['halving_days']
    cycle_top_days = data['cycle_top_days']
    cycle_top_labels = data['cycle_top_labels']

    # --cycle parsen: "i,j" mit i,j ∈ {1..len(halving_days)}
    cycle_pair = None
    if not args.full_range:
        try:
            ci, cj = [int(s) for s in args.cycle.split(',')]
            if not (1 <= ci <= len(halving_days) and 1 <= cj <= len(halving_days)):
                raise ValueError
            ci, cj = sorted((ci, cj))
            cycle_pair = (ci, cj)
        except Exception:
            print(f"  WARN: --cycle '{args.cycle}' ungültig, fallback auf full-range")
            args.full_range = True

    if args.full_range or cycle_pair is None:
        sl = slice(0, len(days_vecs))
        days_plot       = days_vecs
        dates_plot      = dates_vecs
        D_plot          = D
        pc_traj_plot    = pc_for_traj
        halv_plot       = halving_days
        tops_d_plot     = cycle_top_days
        tops_l_plot     = cycle_top_labels
        local_res_plot  = local_res
        range_day_lo    = int(days_vecs[0])
        range_day_hi    = int(days_vecs[-1])
        print(f"  visualization range: full ({len(days_vecs)} points)")
    else:
        ci, cj = cycle_pair
        h_lo = int(halving_days[ci - 1])
        h_hi = int(halving_days[cj - 1])
        i_lo = int(np.argmin(np.abs(days_vecs - h_lo)))
        i_hi = int(np.argmin(np.abs(days_vecs - h_hi)))
        i_lo, i_hi = (min(i_lo, i_hi), max(i_lo, i_hi))
        sl = slice(i_lo, i_hi + 1)
        h3, h4 = h_lo, h_hi   # alias for legacy lines below
        i3, i4 = i_lo, i_hi
        range_day_lo = h_lo
        range_day_hi = h_hi
        days_plot       = days_vecs[sl]
        dates_plot      = dates_vecs[sl]
        D_plot          = D[sl]
        pc_traj_plot    = pc_for_traj[sl]
        halv_plot       = np.array([h_lo, h_hi], dtype=int)
        tops_mask = (cycle_top_days >= days_plot[0]) & (cycle_top_days <= days_plot[-1])
        tops_d_plot = cycle_top_days[tops_mask]
        tops_l_plot = [cycle_top_labels[i] for i in range(len(cycle_top_labels))
                       if tops_mask[i]]
        # Local PCA Result auf den Range slicen (Histogramm + Color-Mapping
        # konsistent zur 3D-Trajektorie)
        local_res_plot = LocalPCAResult(
            k=local_res.k,
            eigenvalues=local_res.eigenvalues[sl],
            two_d_ness=local_res.two_d_ness[sl],
            intrinsic_local_dim=local_res.intrinsic_local_dim[sl],
        )
        print(f"  visualization range: cycle H{ci}..H{cj}  "
              f"days [{days_plot[0]}..{days_plot[-1]}]  "
              f"= {dates_plot[0].strftime('%Y-%m-%d')}..{dates_plot[-1].strftime('%Y-%m-%d')}  "
              f"({len(days_plot)} points)")

    # ── Plots ────────────────────────────────────────────────────────────
    # fig_num bewusst 0..3 (klein), keine Konflikt-Range mehr
    fig0 = plot_residuals(data, percentile=args.percentile, fig_num=0,
                          shade_range=(range_day_lo, range_day_hi))
    fig1 = plot_pure_tde(D_plot, days_plot, dates_plot, halv_plot,
                         tops_d_plot, tops_l_plot,
                         args.M, args.tau, fig_num=1)
    fig2 = plot_intrinsic_dim(twonn_res, fig_num=2)
    fig3 = plot_local_geometry(pc_traj_plot, local_res_plot,
                               days_plot, dates_plot,
                               halv_plot, tops_d_plot, tops_l_plot,
                               master_idx=(0, 1), slaved_idx=2, fig_num=3)

    if args.save_prefix:
        out0 = f"{args.save_prefix}_residuals.png"
        out1 = f"{args.save_prefix}_pure_tde.png"
        out2 = f"{args.save_prefix}_intrinsic_dim.png"
        out3 = f"{args.save_prefix}_local_geometry.png"
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
