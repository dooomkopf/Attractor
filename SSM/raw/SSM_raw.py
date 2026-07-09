#!/usr/bin/env python3
"""
SSM_raw.py — Spectral-Submanifold-Analyse auf BTC-Rohdaten (ziel.csv).

Phase 1, Schritte 1-2 (dieser Stand):
    1) Daten laden, log/lin Signal, ab START_IDX (default 1164)
    2) Time Delay Embedding → PCA → Mode-Plot + Variance-Plot

Phase 1, Schritt 3 (kommt nach User-Review):
    3) Geometrie-Fit Slaved-PCs = W(PC1, PC2), Polynom-Order-Scan,
       Manifold-3D-Plot

Modulares Layout (alles im selben Ordner SSM/raw/):
    data.py       — read_btc_data, load_data
    embedding.py  — build_embedding, pca, smooth_pcs, PCAResult
    plots.py      — plot_modes, plot_variance (später plot_manifold)
    SSM_raw.py    — Entry-Point: CLI + main()

Aufruf:
    python3 SSM_raw.py [--log|--lin] [--M 35] [--tau 41] [--K 5] [--sigma 60]
                       [--start-idx 1164] [--filename /abs/ziel.csv]
                       [--no-show] [--save-prefix /tmp/ssm_raw]
"""

import argparse
import os
import sys

# Modul-Pfad: alle Geschwister im selben Ordner verfügbar machen,
# auch wenn das Skript von woanders aus aufgerufen wird.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from data      import load_data
from embedding import build_embedding, pca, smooth_pcs
from plots     import plot_modes, plot_variance


# ── Style ─────────────────────────────────────────────────────────────────────
_STYLEFILE = '/home/hz/Data/Attractor/hz.mplstyle'
_ZIEL_DEF  = '/home/hz/Data/Attractor/ziel.csv'

if os.path.exists(_STYLEFILE):
    plt.style.use(_STYLEFILE)
mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']


def main():
    ap = argparse.ArgumentParser(
        description='SSM_raw.py — SSM-Phase 1 Schritte 1-2 auf BTC-Rohdaten')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--log', action='store_true', help='Signal = log(price) (Default)')
    grp.add_argument('--lin', action='store_true', help='Signal = price')

    ap.add_argument('--M',         type=int,   default=35,   help='Embedding dim (default 35)')
    ap.add_argument('--tau',       type=int,   default=41,   help='Lag in Tagen (default 41)')
    ap.add_argument('--K',         type=int,   default=5,    help='Anzahl slaved PCs für Phase 3 (default 5)')
    ap.add_argument('--sigma',     type=float, default=60,   help='Smoothing σ für 3D-Viz (default 60)')
    ap.add_argument('--start-idx', type=int,   default=1164, help='Start-day (default 1164)')
    ap.add_argument('--filename',  type=str,   default=_ZIEL_DEF, help='ziel.csv Pfad')
    ap.add_argument('--no-show',   action='store_true', help='Kein plt.show() (headless)')
    ap.add_argument('--save-prefix', type=str, default=None,
                    help='Wenn gesetzt, speichere PNGs mit diesem Pfad-Präfix')
    args = ap.parse_args()

    mode = 'lin' if args.lin else 'log'
    sig_label = 'log\\,p' if mode == 'log' else 'p'

    print(f"SSM_raw  signal={mode}  M={args.M}  τ={args.tau}d  "
          f"K={args.K}  σ={args.sigma}  start_idx={args.start_idx}")
    print(f"  filename: {args.filename}")

    # 1) Daten laden
    days_emb, signal, dates_emb, halving_days = load_data(
        args.filename, args.start_idx, mode=mode)
    print(f"  loaded N={len(signal)}  range [day {days_emb[0]} .. {days_emb[-1]}]"
          f"  = {dates_emb[0].strftime('%Y-%m-%d')} .. {dates_emb[-1].strftime('%Y-%m-%d')}")

    # 2) Embedding + PCA
    D, W = build_embedding(signal, args.M, args.tau)
    print(f"  embedding D shape={D.shape}  W={W}d")
    pca_res = pca(D)
    print(f"  PCA  cum-var(K=2): {np.cumsum(pca_res.var)[1]*100:.2f}%  "
          f"cum-var(K={args.K+2}): {np.cumsum(pca_res.var)[args.K+1]*100:.2f}%")

    # smoothing für später (Manifold-3D in Phase 3) — bereits bauen
    pc_s = smooth_pcs(pca_res.pc, args.sigma)
    _ = pc_s  # noqa: F841 — wird in Schritt 3 verwendet

    # 3) Plots: Mode + Variance
    K_show = max(args.K + 2, 6)
    fig1 = plot_modes(pca_res, args.M, args.tau, K_show, mode_label=sig_label)
    fig2 = plot_variance(pca_res, args.K, signal_label=sig_label)

    # Save vor show, damit PNGs immer geschrieben werden
    if args.save_prefix:
        out1 = f"{args.save_prefix}_modes.png"
        out2 = f"{args.save_prefix}_variance.png"
        fig1.savefig(out1, dpi=150, facecolor='black')
        fig2.savefig(out2, dpi=150, facecolor='black')
        print(f"  saved: {out1}")
        print(f"  saved: {out2}")

    # Show — alle Figures parallel offen via einem einzigen show() am Ende
    if not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
