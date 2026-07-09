#!/usr/bin/env python3
"""02: Embedding — Scree-Plot, Mode-Shapes, Varianzschnitt."""

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ATTRACTOR = os.path.dirname(os.path.dirname(_HERE))
for p in [_HERE, _ATTRACTOR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt
import numpy as np
from SSM_res_data import load_data
from SSM_res_embedding import build_embedding, pca
from SSM_res_plots_modes import plot_modes, plot_variance

try:
    plt.style.use(os.path.join(_ATTRACTOR, 'hz.mplstyle'))
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description='02: Embedding + PCA visualization')
    ap.add_argument('--M', type=int, default=35)
    ap.add_argument('--tau', type=int, default=41)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--start-idx', type=int, default=1164)
    ap.add_argument('--filename', type=str, default=os.path.join(_ATTRACTOR, 'ziel.csv'))
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    data = load_data(args.filename, args.start_idx)
    D, W = build_embedding(data['signal'], args.M, args.tau)
    pca_res = pca(D)

    cum = np.cumsum(pca_res.var)
    print("=" * 72)
    print("SSM/res EMBEDDING + PCA")
    print("=" * 72)
    print(f"  M={args.M}  tau={args.tau}d  W={W}d  N_vec={D.shape[0]}")
    print(f"  Varianz: PC1={pca_res.var[0]*100:.1f}%  PC2={pca_res.var[1]*100:.1f}%  "
          f"PC3={pca_res.var[2]*100:.1f}%  PC4={pca_res.var[3]*100:.1f}%")
    print(f"  Kumulativ: 2PC={cum[1]*100:.1f}%  4PC={cum[3]*100:.1f}%  "
          f"6PC={cum[5]*100:.1f}%  8PC={cum[7]*100:.1f}%")
    print("=" * 72)

    if args.show:
        plot_modes(pca_res, args.M, args.tau, K_show=6, mode_label='log-res', fig_num=1)
        plot_variance(pca_res, args.K, signal_label='log-res', fig_num=2)
        plt.show()


if __name__ == '__main__':
    main()
