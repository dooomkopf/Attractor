#!/usr/bin/env python3
"""03: Intrinsic dimension — TWO-NN + local PCA: is the data cloud really 2D?"""

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
from SSM_res_intrinsic_dim import two_nn
from SSM_res_local_pca import local_pca

try:
    plt.style.use(os.path.join(_ATTRACTOR, 'hz.mplstyle'))
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser(description='03: Intrinsic dimension (TWO-NN + local PCA)')
    ap.add_argument('--M', type=int, default=35)
    ap.add_argument('--tau', type=int, default=41)
    ap.add_argument('--start-idx', type=int, default=1164)
    ap.add_argument('--filename', type=str, default=os.path.join(_ATTRACTOR, 'ziel.csv'))
    ap.add_argument('--n_pca_dim', type=int, default=8, help='PCA truncation for distance calc')
    ap.add_argument('--k_neighbors', type=int, default=20, help='neighbors for local PCA')
    show_group = ap.add_mutually_exclusive_group()
    show_group.add_argument('--show', dest='show', action='store_true')
    show_group.add_argument('--no-show', dest='show', action='store_false')
    ap.set_defaults(show=True)
    args = ap.parse_args()

    data = load_data(args.filename, args.start_idx)
    D, W = build_embedding(data['signal'], args.M, args.tau)
    pca_res = pca(D)
    pc_trunc = pca_res.pc[:, :args.n_pca_dim]

    twonn_result = two_nn(pc_trunc)
    d_twonn = twonn_result.d_intr
    lpca_result = local_pca(pc_trunc, k=args.k_neighbors)
    local_dims = lpca_result.intrinsic_local_dim

    print("=" * 72)
    print("SSM/res INTRINSIC DIMENSION")
    print("=" * 72)
    print(f"  TWO-NN estimate     : {d_twonn:.2f}")
    print(f"  local PCA (k={args.k_neighbors}):")
    print(f"    median            : {np.median(local_dims):.2f}")
    print(f"    mean              : {np.mean(local_dims):.2f}")
    print(f"    p10/p90           : {np.percentile(local_dims,10):.2f} / {np.percentile(local_dims,90):.2f}")
    print("=" * 72)

    if args.show:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        axes[0].hist(local_dims, bins=30, color='cyan', alpha=0.7)
        axes[0].axvline(np.median(local_dims), color='white', ls='--', lw=1)
        axes[0].set_xlabel('local dim')
        axes[0].set_ylabel('count')
        axes[0].set_title(f'Local PCA dims (median={np.median(local_dims):.2f})')

        axes[1].text(0.5, 0.5, f'TWO-NN: {d_twonn:.2f}',
                     ha='center', va='center', fontsize=16, transform=axes[1].transAxes)
        axes[1].set_title('TWO-NN global estimate')
        fig.suptitle('Intrinsic Dimension Check', fontsize=11)
        plt.show()


if __name__ == '__main__':
    main()
