#!/usr/bin/env python3
"""
Pedantischer Härtetest fuer die BTC-Attraktor-Frage.

Zwei exakt an den Originalpipelines orientierte Rekonstruktionen:
  1) Residuen wie in attractor.py
  2) ensemble-n wie in attractor_n_ens/attractor_n_ens_compute.py

Darauf zwei Tests:
  A) Cross-Observable Consistency
  B) Kaplan-Glass-artige Richtungs-Kohärenz
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from contextlib import redirect_stdout
from datetime import datetime

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist

from attractor_n_ens.attractor_n_ens_compute import compute_all


PERCENTILE = 0.01
START_IDX = 1164
TAU = 41
M = 35
NORMALIZE_WINDOWS = False
SMOOTH_SIGMA = 60
WINDOW_SIZES = list(range(90, 181, 10))
PHASE_OFFSET = 0.0
N_KNN = 15
N_CROSS_BASELINE = 20
N_KG_BASELINE = 10
THEILER = (M - 1) * TAU
KG_GRIDS = [8, 10, 12, 14, 16]

HALVINGS = [
    datetime(2012, 11, 28),
    datetime(2016, 7, 9),
    datetime(2020, 5, 11),
    datetime(2024, 4, 20),
]
CYCLE_TOPS = [
    (datetime(2013, 12, 4), "T1", "#0000FF"),
    (datetime(2017, 12, 16), "T2", "#90EE90"),
    (datetime(2021, 11, 9), "T3", "#FF69B4"),
    (datetime(2025, 10, 7), "T4", "orange"),
]
CYCLE_BOTTOMS = [
    (datetime(2015, 1, 14), "B1", "#0000FF"),
    (datetime(2018, 12, 14), "B2", "#90EE90"),
    (datetime(2022, 11, 21), "B3", "#FF69B4"),
    (datetime(2026, 2, 6), "B4", "orange"),
]


try:
    plt.style.use("hz.mplstyle")
except Exception:
    plt.rcParams.update({
        "figure.facecolor": "black",
        "axes.facecolor": "#1A1A1A",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#CCCCCC",
        "text.color": "#CCCCCC",
        "xtick.color": "#CCCCCC",
        "ytick.color": "#CCCCCC",
        "grid.color": "#666666",
        "legend.facecolor": "#1A1A1A",
        "legend.edgecolor": "#CCCCCC",
        "savefig.facecolor": "black",
        "font.size": 11,
    })


def read_btc_data(filename: str):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                data.append((int(parts[0]), float(parts[1]), datetime.strptime(parts[2], "%d.%m.%Y")))
    days = np.array([d[0] for d in data])
    prices = np.array([d[1] for d in data])
    dates = np.array([d[2] for d in data])
    return days, prices, dates


def make_segments(x, y):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def make_segments3d(x, y, z):
    pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def zscore_columns(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, ddof=1, keepdims=True)
    sd[sd == 0.0] = 1.0
    return (x - mu) / sd


def _build_residual_embedding():
    days_all, prices_all, dates_all = read_btc_data("ziel.csv")

    log_days_all = np.log(days_all)
    log_btc_all = np.log(prices_all)
    X_all = sm.add_constant(log_days_all)
    qr = QuantReg(log_btc_all, X_all).fit(q=PERCENTILE)
    log_fit_all = qr.predict(X_all)
    residuals_all = prices_all / np.exp(log_fit_all)
    log_res_all = np.log(residuals_all)

    d_min, d_max = days_all[0], days_all[-1]
    t_norm_all = (days_all - d_min) / (d_max - d_min)

    mask_emb = days_all >= START_IDX
    log_res = log_res_all[mask_emb]
    days_emb = days_all[mask_emb]
    N = len(log_res)
    W = (M - 1) * TAU

    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = log_res[W - j * TAU : N - j * TAU]
    if NORMALIZE_WINDOWS:
        D -= D.mean(axis=1, keepdims=True)

    D_c = D - D.mean(axis=0)
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc = D_c @ Vt.T
    var = s**2 / (s**2).sum()

    pc_s = pc.copy()
    for j in range(pc_s.shape[1]):
        pc_s[:, j] = gaussian_filter1d(pc[:, j], sigma=SMOOTH_SIGMA)

    days_vecs = days_emb[W:]
    t_norm_vec = (days_vecs - d_min) / (d_max - d_min)

    theta = np.arctan2(pc[:, 1], pc[:, 0])
    r = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2)
    theta_uw = np.unwrap(theta)
    if theta_uw[-1] < theta_uw[0]:
        theta = -theta
        theta_uw = -theta_uw

    halving_days = []
    for hd in HALVINGS:
        diffs = np.array([abs((d - hd).days) for d in dates_all])
        halving_days.append(days_all[np.argmin(diffs)])
    halving_days = np.array(halving_days)
    hidx_h2 = np.argmin(np.abs(days_vecs - halving_days[1]))
    if np.abs(days_vecs[hidx_h2] - halving_days[1]) < 200:
        shift = float(theta[hidx_h2])
        theta = (theta - shift) % (2 * np.pi)
        theta_uw = theta_uw - shift

    return {
        "days_all": days_all,
        "prices_all": prices_all,
        "dates_all": dates_all,
        "residuals_all": residuals_all,
        "log_res_all": log_res_all,
        "days_vecs": days_vecs,
        "pc": pc,
        "pc_s": pc_s,
        "var": var,
        "theta": theta,
        "r": r,
        "theta_uw": theta_uw,
        "t_norm_all": t_norm_all,
        "t_norm_vec": t_norm_vec,
        "halving_days": halving_days,
    }


def _ensemble_params():
    return dict(
        END_CUT=0,
        START_IDX=START_IDX,
        TAU=TAU,
        M=M,
        SMOOTH_SIGMA=SMOOTH_SIGMA,
        LABEL_WINDOW=30,
        CMAP=plt.cm.coolwarm,
        WINDOW_SIZES=WINDOW_SIZES,
        PHASE_OFFSET=PHASE_OFFSET,
        HALVINGS=HALVINGS,
        CYCLE_TOPS=CYCLE_TOPS,
        CYCLE_BOTTOMS=CYCLE_BOTTOMS,
        SHOW_FIG4=False,
    )


def _build_ensemble_embedding():
    days_all, prices_all, dates_all = read_btc_data("ziel.csv")
    with redirect_stdout(io.StringIO()):
        data = compute_all(days_all, prices_all, dates_all, _ensemble_params())
    return data


def _common_alignment(res_data, ens_data):
    common_days, res_idx, ens_idx = np.intersect1d(
        res_data["days_vecs"],
        ens_data["days_vecs"],
        return_indices=True,
    )
    if common_days.size == 0:
        raise RuntimeError("Keine gemeinsamen Tage zwischen Residuen und ensemble-n gefunden.")
    return common_days, res_idx, ens_idx


def _valid_knn_indices(dist_row, row_index, k, theiler, source_index_map=None):
    n = len(dist_row)
    lo = max(0, row_index - theiler)
    hi = min(n, row_index + theiler + 1)
    if lo <= 0 and hi >= n:
        return None

    if lo <= 0:
        valid_idx = np.arange(hi, n)
    elif hi >= n:
        valid_idx = np.arange(0, lo)
    else:
        valid_idx = np.concatenate([np.arange(0, lo), np.arange(hi, n)])

    if valid_idx.size < k:
        return None

    row_vals = dist_row[valid_idx]
    sel = np.argpartition(row_vals, k - 1)[:k]
    return valid_idx[sel]


def _knn_sets_from_distance(dist, k, theiler):
    n = dist.shape[0]
    sets = [None] * n
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        nbrs = _valid_knn_indices(dist[i], i, k, theiler)
        if nbrs is None:
            continue
        sets[i] = set(int(v) for v in nbrs)
        valid[i] = True
    return sets, valid


def _permuted_knn_sets(dist, perm, k, theiler):
    n = len(perm)
    sets = [None] * n
    valid = np.zeros(n, dtype=bool)
    for i in range(n):
        p_i = perm[i]
        lo = max(0, i - theiler)
        hi = min(n, i + theiler + 1)
        if lo <= 0 and hi >= n:
            continue
        if lo <= 0:
            valid_idx = np.arange(hi, n)
        elif hi >= n:
            valid_idx = np.arange(0, lo)
        else:
            valid_idx = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        if valid_idx.size < k:
            continue
        row_vals = dist[p_i, perm[valid_idx]]
        sel = np.argpartition(row_vals, k - 1)[:k]
        nbrs = valid_idx[sel]
        sets[i] = set(int(v) for v in nbrs)
        valid[i] = True
    return sets, valid


def _jaccard(a, b):
    if a is None or b is None:
        return np.nan
    union = len(a | b)
    if union == 0:
        return np.nan
    return len(a & b) / union


def cross_observable_consistency(res_points, ens_points, k=N_KNN, theiler=THEILER, seed=42):
    res_std = zscore_columns(res_points)
    ens_std = zscore_columns(ens_points)

    dist_res = cdist(res_std, res_std).astype(np.float32, copy=False)
    dist_ens = cdist(ens_std, ens_std).astype(np.float32, copy=False)

    res_sets, res_valid = _knn_sets_from_distance(dist_res, k, theiler)
    ens_sets, ens_valid = _knn_sets_from_distance(dist_ens, k, theiler)
    valid = res_valid & ens_valid

    overlap = np.array([_jaccard(res_sets[i], ens_sets[i]) for i in range(len(valid))], dtype=float)
    overlap = overlap[valid]

    rng = np.random.default_rng(seed)
    baseline_overlaps = []
    baseline_medians = []
    for _ in range(N_CROSS_BASELINE):
        perm = rng.permutation(len(ens_points))
        perm_sets, perm_valid = _permuted_knn_sets(dist_ens, perm, k, theiler)
        valid_b = res_valid & perm_valid
        ov = np.array([_jaccard(res_sets[i], perm_sets[i]) for i in range(len(valid_b))], dtype=float)
        ov = ov[valid_b]
        baseline_overlaps.append(ov)
        baseline_medians.append(np.nanmedian(ov))

    baseline_pooled = np.concatenate(baseline_overlaps) if baseline_overlaps else np.array([], dtype=float)

    return {
        "overlap": overlap,
        "overlap_median": float(np.nanmedian(overlap)),
        "overlap_mean": float(np.nanmean(overlap)),
        "baseline_pooled": baseline_pooled,
        "baseline_median": float(np.nanmedian(baseline_pooled)) if baseline_pooled.size else np.nan,
        "baseline_mean": float(np.nanmean(baseline_pooled)) if baseline_pooled.size else np.nan,
        "baseline_quantiles": tuple(float(np.nanquantile(baseline_pooled, q)) for q in (0.10, 0.50, 0.90)) if baseline_pooled.size else (np.nan, np.nan, np.nan),
        "valid_mask": valid,
    }


def kaplan_glass_score(points, grid_size):
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("points muss Form (N, 3) haben.")
    if len(x) < 3:
        return np.nan

    lo = np.min(x, axis=0)
    hi = np.max(x, axis=0)
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    u = (x - lo) / span
    u = np.clip(u, 0.0, 1.0)

    step = u[1:] - u[:-1]
    step_norm = np.linalg.norm(step, axis=1)
    good = step_norm > 1e-12
    if not np.any(good):
        return np.nan

    step_u = step[good] / step_norm[good, None]
    mid = 0.5 * (u[1:] + u[:-1])[good]
    cells = np.clip((mid * grid_size).astype(int), 0, grid_size - 1)

    sums = np.zeros((grid_size, grid_size, grid_size, 3), dtype=float)
    counts = np.zeros((grid_size, grid_size, grid_size), dtype=int)
    for vec, cell in zip(step_u, cells):
        i, j, k = cell
        sums[i, j, k] += vec
        counts[i, j, k] += 1

    valid = counts >= 2
    if not np.any(valid):
        return np.nan

    coher = np.linalg.norm(sums[valid], axis=1) / counts[valid]
    weights = counts[valid].astype(float)
    return float(np.sum(coher * weights) / np.sum(weights))


def kaplan_glass_curve(points, grid_sizes, n_shuffle=N_KG_BASELINE, seed=42):
    scores = np.array([kaplan_glass_score(points, g) for g in grid_sizes], dtype=float)
    rng = np.random.default_rng(seed)
    shuffle_scores = np.zeros((n_shuffle, len(grid_sizes)), dtype=float)
    for r in range(n_shuffle):
        perm = rng.permutation(len(points))
        shuffled = points[perm]
        shuffle_scores[r] = [kaplan_glass_score(shuffled, g) for g in grid_sizes]

    return {
        "scores": scores,
        "shuffle_scores": shuffle_scores,
        "shuffle_mean": np.nanmean(shuffle_scores, axis=0),
        "shuffle_q10": np.nanquantile(shuffle_scores, 0.10, axis=0),
        "shuffle_q90": np.nanquantile(shuffle_scores, 0.90, axis=0),
    }


def _plot_3d(ax, points, t_norm, title, label, color="white"):
    segs = make_segments3d(points[:, 0], points[:, 1], points[:, 2])
    lc = Line3DCollection(segs, cmap=plt.cm.coolwarm, linewidth=1.3, alpha=0.95)
    lc.set_array(t_norm)
    ax.add_collection3d(lc)
    for arr, setter in [(points[:, 0], ax.set_xlim), (points[:, 1], ax.set_ylim), (points[:, 2], ax.set_zlim)]:
        pad = (arr.max() - arr.min()) * 0.05
        setter(arr.min() - pad, arr.max() + pad)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor("#444444")
    ax.set_title(title, color="#E0E0E0", fontsize=10)
    ax.tick_params(colors="#CCCCCC")
    ax.set_xlabel(f"{label}1", color="#CCCCCC", labelpad=8)
    ax.set_ylabel(f"{label}2", color="#CCCCCC", labelpad=8)
    ax.set_zlabel(f"{label}3", color="#CCCCCC", labelpad=8)
    return lc


def _fmt_curve(values):
    return ", ".join(f"{g}:{v:.4f}" for g, v in zip(KG_GRIDS, values))


def main():
    ap = argparse.ArgumentParser(description="Pedantischer BTC-Attraktor-Haertetest")
    ap.add_argument("--save", type=str, default=None, help="Optionaler PNG-Ausgabepfad")
    ap.add_argument("--no-show", action="store_true", help="Plot nicht interaktiv anzeigen")
    ap.add_argument("--k", type=int, default=N_KNN, help="k fuer kNN-Overlap")
    ap.add_argument("--seed", type=int, default=42, help="Zufalls-Seed")
    args = ap.parse_args()

    res = _build_residual_embedding()
    ens = _build_ensemble_embedding()

    common_days, res_idx, ens_idx = _common_alignment(res, ens)
    res_common = res["pc"][res_idx, :3]
    ens_common = ens["pc"][ens_idx, :3]
    ens_common_s = ens["pc_s"][ens_idx, :3]
    common_t = np.linspace(0.0, 1.0, len(common_days))

    cross_main = cross_observable_consistency(res_common, ens_common_s, k=args.k, theiler=THEILER, seed=args.seed)
    cross_raw = cross_observable_consistency(res_common, ens_common, k=args.k, theiler=THEILER, seed=args.seed)
    kg_res_raw = kaplan_glass_curve(res_common, KG_GRIDS, n_shuffle=N_KG_BASELINE, seed=args.seed)
    kg_ens_raw = kaplan_glass_curve(ens_common, KG_GRIDS, n_shuffle=N_KG_BASELINE, seed=args.seed + 1)

    print("\n=== Verify Summary ===")
    print(f"Common days: {len(common_days)}")
    print(f"kNN Jaccard main raw/ens-smoothed median/mean: {cross_main['overlap_median']:.4f} / {cross_main['overlap_mean']:.4f}")
    print(f"kNN Jaccard sensitivity raw/raw median/mean: {cross_raw['overlap_median']:.4f} / {cross_raw['overlap_mean']:.4f}")
    print(f"Baseline Jaccard main median/mean: {cross_main['baseline_median']:.4f} / {cross_main['baseline_mean']:.4f}")
    print(f"Baseline Jaccard raw/raw median/mean: {cross_raw['baseline_median']:.4f} / {cross_raw['baseline_mean']:.4f}")
    print(f"KG residuals raw only: {_fmt_curve(kg_res_raw['scores'])}")
    print(f"KG residuals shuffle:  {_fmt_curve(kg_res_raw['shuffle_mean'])}")
    print(f"KG ensemble-n raw only:{_fmt_curve(kg_ens_raw['scores'])}")
    print(f"KG ensemble-n shuffle: {_fmt_curve(kg_ens_raw['shuffle_mean'])}")
    if args.save:
        print(f"Save target: {Path(args.save).resolve()}")
    else:
        print("Save target: none")

    fig = plt.figure(figsize=(22, 14))
    fig.patch.set_facecolor("black")
    gs = gridspec.GridSpec(3, 2, figure=fig, left=0.04, right=0.98, top=0.93, bottom=0.05, hspace=0.26, wspace=0.12)

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    ax1.set_facecolor("#1A1A1A")
    _plot_3d(ax1, res_common, common_t, "Residuen-Attraktor raw (common days)", "PC", color="white")
    ax1.scatter(res_common[-1, 0], res_common[-1, 1], res_common[-1, 2], color="white", s=35, zorder=10)

    ax2.set_facecolor("#1A1A1A")
    _plot_3d(ax2, ens_common_s, common_t, "ensemble-n-Attraktor smoothed view", "PC", color="white")
    ax2.scatter(ens_common_s[-1, 0], ens_common_s[-1, 1], ens_common_s[-1, 2], color="white", s=35, zorder=10)

    ax3.set_facecolor("#1A1A1A")
    valid_main = cross_main["valid_mask"]
    x_days_main = common_days[valid_main]
    valid_raw = cross_raw["valid_mask"]
    x_days_raw = common_days[valid_raw]
    ax3.plot(x_days_main, cross_main["overlap"], color="#4FD9E8", linewidth=1.0, alpha=0.95, label="main: raw/ens-smoothed")
    ax3.plot(x_days_raw, cross_raw["overlap"], color="#FFAA55", linewidth=1.0, alpha=0.80, label="sensitivity: raw/raw")
    q10, q50, q90 = cross_main["baseline_quantiles"]
    ax3.axhline(q50, color="#AAAAAA", linestyle="--", linewidth=1.0, label="baseline median")
    ax3.axhspan(q10, q90, color="#999999", alpha=0.14, label="baseline 10-90%")
    ax3.set_ylim(0, 1.0)
    ax3.set_title("Cross-Observable kNN-Jaccard-Overlap", color="#CCCCCC", fontsize=10)
    ax3.set_xlabel("Day index", color="#AAAAAA")
    ax3.set_ylabel("Jaccard", color="#AAAAAA")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=8, loc="lower left")
    ax3.tick_params(colors="#AAAAAA")

    ax4.set_facecolor("#1A1A1A")
    ax4.hist(cross_main["overlap"], bins=24, color="#4FD9E8", alpha=0.45, density=True, label="main: raw/ens-smoothed")
    ax4.hist(cross_raw["overlap"], bins=24, color="#FFAA55", alpha=0.45, density=True, label="sensitivity: raw/raw")
    if cross_main["baseline_pooled"].size:
        ax4.hist(cross_main["baseline_pooled"], bins=24, color="#AAAAAA", alpha=0.30, density=True, label="baseline")
    ax4.set_title("Overlap distribution", color="#CCCCCC", fontsize=10)
    ax4.set_xlabel("Jaccard", color="#AAAAAA")
    ax4.set_ylabel("Density", color="#AAAAAA")
    ax4.grid(True, alpha=0.25)
    ax4.legend(fontsize=8)
    ax4.tick_params(colors="#AAAAAA")

    ax5.set_facecolor("#1A1A1A")
    grids = np.array(KG_GRIDS, dtype=int)
    ax5.plot(grids, kg_res_raw["scores"], color="#4FD9E8", linewidth=1.6, marker="o", label="Residuals raw")
    ax5.plot(grids, kg_res_raw["shuffle_mean"], color="#4FD9E8", linestyle=":", linewidth=1.2, label="Residuals raw shuffle")
    ax5.fill_between(grids, kg_res_raw["shuffle_q10"], kg_res_raw["shuffle_q90"], color="#4FD9E8", alpha=0.10)
    ax5.plot(grids, kg_ens_raw["scores"], color="#FFAA55", linewidth=1.6, marker="o", label="ensemble-n raw")
    ax5.plot(grids, kg_ens_raw["shuffle_mean"], color="#FFAA55", linestyle=":", linewidth=1.2, label="ensemble-n raw shuffle")
    ax5.fill_between(grids, kg_ens_raw["shuffle_q10"], kg_ens_raw["shuffle_q90"], color="#FFAA55", alpha=0.10)
    ax5.set_title("Kaplan-Glass-style directional coherence (raw only)", color="#CCCCCC", fontsize=10)
    ax5.set_xlabel("Grid size", color="#AAAAAA")
    ax5.set_ylabel("Score", color="#AAAAAA")
    ax5.set_ylim(0, 1.0)
    ax5.grid(True, alpha=0.25)
    ax5.legend(fontsize=8, loc="lower left")
    ax5.tick_params(colors="#AAAAAA")

    ax6.set_facecolor("#111111")
    ax6.axis("off")
    summary = (
        f"VERIFY SUMMARY\n\n"
        f"Common days: {len(common_days)}\n"
        f"Theiler: {THEILER}\n"
        f"kNN k: {args.k}\n\n"
        f"Jaccard median / mean\n"
        f"  main raw/ens-smoothed: {cross_main['overlap_median']:.4f} / {cross_main['overlap_mean']:.4f}\n"
        f"  sensitivity raw/raw: {cross_raw['overlap_median']:.4f} / {cross_raw['overlap_mean']:.4f}\n"
        f"  baseline: {cross_main['baseline_median']:.4f} / {cross_main['baseline_mean']:.4f}\n\n"
        f"Kaplan-Glass final grid (16), raw only\n"
        f"  residuals raw: {kg_res_raw['scores'][-1]:.4f}\n"
        f"  ensemble-n raw: {kg_ens_raw['scores'][-1]:.4f}\n"
        f"  residuals shuffle: {kg_res_raw['shuffle_mean'][-1]:.4f}\n"
        f"  ensemble-n shuffle: {kg_ens_raw['shuffle_mean'][-1]:.4f}\n\n"
        f"Note: KG on pc_s disabled.\n"
        f"Gaussian smoothing inflates\n"
        f"directional coherence trivially.\n"
    )
    ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, va="top", ha="left",
             color="#DDDDDD", fontsize=9, linespacing=1.05,
             fontfamily="monospace", clip_on=True)

    fig.suptitle("BTC attractor verification: Residuen vs ensemble-n", color="#DDDDDD", fontsize=13, y=0.98)

    if args.save:
        save_path = Path(args.save)
        if save_path.parent != Path(""):
            save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
