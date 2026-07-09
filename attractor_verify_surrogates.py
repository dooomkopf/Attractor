#!/usr/bin/env python3
"""
Surrogate-Haertetest fuer die BTC-Attraktorfrage.

Verglichen werden die gleichen Rekonstruktionen wie in attractor_verify.py:
  1) Residuen-Attraktor (raw)
  2) ensemble-n-Attraktor (smoothed view fuer Cross-Observable, raw fuer KG)

Darauf vier Surrogatfamilien:
  A) FT auf Residuen
  B) IAAFT auf Residuen
  C) FT auf ensemble-n
  D) IAAFT auf ensemble-n

Keine p-Werte. Es werden nur Originalwerte gegen Surrogat-Verteilungen gestellt.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-attractor")
import matplotlib.pyplot as plt

import attractor_verify as av
from single_opt_FT import ft_surrogate
from single_opt_IAAFT import iaaft_surrogate


N_SURR = 12


def _embed_residual_signal(log_res, days_emb):
    n = len(log_res)
    w = (av.M - 1) * av.TAU
    d = np.empty((n - w, av.M))
    for j in range(av.M):
        d[:, j] = log_res[w - j * av.TAU : n - j * av.TAU]
    if av.NORMALIZE_WINDOWS:
        d -= d.mean(axis=1, keepdims=True)

    d_c = d - d.mean(axis=0)
    _, _, vt = np.linalg.svd(d_c, full_matrices=False)
    pc = d_c @ vt.T
    pc_s = pc.copy()
    for j in range(pc_s.shape[1]):
        pc_s[:, j] = av.gaussian_filter1d(pc[:, j], sigma=av.SMOOTH_SIGMA)
    return {"pc": pc, "pc_s": pc_s, "days_vecs": days_emb[w:]}


def _embed_ensemble_signal(sig, days_emb):
    n = len(sig)
    w = (av.M - 1) * av.TAU
    d = np.empty((n - w, av.M))
    for j in range(av.M):
        d[:, j] = sig[w - j * av.TAU : n - j * av.TAU]

    d_c = d - d.mean(axis=0)
    _, _, vt = np.linalg.svd(d_c, full_matrices=False)
    pc = d_c @ vt.T
    pc_s = pc.copy()
    for j in range(pc_s.shape[1]):
        pc_s[:, j] = av.gaussian_filter1d(pc[:, j], sigma=av.SMOOTH_SIGMA)

    pc[:, 1] *= -1.0
    pc_s[:, 1] *= -1.0

    c, s = np.cos(av.PHASE_OFFSET), np.sin(av.PHASE_OFFSET)
    for arr in (pc, pc_s):
        pc1 = arr[:, 0] * c - arr[:, 1] * s
        pc2 = arr[:, 0] * s + arr[:, 1] * c
        arr[:, 0], arr[:, 1] = pc1, pc2

    pc[:, 2] *= -1.0
    pc_s[:, 2] *= -1.0
    return {"pc": pc, "pc_s": pc_s, "days_vecs": days_emb[w:]}


def _cross_score(points_a, points_b, k=av.N_KNN, theiler=av.THEILER):
    a_std = av.zscore_columns(points_a)
    b_std = av.zscore_columns(points_b)
    dist_a = av.cdist(a_std, a_std).astype(np.float32, copy=False)
    dist_b = av.cdist(b_std, b_std).astype(np.float32, copy=False)
    sets_a, valid_a = av._knn_sets_from_distance(dist_a, k, theiler)
    sets_b, valid_b = av._knn_sets_from_distance(dist_b, k, theiler)
    valid = valid_a & valid_b
    overlap = np.array([av._jaccard(sets_a[i], sets_b[i]) for i in range(len(valid))], dtype=float)
    overlap = overlap[valid]
    return {
        "mean": float(np.nanmean(overlap)),
        "median": float(np.nanmedian(overlap)),
    }


def _kg_scores(points):
    return np.array([av.kaplan_glass_score(points, g) for g in av.KG_GRIDS], dtype=float)


def _quantile_band(arr):
    arr = np.asarray(arr, dtype=float)
    return (
        np.nanmedian(arr, axis=0),
        np.nanquantile(arr, 0.10, axis=0),
        np.nanquantile(arr, 0.90, axis=0),
    )


def _print_family(label, cross_means, kg_curves):
    cross_means = np.asarray(cross_means, dtype=float)
    kg_curves = np.asarray(kg_curves, dtype=float)
    kg_med, _, _ = _quantile_band(kg_curves)
    print(
        f"{label}: cross mean median={np.nanmedian(cross_means):.4f}  "
        f"q10={np.nanquantile(cross_means, 0.10):.4f}  "
        f"q90={np.nanquantile(cross_means, 0.90):.4f}"
    )
    print(f"{label}: KG median curve  {av._fmt_curve(kg_med)}")


def _plot_strip(ax, data_dict, original_main, original_raw, baseline_q90):
    ax.set_facecolor("#1A1A1A")
    labels = list(data_dict.keys())
    colors = ["#4FD9E8", "#4FD9E8", "#FFAA55", "#FFAA55"]
    markers = ["o", "s", "o", "s"]
    rng = np.random.default_rng(12345)
    for i, (label, values) in enumerate(data_dict.items(), start=1):
        values = np.asarray(values, dtype=float)
        x = i + rng.uniform(-0.12, 0.12, size=len(values))
        ax.scatter(x, values, s=28, alpha=0.75, color=colors[i - 1], marker=markers[i - 1], label=label if i == 1 else "_")
        med = float(np.nanmedian(values))
        ax.plot([i - 0.18, i + 0.18], [med, med], color="white", linewidth=2.0)

    ax.axhspan(0.0, baseline_q90, color="#999999", alpha=0.14, label="Original baseline q90")
    ax.axhline(original_main, color="#4FD9E8", linestyle="--", linewidth=1.4, label="Original main")
    ax.axhline(original_raw, color="#FFAA55", linestyle=":", linewidth=1.4, label="Original raw/raw")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Cross mean Jaccard", color="#AAAAAA")
    ax.set_title("Cross-Observable Consistency vs Surrogates", color="#CCCCCC", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(colors="#AAAAAA")


def _plot_kg_panel(ax, title, orig_scores, orig_shuffle, family_a, family_b, color_a, color_b):
    ax.set_facecolor("#1A1A1A")
    grids = np.array(av.KG_GRIDS, dtype=int)
    ax.plot(grids, orig_scores, color="white", linewidth=1.8, marker="o", label="Original")
    ax.plot(grids, orig_shuffle["shuffle_mean"], color="#AAAAAA", linestyle=":", linewidth=1.2, label="Original shuffle")
    ax.fill_between(grids, orig_shuffle["shuffle_q10"], orig_shuffle["shuffle_q90"], color="#AAAAAA", alpha=0.10)

    med_a, q10_a, q90_a = _quantile_band(family_a)
    med_b, q10_b, q90_b = _quantile_band(family_b)
    ax.plot(grids, med_a, color=color_a, linewidth=1.6, marker="o", label="FT median")
    ax.fill_between(grids, q10_a, q90_a, color=color_a, alpha=0.10)
    ax.plot(grids, med_b, color=color_b, linewidth=1.6, marker="s", label="IAAFT median")
    ax.fill_between(grids, q10_b, q90_b, color=color_b, alpha=0.10)

    ax.set_title(title, color="#CCCCCC", fontsize=10)
    ax.set_xlabel("Grid size", color="#AAAAAA")
    ax.set_ylabel("KG score", color="#AAAAAA")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.tick_params(colors="#AAAAAA")
    ax.legend(fontsize=8, loc="lower left")


def main():
    ap = argparse.ArgumentParser(description="BTC-Attraktor: Verify gegen FT/IAAFT-Surrogates")
    ap.add_argument("--n-surr", type=int, default=N_SURR, help="Anzahl Surrogates pro Familie")
    ap.add_argument("--k", type=int, default=av.N_KNN, help="k fuer Cross-Observable-kNN")
    ap.add_argument("--seed", type=int, default=42, help="Zufalls-Seed")
    ap.add_argument("--save", type=str, default=None, help="Optionaler PNG-Ausgabepfad")
    ap.add_argument("--no-show", action="store_true", help="Plot nicht interaktiv anzeigen")
    args = ap.parse_args()

    res = av._build_residual_embedding()
    ens = av._build_ensemble_embedding()
    common_days, res_idx, ens_idx = av._common_alignment(res, ens)

    res_common = res["pc"][res_idx, :3]
    ens_common = ens["pc"][ens_idx, :3]
    ens_common_s = ens["pc_s"][ens_idx, :3]

    cross_main = av.cross_observable_consistency(res_common, ens_common_s, k=args.k, theiler=av.THEILER, seed=args.seed)
    cross_raw = av.cross_observable_consistency(res_common, ens_common, k=args.k, theiler=av.THEILER, seed=args.seed)
    kg_res_orig = av.kaplan_glass_curve(res_common, av.KG_GRIDS, n_shuffle=av.N_KG_BASELINE, seed=args.seed)
    kg_ens_orig = av.kaplan_glass_curve(ens_common, av.KG_GRIDS, n_shuffle=av.N_KG_BASELINE, seed=args.seed + 1)

    res_mask = res["days_all"] >= av.START_IDX
    res_sig = res["log_res_all"][res_mask]
    res_days_emb = res["days_all"][res_mask]

    half_max = max(av.WINDOW_SIZES) // 2
    ens_mask = (ens["days_n_all"] >= av.START_IDX - half_max) & np.isfinite(ens["daily_n_all"])
    ens_sig = ens["daily_n_all"][ens_mask]
    ens_days_emb = ens["days_n_all"][ens_mask]

    rng_res_ft = np.random.default_rng(args.seed + 10)
    rng_res_iaaft = np.random.default_rng(args.seed + 20)
    rng_ens_ft = np.random.default_rng(args.seed + 30)
    rng_ens_iaaft = np.random.default_rng(args.seed + 40)

    res_ft_cross, res_iaaft_cross = [], []
    ens_ft_cross, ens_iaaft_cross = [], []
    res_ft_kg, res_iaaft_kg = [], []
    ens_ft_kg, ens_iaaft_kg = [], []

    print(f"\n=== Surrogate Verify ===")
    print(f"Common days: {len(common_days)}")
    print(f"Original main raw/ens-smoothed mean: {cross_main['overlap_mean']:.4f}")
    print(f"Original raw/raw mean: {cross_raw['overlap_mean']:.4f}")
    print(f"Original baseline q90: {cross_main['baseline_quantiles'][2]:.4f}")
    print(f"Surrogates per family: {args.n_surr}")

    for _ in range(args.n_surr):
        res_ft_sig = ft_surrogate(res_sig, rng_res_ft)
        res_ft_emb = _embed_residual_signal(res_ft_sig, res_days_emb)
        res_ft_cross.append(_cross_score(res_ft_emb["pc"][res_idx, :3], ens_common_s, k=args.k, theiler=av.THEILER)["mean"])
        res_ft_kg.append(_kg_scores(res_ft_emb["pc"][res_idx, :3]))

        res_iaaft_sig = iaaft_surrogate(res_sig, rng_res_iaaft)
        res_iaaft_emb = _embed_residual_signal(res_iaaft_sig, res_days_emb)
        res_iaaft_cross.append(_cross_score(res_iaaft_emb["pc"][res_idx, :3], ens_common_s, k=args.k, theiler=av.THEILER)["mean"])
        res_iaaft_kg.append(_kg_scores(res_iaaft_emb["pc"][res_idx, :3]))

        ens_ft_sig = ft_surrogate(ens_sig, rng_ens_ft)
        ens_ft_emb = _embed_ensemble_signal(ens_ft_sig, ens_days_emb)
        ens_ft_cross.append(_cross_score(res_common, ens_ft_emb["pc_s"][ens_idx, :3], k=args.k, theiler=av.THEILER)["mean"])
        ens_ft_kg.append(_kg_scores(ens_ft_emb["pc"][ens_idx, :3]))

        ens_iaaft_sig = iaaft_surrogate(ens_sig, rng_ens_iaaft)
        ens_iaaft_emb = _embed_ensemble_signal(ens_iaaft_sig, ens_days_emb)
        ens_iaaft_cross.append(_cross_score(res_common, ens_iaaft_emb["pc_s"][ens_idx, :3], k=args.k, theiler=av.THEILER)["mean"])
        ens_iaaft_kg.append(_kg_scores(ens_iaaft_emb["pc"][ens_idx, :3]))

    _print_family("Residual FT", res_ft_cross, res_ft_kg)
    _print_family("Residual IAAFT", res_iaaft_cross, res_iaaft_kg)
    _print_family("ensemble-n FT", ens_ft_cross, ens_ft_kg)
    _print_family("ensemble-n IAAFT", ens_iaaft_cross, ens_iaaft_kg)

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("black")
    ax1, ax2, ax3, ax4 = axs.ravel()

    _plot_strip(
        ax1,
        {
            "FT res": res_ft_cross,
            "IAAFT res": res_iaaft_cross,
            "FT ens": ens_ft_cross,
            "IAAFT ens": ens_iaaft_cross,
        },
        original_main=cross_main["overlap_mean"],
        original_raw=cross_raw["overlap_mean"],
        baseline_q90=cross_main["baseline_quantiles"][2],
    )
    ax1.legend(fontsize=8, loc="upper right")

    ax2.set_facecolor("#111111")
    ax2.axis("off")
    summary = (
        f"SURROGATE VERIFY\n\n"
        f"Common days: {len(common_days)}\n"
        f"N surrogates/family: {args.n_surr}\n"
        f"kNN k: {args.k}\n\n"
        f"Original cross mean\n"
        f"  main raw/ens-smoothed: {cross_main['overlap_mean']:.4f}\n"
        f"  sensitivity raw/raw:   {cross_raw['overlap_mean']:.4f}\n"
        f"  baseline q90:          {cross_main['baseline_quantiles'][2]:.4f}\n\n"
        f"Median surrogate cross mean\n"
        f"  FT res:      {np.nanmedian(res_ft_cross):.4f}\n"
        f"  IAAFT res:   {np.nanmedian(res_iaaft_cross):.4f}\n"
        f"  FT ens:      {np.nanmedian(ens_ft_cross):.4f}\n"
        f"  IAAFT ens:   {np.nanmedian(ens_iaaft_cross):.4f}\n\n"
        f"Note:\n"
        f"  Cross uses raw residuals vs smoothed ensemble-n.\n"
        f"  KG uses raw PCs only.\n"
    )
    ax2.text(
        0.02, 0.98, summary, transform=ax2.transAxes,
        va="top", ha="left", color="#DDDDDD",
        fontsize=9, linespacing=1.05, fontfamily="monospace", clip_on=True,
    )

    _plot_kg_panel(
        ax3,
        "Kaplan-Glass on Residual Attractor (raw)",
        kg_res_orig["scores"],
        kg_res_orig,
        np.asarray(res_ft_kg, dtype=float),
        np.asarray(res_iaaft_kg, dtype=float),
        "#4FD9E8",
        "#FF66AA",
    )
    _plot_kg_panel(
        ax4,
        "Kaplan-Glass on ensemble-n Attractor (raw)",
        kg_ens_orig["scores"],
        kg_ens_orig,
        np.asarray(ens_ft_kg, dtype=float),
        np.asarray(ens_iaaft_kg, dtype=float),
        "#FFAA55",
        "#88FF88",
    )

    fig.suptitle("BTC attractor verification against FT / IAAFT surrogates", color="#DDDDDD", fontsize=13, y=0.98)

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
