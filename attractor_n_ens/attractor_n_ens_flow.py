#!/usr/bin/env python3
"""Explizite quadratische Fluss-Rekonstruktion auf dem realen Ensemble-n-Attraktor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attractor_n_ens.attractor_n_ens_utils import read_btc_data
from attractor_n_ens.attractor_n_ens_compute import compute_all
from attractor_n_ens.attractor_n_ens_flow_Modul01 import (
    build_seed_points,
    classify_fixed_point,
    find_fixed_points,
    fit_quadratic_flow,
    jacobian,
    trajectory_diagnostics,
)
from attractor_n_ens.attractor_n_ens_flow_Modul02 import build_flow_diagnostics, build_flow_overview

END_CUT = 0
START_IDX = 1164
TAU = 30
M = 50
SMOOTH_SIGMA = 60
LABEL_WINDOW = 30
SHOW_FIG4 = False
CMAP = plt.cm.coolwarm
WINDOW_SIZES = list(range(90, 181, 10))
PHASE_OFFSET = 0.0
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


def configure_style() -> None:
    try:
        plt.style.use("hz.mplstyle")
    except Exception:
        plt.rcParams.update(
            {
                "figure.facecolor": "#0a0a0a",
                "axes.facecolor": "#1A1A1A",
                "axes.edgecolor": "#CCCCCC",
                "axes.labelcolor": "#CCCCCC",
                "text.color": "#CCCCCC",
                "xtick.color": "#CCCCCC",
                "ytick.color": "#CCCCCC",
                "grid.color": "#666666",
                "legend.facecolor": "#1A1A1A",
                "legend.edgecolor": "#808080",
                "savefig.facecolor": "#0a0a0a",
            }
        )


def make_params() -> dict:
    return dict(
        END_CUT=END_CUT,
        START_IDX=START_IDX,
        TAU=TAU,
        M=M,
        SMOOTH_SIGMA=SMOOTH_SIGMA,
        LABEL_WINDOW=LABEL_WINDOW,
        CMAP=CMAP,
        WINDOW_SIZES=WINDOW_SIZES,
        PHASE_OFFSET=PHASE_OFFSET,
        HALVINGS=HALVINGS,
        CYCLE_TOPS=CYCLE_TOPS,
        CYCLE_BOTTOMS=CYCLE_BOTTOMS,
        SHOW_FIG4=SHOW_FIG4,
    )


def halving_local_info(data: dict, diagnostics: dict, flow) -> list[dict]:
    info = []
    for hday in data["halving_days"]:
        idx = int(np.argmin(np.abs(data["days_vecs"] - hday)))
        eigvals = np.linalg.eigvals(jacobian(flow, data["pc_s"][idx, :3]))
        info.append(
            {
                "day": float(data["days_vecs"][idx]),
                "eigvals": eigvals,
                "kind": classify_fixed_point(eigvals),
                "max_real": float(diagnostics["max_real"][idx]),
                "min_real": float(diagnostics["min_real"][idx]),
                "divergence": float(diagnostics["divergence"][idx]),
            }
        )
    return info


def print_summary(data: dict, flow, diagnostics: dict, fixed_points: list[dict], halving_info: list[dict]) -> None:
    print("Quadratic flow fit on real ensemble-n PCA")
    print(f"vectors={len(data['pc_s'])}  M={M}  tau={TAU}d  smooth_sigma={SMOOTH_SIGMA}")
    print(f"R2      = {flow.r2[0]:+.4f}, {flow.r2[1]:+.4f}, {flow.r2[2]:+.4f}")
    print(f"RMSE    = {flow.rmse[0]:+.5f}, {flow.rmse[1]:+.5f}, {flow.rmse[2]:+.5f}")
    print(f"dir_cos = median {np.median(flow.dir_cos):+.4f}   p10 {np.quantile(flow.dir_cos, 0.10):+.4f}")
    print(f"mean(div J) = {np.mean(diagnostics['divergence']):+.5f}")
    print(f"fixed points found = {len(fixed_points)}")
    for i, item in enumerate(fixed_points[:8], start=1):
        coords = ", ".join(f"{v:+.4f}" for v in item["point"])
        eigs = ", ".join(
            f"{np.real(ev):+.4f}" if abs(np.imag(ev)) < 1e-6 else f"{np.real(ev):+.4f}{np.imag(ev):+.4f}i"
            for ev in item["eigvals"]
        )
        print(f"  FP{i}: {item['kind']:>12}  cloud_d={item['distance_to_cloud']:.4f}  X=({coords})  eig=({eigs})")
    print("Halving local linearization:")
    for i, item in enumerate(halving_info, start=1):
        eigs = ", ".join(
            f"{np.real(ev):+.4f}" if abs(np.imag(ev)) < 1e-6 else f"{np.real(ev):+.4f}{np.imag(ev):+.4f}i"
            for ev in item["eigvals"]
        )
        print(
            f"  H{i}: {item['kind']:>12}  div={item['divergence']:+.4f}  "
            f"maxRe={item['max_real']:+.4f}  minRe={item['min_real']:+.4f}  eig=({eigs})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Quadratische Fluss-Rekonstruktion auf realem Ensemble-n-Attraktor")
    parser.add_argument("--save-prefix", type=str, default="", help="optionales PNG-Prefix")
    parser.add_argument("--no-show", action="store_true", help="Figures nicht interaktiv anzeigen")
    args = parser.parse_args()

    configure_style()
    params = make_params()
    data_path = Path(__file__).resolve().parent.parent / "ziel.csv"
    days_all, prices_all, dates_all = read_btc_data(str(data_path))
    data = compute_all(days_all, prices_all, dates_all, params)

    points = data["pc_s"][:, :3]
    days = data["days_vecs"].astype(float)

    flow = fit_quadratic_flow(points, days, ridge=1e-6)
    diagnostics = trajectory_diagnostics(flow, points)
    seeds = build_seed_points(points, days, data)
    fixed_points = find_fixed_points(flow, points, seeds)
    hinfo = halving_local_info(data, diagnostics, flow)
    print_summary(data, flow, diagnostics, fixed_points, hinfo)

    fig1 = build_flow_overview(data, flow, diagnostics, fixed_points)
    fig2 = build_flow_diagnostics(data, flow, diagnostics, hinfo)

    if args.save_prefix:
        prefix = Path(args.save_prefix)
        fig1.savefig(f"{prefix}_overview.png", dpi=300, facecolor="#0a0a0a")
        fig2.savefig(f"{prefix}_diagnostics.png", dpi=300, facecolor="#0a0a0a")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
