#!/usr/bin/env python3
"""Wang/Nambu-Zerlegung des rekonstruierten realen Ensemble-n-Attraktors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attractor_n_ens.attractor_n_ens_utils import read_btc_data
from attractor_n_ens.attractor_n_ens_compute import compute_all
from attractor_n_ens.attractor_n_ens_flow_Modul01 import fit_quadratic_flow
from attractor_n_ens.attractor_n_ens_nambu_Modul01 import (
    build_nambu_model,
    dominant_terms,
    format_potential_expression,
    format_scalar_expression,
)
from attractor_n_ens.attractor_n_ens_nambu_Modul02 import build_nambu_diagnostics, build_nambu_overview

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
POLY_DEGREE = 3
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


def load_cycle_events(days_all, dates_all) -> dict:
    json_path = Path(__file__).resolve().parents[2] / "gold" / "cycles.json"
    if not json_path.exists():
        return {
            "cycle_peak_days": np.array([], dtype=float),
            "cycle_peak_labels": [],
            "cycle_peak_colors": [],
            "cycle_bottom_days": np.array([], dtype=float),
            "cycle_bottom_labels": [],
            "cycle_bottom_colors": [],
        }

    payload = json.loads(json_path.read_text())

    def nearest_day(date_string: str) -> float:
        dt = datetime.strptime(date_string, "%d.%m.%Y")
        diffs = np.array([abs((item - dt).days) for item in dates_all], dtype=float)
        return float(days_all[int(np.argmin(diffs))])

    peak_days: list[float] = []
    peak_labels: list[str] = []
    peak_colors: list[str] = []
    peak_counter = 1
    for period in payload.get("halving_periods", []):
        for peak in period.get("peaks", []):
            if peak.get("implemented") != "yes":
                continue
            peak_days.append(nearest_day(peak["date"]))
            peak_labels.append(f"T{peak_counter}")
            peak_colors.append(peak.get("color", "orange"))
            peak_counter += 1

    bottom_color_map = {
        "2013": "#0000FF",
        "2017": "#90EE90",
        "2021": "#FF69B4",
        "2025": "orange",
    }
    bottom_days: list[float] = []
    bottom_labels: list[str] = []
    bottom_colors: list[str] = []
    for bottom in payload.get("cycle_bottoms", []):
        bottom_days.append(nearest_day(bottom["date"]))
        bottom_labels.append(f"B{bottom.get('nr', len(bottom_labels) + 1)}")
        cycle_key = str(bottom.get("cycle_after", ""))[:4]
        bottom_colors.append(bottom_color_map.get(cycle_key, "#DDDDDD"))

    return {
        "cycle_peak_days": np.array(peak_days, dtype=float),
        "cycle_peak_labels": peak_labels,
        "cycle_peak_colors": peak_colors,
        "cycle_bottom_days": np.array(bottom_days, dtype=float),
        "cycle_bottom_labels": bottom_labels,
        "cycle_bottom_colors": bottom_colors,
    }


def merge_cycle_events(data: dict, cycle_events: dict) -> dict:
    merged = dict(cycle_events)

    peak_days = list(map(float, merged.get("cycle_peak_days", [])))
    peak_labels = list(merged.get("cycle_peak_labels", []))
    peak_colors = list(merged.get("cycle_peak_colors", []))
    for idx, (day, color) in enumerate(zip(data.get("used_peak_days", []), data.get("peak_colors", [])), start=1):
        day = float(day)
        if any(abs(day - existing) <= 2.0 for existing in peak_days):
            continue
        peak_days.append(day)
        peak_labels.append(f"T{idx}")
        peak_colors.append(color)

    bottom_days = list(map(float, merged.get("cycle_bottom_days", [])))
    bottom_labels = list(merged.get("cycle_bottom_labels", []))
    bottom_colors = list(merged.get("cycle_bottom_colors", []))
    for idx, (day, color) in enumerate(zip(data.get("used_bottom_days", []), data.get("bottom_colors", [])), start=1):
        day = float(day)
        if any(abs(day - existing) <= 2.0 for existing in bottom_days):
            continue
        bottom_days.append(day)
        bottom_labels.append(f"B{idx}")
        bottom_colors.append(color)

    merged["cycle_peak_days"] = np.array(peak_days, dtype=float)
    merged["cycle_peak_labels"] = peak_labels
    merged["cycle_peak_colors"] = peak_colors
    merged["cycle_bottom_days"] = np.array(bottom_days, dtype=float)
    merged["cycle_bottom_labels"] = bottom_labels
    merged["cycle_bottom_colors"] = bottom_colors
    return merged


def print_summary(nambu) -> None:
    tail = nambu.singular_values[-6:]
    print("Wang/Nambu decomposition on real ensemble-n PCA")
    print(f"degree = {POLY_DEGREE}")
    print(f"center = ({nambu.center[0]:+.5f}, {nambu.center[1]:+.5f}, {nambu.center[2]:+.5f})")
    print(
        "div(F) = "
        f"{nambu.divergence_coef[0]:+.5f} "
        f"{nambu.divergence_coef[1]:+.5f}u "
        f"{nambu.divergence_coef[2]:+.5f}v "
        f"{nambu.divergence_coef[3]:+.5f}w"
    )
    print(f"D(u,v,w) ≈ {format_potential_expression(nambu.divergence_coef)}")
    print(f"smallest singulars = {', '.join(f'{value:.4f}' for value in tail)}")
    print(f"alignment median/p10 = {float(np.nanmedian(nambu.align_cos)):+.4f} / {float(np.nanquantile(nambu.align_cos, 0.10)):+.4f}")
    print(f"relative residual median/p90 = {float(np.nanmedian(nambu.rel_residual)):+.4f} / {float(np.nanquantile(nambu.rel_residual, 0.90)):+.4f}")
    print(f"ND drift scores   = H1 {nambu.drift_score_h1_nd:.4f}   H2 {nambu.drift_score_h2_nd:.4f}")
    print(f"Diss drift scores = H1 {nambu.drift_score_h1_diss:.4f}   H2 {nambu.drift_score_h2_diss:.4f}")
    print(f"Full drift scores = H1 {nambu.drift_score_h1_full:.4f}   H2 {nambu.drift_score_h2_full:.4f}")
    print(f"H1(u,v,w) ≈ {format_scalar_expression(nambu.h1)}")
    print(f"H2(u,v,w) ≈ {format_scalar_expression(nambu.h2)}")
    print("H1 dominant terms:", ", ".join(f"{coef:+.4f} {label}" for label, coef in dominant_terms(nambu.h1)))
    print("H2 dominant terms:", ", ".join(f"{coef:+.4f} {label}" for label, coef in dominant_terms(nambu.h2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Wang/Nambu-Zerlegung auf realem Ensemble-n-Attraktor")
    parser.add_argument("--save-prefix", type=str, default="", help="optionales PNG-Prefix")
    parser.add_argument("--no-show", action="store_true", help="Figures nicht interaktiv anzeigen")
    args = parser.parse_args()

    configure_style()
    params = make_params()
    data_path = Path(__file__).resolve().parent.parent / "ziel.csv"
    days_all, prices_all, dates_all = read_btc_data(str(data_path))
    data = compute_all(days_all, prices_all, dates_all, params)
    data.update(merge_cycle_events(data, load_cycle_events(days_all, dates_all)))

    points = data["pc_s"][:, :3]
    days = data["days_vecs"].astype(float)
    flow = fit_quadratic_flow(points, days, ridge=1e-6)
    nambu = build_nambu_model(flow, points, degree=POLY_DEGREE)
    print_summary(nambu)

    fig1 = build_nambu_overview(data, nambu)
    fig2 = build_nambu_diagnostics(data, nambu)

    if args.save_prefix:
        prefix = Path(args.save_prefix)
        fig1.savefig(f"{prefix}_overview.png", dpi=300, facecolor="#0a0a0a")
        fig2.savefig(f"{prefix}_diagnostics.png", dpi=300, facecolor="#0a0a0a")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
