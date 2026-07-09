"""Plots fuer die Wang/Nambu-Zerlegung auf dem realen Ensemble-n-Attraktor."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from attractor_n_ens.attractor_n_ens_nambu_Modul01 import dominant_terms


def _set_window_title(fig: plt.Figure, title: str) -> None:
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass


def _style_3d(ax) -> None:
    ax.set_facecolor("#1A1A1A")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#444444")
    ax.tick_params(colors="#CCCCCC")
    ax.xaxis.label.set_color("#CCCCCC")
    ax.yaxis.label.set_color("#CCCCCC")
    ax.zaxis.label.set_color("#CCCCCC")


def _style_2d(ax) -> None:
    ax.set_facecolor("#1A1A1A")
    ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.5)
    ax.tick_params(colors="#CCCCCC")
    ax.xaxis.label.set_color("#E0E0E0")
    ax.yaxis.label.set_color("#E0E0E0")


def _time_colored_segments(points: np.ndarray, colors: np.ndarray) -> Line3DCollection:
    segs = np.stack([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segs, cmap=plt.cm.coolwarm, linewidth=2.0, alpha=0.90)
    lc.set_array(colors[:-1])
    return lc


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-12:
        return values * 0.0
    return (values - np.mean(values)) / std


def _term_lines(name: str, terms: list[tuple[str, float]]) -> list[str]:
    if not terms:
        return [f"{name}: 0"]
    lines = [f"{name} dominant:"]
    lines.extend(f"  {coef:+.4f} {label}" for label, coef in terms[:4])
    return lines


def _event_series(
    days: np.ndarray,
    values: np.ndarray,
    event_days,
    labels,
    colors,
    max_day_gap: float = 90.0,
) -> list[tuple[float, float, str, str]]:
    events = []
    if event_days is None:
        return events
    for day, label, color in zip(event_days, labels, colors):
        day = float(day)
        if day > float(days[-1]):
            events.append((day, float(values[-1]), label, color))
            continue
        if day < float(days[0]):
            events.append((day, float(values[0]), label, color))
            continue

        idx = int(np.argmin(np.abs(days - day)))
        if np.abs(days[idx] - day) <= max_day_gap:
            events.append((day, float(values[idx]), label, color))
    return events


def _add_event_guides(ax, event_days, colors, style=":", alpha=0.30, linewidth=0.9) -> None:
    if event_days is None:
        return
    for day, color in zip(event_days, colors):
        ax.axvline(day, color=color, linestyle=style, linewidth=linewidth, alpha=alpha)


def _mark_events(ax, days: np.ndarray, values: np.ndarray, event_days, labels, colors, marker: str) -> None:
    for day, value, label, color in _event_series(days, values, event_days, labels, colors):
        ax.scatter(day, value, color=color, marker=marker, s=58, zorder=12, edgecolors="white", linewidths=1.0)
        ax.annotate(label, (day, value), textcoords="offset points", xytext=(5, 5), color=color, fontsize=8, fontweight="bold")


def _valid_time_anchor(days: np.ndarray, target_day: float, max_day_gap: float = 200.0) -> tuple[int, bool]:
    idx = int(np.argmin(np.abs(days - target_day)))
    return idx, bool(np.abs(days[idx] - target_day) <= max_day_gap)


def build_nambu_overview(data: dict, nambu) -> plt.Figure:
    points = data["pc_s"][:, :3]
    days = data["days_vecs"]
    halving_days = data["halving_days"]
    t_norm = (days - days[0]) / max(days[-1] - days[0], 1.0)

    fig = plt.figure(figsize=(18, 10))
    _set_window_title(fig, "Ensemble-n Nambu Overview")

    ax3d = fig.add_axes([0.04, 0.08, 0.56, 0.84], projection="3d")
    ax_h = fig.add_axes([0.65, 0.56, 0.31, 0.32])
    ax_text = fig.add_axes([0.65, 0.09, 0.31, 0.37])

    _style_3d(ax3d)
    _style_2d(ax_h)
    ax_text.set_facecolor("#1A1A1A")
    ax_text.set_axis_off()

    ax3d.add_collection3d(_time_colored_segments(points, t_norm))

    for arr, setter in ((points[:, 0], ax3d.set_xlim), (points[:, 1], ax3d.set_ylim), (points[:, 2], ax3d.set_zlim)):
        pad = 0.06 * (arr.max() - arr.min())
        setter(arr.min() - pad, arr.max() + pad)

    halving_lines = []
    for i, hday in enumerate(halving_days):
        idx, ok = _valid_time_anchor(days, hday)
        if not ok:
            continue
        halving_lines.append(f"H{i+1}: day={int(days[idx])}")
        ax3d.scatter(
            points[idx, 0],
            points[idx, 1],
            points[idx, 2],
            color="white",
            s=62,
            edgecolors="black",
            linewidths=1.0,
            zorder=10,
        )
        ax3d.text(points[idx, 0], points[idx, 1], points[idx, 2], f" H{i+1}", color="white", fontsize=8)

    scatter = ax_h.scatter(
        nambu.h1.values,
        nambu.h2.values,
        c=t_norm,
        cmap=plt.cm.coolwarm,
        s=11,
        alpha=0.82,
        edgecolors="none",
    )
    for i, hday in enumerate(halving_days):
        idx, ok = _valid_time_anchor(days, hday)
        if not ok:
            continue
        ax_h.scatter(nambu.h1.values[idx], nambu.h2.values[idx], color="white", s=46, edgecolors="black", linewidths=1.0)
        ax_h.annotate(f"H{i+1}", (nambu.h1.values[idx], nambu.h2.values[idx]), textcoords="offset points", xytext=(5, 4), color="white", fontsize=8)
    cbar = fig.colorbar(scatter, ax=ax_h, fraction=0.046, pad=0.04)
    cbar.set_label("normalized time", color="#E0E0E0")
    cbar.ax.tick_params(colors="#CCCCCC")

    ax3d.set_xlabel("PC1")
    ax3d.set_ylabel("PC2")
    ax3d.set_zlabel("PC3")
    ax_h.set_xlabel("H1")
    ax_h.set_ylabel("H2")
    ax_h.set_title("Quasi-invariant plane", color="#CCCCCC", fontsize=10)

    tail = nambu.singular_values[-4:]
    text_lines = [
        "Wang/Nambu split on real ensemble-n flow",
        "",
        "Centered coordinates:",
        f"u = PC1 - {nambu.center[0]:+.4f}",
        f"v = PC2 - {nambu.center[1]:+.4f}",
        f"w = PC3 - {nambu.center[2]:+.4f}",
        "",
        "Divergence fit:",
        f"div F = {nambu.divergence_coef[0]:+.4f}"
        f" {nambu.divergence_coef[1]:+.4f}u"
        f" {nambu.divergence_coef[2]:+.4f}v"
        f" {nambu.divergence_coef[3]:+.4f}w",
        "",
        f"cos(N,F_ND) med/p10 = {np.nanmedian(nambu.align_cos):+.3f} / {np.nanquantile(nambu.align_cos, 0.10):+.3f}",
        f"rel resid med/p90  = {np.nanmedian(nambu.rel_residual):+.3f} / {np.nanquantile(nambu.rel_residual, 0.90):+.3f}",
        f"ND drift H1/H2     = {nambu.drift_score_h1_nd:.3f} / {nambu.drift_score_h2_nd:.3f}",
        f"Full drift H1/H2   = {nambu.drift_score_h1_full:.3f} / {nambu.drift_score_h2_full:.3f}",
        "",
        f"smallest singulars = {tail[-2]:.3f}, {tail[-1]:.3f}",
        "",
        *_term_lines("H1", dominant_terms(nambu.h1)),
        "",
        *_term_lines("H2", dominant_terms(nambu.h2)),
        "",
        "Halving anchors:",
        *halving_lines[:5],
    ]

    ax_text.text(
        0.04,
        0.96,
        "\n".join(text_lines),
        va="top",
        ha="left",
        color="#DDDDDD",
        fontsize=10,
        family="monospace",
    )

    with plt.rc_context({"text.usetex": False}):
        plt.suptitle(
            "Real Ensemble-n: Wang/Nambu Decomposition",
            color="#CCCCCC",
            fontsize=13,
            y=0.985,
            fontname="Comfortaa",
            fontweight="bold",
        )
    plt.subplots_adjust(top=0.93)
    return fig


def build_nambu_diagnostics(data: dict, nambu) -> plt.Figure:
    days = data["days_vecs"]
    halving_days = data["halving_days"]
    peak_days = data.get("cycle_peak_days")
    peak_labels = data.get("cycle_peak_labels", [])
    peak_colors = data.get("cycle_peak_colors", [])
    bottom_days = data.get("cycle_bottom_days")
    bottom_labels = data.get("cycle_bottom_labels", [])
    bottom_colors = data.get("cycle_bottom_colors", [])
    event_candidates = [float(days[-1])]
    if peak_days is not None and len(peak_days):
        event_candidates.append(float(np.max(peak_days)))
    if bottom_days is not None and len(bottom_days):
        event_candidates.append(float(np.max(bottom_days)))
    x_max = max(event_candidates) + 30.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex="col")
    _set_window_title(fig, "Ensemble-n Nambu Diagnostics")
    axes = axes.ravel()

    for ax in axes:
        _style_2d(ax)

    ax = axes[0]
    ax.plot(days, _zscore(nambu.h1.values), color="#55CCFF", linewidth=1.1, label="H1")
    ax.plot(days, _zscore(nambu.h2.values), color="#FFAA55", linewidth=1.1, label="H2")
    ax.plot(days, _zscore(nambu.potential_values), color="#CC66FF", linewidth=1.0, alpha=0.90, label="D")
    for hday in halving_days:
        ax.axvline(hday, color="#777777", linestyle="--", linewidth=0.8, alpha=0.7)
    _add_event_guides(ax, peak_days, peak_colors)
    _add_event_guides(ax, bottom_days, bottom_colors)
    ax.set_ylabel("standardized value")
    ax.set_title("H1, H2, D along real BTC trajectory", color="#CCCCCC", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, facecolor="#1A1A1A", edgecolor="#808080", labelcolor="#E0E0E0")

    ax = axes[1]
    ax.plot(days, nambu.drift_h1_full, color="#FF6666", linewidth=1.0, alpha=0.90, label="dH1/dt full")
    ax.plot(days, nambu.drift_h1_nd, color="#66FF99", linewidth=1.0, alpha=0.90, label="dH1/dt F_ND")
    ax.plot(days, nambu.drift_h1_diss, color="#FFD166", linewidth=1.1, alpha=0.95, label="dH1/dt ∇D")
    for hday in halving_days:
        ax.axvline(hday, color="#777777", linestyle="--", linewidth=0.8, alpha=0.7)
    _add_event_guides(ax, peak_days, peak_colors)
    _add_event_guides(ax, bottom_days, bottom_colors)
    ax.set_ylabel("drift")
    ax.set_title("H1 drift", color="#CCCCCC", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, facecolor="#1A1A1A", edgecolor="#808080", labelcolor="#E0E0E0")

    ax = axes[2]
    ax.plot(days, nambu.drift_h2_full, color="#FF8888", linewidth=1.0, alpha=0.90, label="dH2/dt full")
    ax.plot(days, nambu.drift_h2_nd, color="#55DD88", linewidth=1.0, alpha=0.90, label="dH2/dt F_ND")
    ax.plot(days, nambu.drift_h2_diss, color="#FFD166", linewidth=1.1, alpha=0.95, label="dH2/dt ∇D")
    for hday in halving_days:
        ax.axvline(hday, color="#777777", linestyle="--", linewidth=0.8, alpha=0.7)
    _add_event_guides(ax, peak_days, peak_colors)
    _add_event_guides(ax, bottom_days, bottom_colors)
    _mark_events(ax, days, nambu.drift_h2_diss, peak_days, peak_labels, peak_colors, marker="^")
    _mark_events(ax, days, nambu.drift_h2_diss, bottom_days, bottom_labels, bottom_colors, marker="s")
    ax.set_ylabel("drift")
    ax.set_title("H2 drift", color="#CCCCCC", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, facecolor="#1A1A1A", edgecolor="#808080", labelcolor="#E0E0E0")

    ax = axes[3]
    ax.plot(days, nambu.align_cos, color="#66CCFF", linewidth=1.1, label="cos(F_ND, ∇H1×∇H2)")
    for hday in halving_days:
        ax.axvline(hday, color="#777777", linestyle="--", linewidth=0.8, alpha=0.7)
    _add_event_guides(ax, peak_days, peak_colors)
    _add_event_guides(ax, bottom_days, bottom_colors)
    ax.set_ylabel("alignment")
    ax.set_title("Nambu alignment and residual", color="#CCCCCC", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(days, nambu.rel_residual, color="#FFBB55", linewidth=1.0, alpha=0.95, label="relative residual")
    ax2.tick_params(colors="#CCCCCC")
    ax2.yaxis.label.set_color("#E0E0E0")
    ax2.set_ylabel("relative residual")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=8,
        facecolor="#1A1A1A",
        edgecolor="#808080",
        labelcolor="#E0E0E0",
    )

    axes[2].set_xlabel("Day since Genesis")
    axes[3].set_xlabel("Day since Genesis")
    for ax in axes:
        ax.set_xlim(float(days[0]), x_max)

    with plt.rc_context({"text.usetex": False}):
        plt.suptitle(
            "Real Ensemble-n: Quasi-Invariant Drift and Nambu Alignment",
            color="#CCCCCC",
            fontsize=13,
            y=0.985,
            fontname="Comfortaa",
            fontweight="bold",
        )
    plt.subplots_adjust(top=0.93, hspace=0.25, wspace=0.18)
    return fig
