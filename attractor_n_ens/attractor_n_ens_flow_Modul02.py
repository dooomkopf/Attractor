"""Plots fuer die Fluss-Rekonstruktion auf dem realen Ensemble-n-Attraktor."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

_FIXED_COLORS = {
    "saddle": "#FF4444",
    "saddle-focus": "#FF8844",
    "sink": "#44CC44",
    "spiral-sink": "#66FF99",
    "source": "#FFDD44",
    "spiral-source": "#FFDDAA",
    "center": "white",
}


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
    lc = Line3DCollection(segs, cmap=plt.cm.coolwarm, linewidth=2.0, alpha=0.85)
    lc.set_array(colors[:-1])
    return lc


def _vector_segments(points: np.ndarray, vectors: np.ndarray, color: str, scale: float) -> Line3DCollection:
    segs = np.stack([points, points + scale * vectors], axis=1)
    return Line3DCollection(segs, colors=color, linewidth=1.4, alpha=0.95)


def _format_eigs(eigvals: np.ndarray) -> str:
    parts = []
    for eig in eigvals:
        if abs(np.imag(eig)) < 1e-6:
            parts.append(f"{np.real(eig):+.3f}")
        else:
            parts.append(f"{np.real(eig):+.3f}{np.imag(eig):+.3f}i")
    return ", ".join(parts)


def build_flow_overview(data: dict, flow, diagnostics: dict, fixed_points: list[dict]) -> plt.Figure:
    points = data["pc_s"][:, :3]
    days = data["days_vecs"]
    halving_days = data["halving_days"]
    t_norm = (days - days[0]) / max(days[-1] - days[0], 1.0)

    fig = plt.figure(figsize=(18, 10))
    _set_window_title(fig, "Ensemble-n Flow Overview")

    ax3d = fig.add_axes([0.04, 0.08, 0.58, 0.84], projection="3d")
    ax_text = fig.add_axes([0.67, 0.56, 0.29, 0.32])
    ax_proj = fig.add_axes([0.67, 0.10, 0.29, 0.34])

    _style_3d(ax3d)
    _style_2d(ax_proj)
    ax_text.set_facecolor("#1A1A1A")
    ax_text.set_axis_off()

    ax3d.add_collection3d(_time_colored_segments(points, t_norm))

    extent = np.max(points, axis=0) - np.min(points, axis=0)
    scale = 0.10 * np.linalg.norm(extent) / max(np.median(np.linalg.norm(flow.velocity, axis=1)), 1e-8)
    idx = np.linspace(0, len(points) - 1, min(50, len(points)), dtype=int)
    ax3d.add_collection3d(_vector_segments(points[idx], flow.velocity[idx], "#55CCFF", scale))
    ax3d.add_collection3d(_vector_segments(points[idx], flow.velocity_fit[idx], "#FFAA55", scale))

    for arr, setter in ((points[:, 0], ax3d.set_xlim), (points[:, 1], ax3d.set_ylim), (points[:, 2], ax3d.set_zlim)):
        pad = 0.06 * (arr.max() - arr.min())
        setter(arr.min() - pad, arr.max() + pad)

    halving_lines = []
    for i, hday in enumerate(halving_days):
        hidx = int(np.argmin(np.abs(days - hday)))
        halving_lines.append(f"H{i+1}: day={int(days[hidx])}")
        ax3d.scatter(
            points[hidx, 0],
            points[hidx, 1],
            points[hidx, 2],
            color="white",
            s=60,
            edgecolors="black",
            linewidths=1.0,
            zorder=10,
        )
        ax3d.text(points[hidx, 0], points[hidx, 1], points[hidx, 2], f" H{i+1}", color="white", fontsize=8)

    seen = set()
    for item in fixed_points:
        color = _FIXED_COLORS.get(item["kind"], "white")
        label = item["kind"] if item["kind"] not in seen else "_"
        seen.add(item["kind"])
        ax3d.scatter(
            item["point"][0],
            item["point"][1],
            item["point"][2],
            color=color,
            s=110,
            marker="X",
            edgecolors="black",
            linewidths=1.0,
            label=label,
            zorder=12,
        )

    ax3d.set_xlabel("PC1")
    ax3d.set_ylabel("PC2")
    ax3d.set_zlabel("PC3")
    ax3d.legend(loc="upper left", fontsize=8, facecolor="#1A1A1A", edgecolor="#808080", labelcolor="#E0E0E0")

    ax_proj.plot(points[:, 0], points[:, 1], color="#B0B0B0", linewidth=1.2, alpha=0.75)
    ax_proj.scatter(points[idx, 0], points[idx, 1], color="#55CCFF", s=14, alpha=0.9, label="real dX/dt")
    ax_proj.scatter(points[idx, 0], points[idx, 1], color="#FFAA55", s=8, alpha=0.9, label="fit dX/dt")
    for item in fixed_points:
        ax_proj.scatter(item["point"][0], item["point"][1], color=_FIXED_COLORS.get(item["kind"], "white"), s=70, marker="X")
    ax_proj.set_xlabel("PC1")
    ax_proj.set_ylabel("PC2")
    ax_proj.set_title("Projection PC1-PC2", color="#CCCCCC", fontsize=10)
    pad_x = 0.06 * (points[:, 0].max() - points[:, 0].min())
    pad_y = 0.06 * (points[:, 1].max() - points[:, 1].min())
    ax_proj.set_xlim(points[:, 0].min() - pad_x, points[:, 0].max() + pad_x)
    ax_proj.set_ylim(points[:, 1].min() - pad_y, points[:, 1].max() + pad_y)
    ax_proj.legend(loc="upper left", fontsize=8, facecolor="#1A1A1A", edgecolor="#808080", labelcolor="#E0E0E0")

    text_lines = [
        "Quadratic macro-flow on real ensemble-n PCA",
        "",
        f"R²:    {flow.r2[0]:.3f}   {flow.r2[1]:.3f}   {flow.r2[2]:.3f}",
        f"RMSE:  {flow.rmse[0]:.4f}  {flow.rmse[1]:.4f}  {flow.rmse[2]:.4f}",
        f"dir cos median: {np.median(flow.dir_cos):.3f}",
        f"dir cos p10:    {np.quantile(flow.dir_cos, 0.10):.3f}",
        f"mean div:       {np.mean(diagnostics['divergence']):+.4f}",
        f"fixed points:   {len(fixed_points)}",
        "",
        "Halving anchors:",
        *halving_lines[:5],
    ]
    if fixed_points:
        text_lines.extend(["", "Closest fixed point:", f"{fixed_points[0]['kind']}  d={fixed_points[0]['distance_to_cloud']:.3f}", _format_eigs(fixed_points[0]["eigvals"])])

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
            "Real Ensemble-n: Quadratic Flow Reconstruction",
            color="#CCCCCC",
            fontsize=13,
            y=0.985,
            fontname="Comfortaa",
            fontweight="bold",
        )
    plt.subplots_adjust(top=0.93)
    return fig


def build_flow_diagnostics(data: dict, flow, diagnostics: dict, halving_info: list[dict]) -> plt.Figure:
    days = data["days_vecs"]
    halving_days = data["halving_days"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex="col")
    _set_window_title(fig, "Ensemble-n Flow Diagnostics")
    axes = axes.ravel()

    labels = ["PC1", "PC2", "PC3"]
    for j, ax in enumerate(axes[:3]):
        _style_2d(ax)
        ax.plot(days, flow.velocity[:, j], color="#55CCFF", linewidth=1.0, alpha=0.9, label="real")
        ax.plot(days, flow.velocity_fit[:, j], color="#FFAA55", linewidth=1.0, alpha=0.85, label="quadratic fit")
        for hday in halving_days:
            ax.axvline(hday, color="#777777", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(f"d{labels[j]}/dt")
        ax.set_title(f"Velocity {labels[j]}", color="#CCCCCC", fontsize=10)
        ax.legend(loc="upper left", fontsize=8, facecolor="#1A1A1A", edgecolor="#808080", labelcolor="#E0E0E0")

    ax = axes[3]
    _style_2d(ax)
    ax.plot(days, diagnostics["divergence"], color="#DD6666", linewidth=1.2, label="div J")
    ax.plot(days, diagnostics["max_real"], color="#66FF66", linewidth=1.0, alpha=0.9, label="max Re λ")
    ax.plot(days, diagnostics["mid_real"], color="#DDDD66", linewidth=0.9, alpha=0.85, label="mid Re λ")
    ax.plot(days, diagnostics["min_real"], color="#66AAFF", linewidth=1.0, alpha=0.9, label="min Re λ")
    for i, hday in enumerate(halving_days):
        ax.axvline(hday, color="#777777", linestyle="--", linewidth=0.8, alpha=0.7)
        info = halving_info[i]
        ax.scatter(info["day"], info["max_real"], color="white", s=38, zorder=10, edgecolors="black", linewidths=1.0)
        ax.annotate(f"H{i+1}", (info["day"], info["max_real"]), textcoords="offset points", xytext=(5, 5), color="white", fontsize=8)
    ax.set_ylabel("local linearization")
    ax.set_title("Divergence and Jacobian spectrum", color="#CCCCCC", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, facecolor="#1A1A1A", edgecolor="#808080", labelcolor="#E0E0E0")

    axes[2].set_xlabel("Day since Genesis")
    axes[3].set_xlabel("Day since Genesis")

    with plt.rc_context({"text.usetex": False}):
        plt.suptitle(
            "Real Ensemble-n: Flow Fit vs Local Linearization",
            color="#CCCCCC",
            fontsize=13,
            y=0.985,
            fontname="Comfortaa",
            fontweight="bold",
        )
    plt.subplots_adjust(top=0.93, hspace=0.25, wspace=0.16)
    return fig
