"""Quadratische Fluss-Rekonstruktion auf dem realen Ensemble-n-Attraktor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root

FEATURE_LABELS = ("1", "x", "y", "z", "x^2", "xy", "xz", "y^2", "yz", "z^2")


@dataclass
class FlowModel:
    center: np.ndarray
    coef: np.ndarray
    velocity: np.ndarray
    velocity_fit: np.ndarray
    r2: np.ndarray
    rmse: np.ndarray
    dir_cos: np.ndarray


def estimate_velocity(points: np.ndarray, times: np.ndarray) -> np.ndarray:
    vel = np.empty_like(points)
    for j in range(points.shape[1]):
        vel[:, j] = np.gradient(points[:, j], times)
    return vel


def build_quadratic_design(points_centered: np.ndarray) -> np.ndarray:
    x = points_centered[:, 0]
    y = points_centered[:, 1]
    z = points_centered[:, 2]
    return np.column_stack(
        [
            np.ones(len(points_centered)),
            x,
            y,
            z,
            x * x,
            x * y,
            x * z,
            y * y,
            y * z,
            z * z,
        ]
    )


def fit_quadratic_flow(points: np.ndarray, times: np.ndarray, ridge: float = 1e-6) -> FlowModel:
    center = np.mean(points, axis=0)
    points_centered = points - center
    velocity = estimate_velocity(points, times)
    design = build_quadratic_design(points_centered)

    reg = np.eye(design.shape[1], dtype=float)
    reg[0, 0] = 1e-12
    coef = np.linalg.solve(design.T @ design + ridge * reg, design.T @ velocity)
    velocity_fit = design @ coef

    resid = velocity - velocity_fit
    ss_res = np.sum(resid * resid, axis=0)
    vel_centered = velocity - np.mean(velocity, axis=0)
    ss_tot = np.sum(vel_centered * vel_centered, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    rmse = np.sqrt(np.mean(resid * resid, axis=0))

    vel_norm = np.linalg.norm(velocity, axis=1)
    fit_norm = np.linalg.norm(velocity_fit, axis=1)
    denom = np.maximum(vel_norm * fit_norm, 1e-12)
    dir_cos = np.sum(velocity * velocity_fit, axis=1) / denom

    return FlowModel(
        center=center,
        coef=coef,
        velocity=velocity,
        velocity_fit=velocity_fit,
        r2=r2,
        rmse=rmse,
        dir_cos=dir_cos,
    )


def vector_field(model: FlowModel, points: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(points).astype(float)
    design = build_quadratic_design(pts - model.center)
    field = design @ model.coef
    return field[0] if np.ndim(points) == 1 else field


def jacobian(model: FlowModel, point: np.ndarray) -> np.ndarray:
    x, y, z = (np.asarray(point, dtype=float) - model.center)
    jac = np.empty((3, 3), dtype=float)
    for k in range(3):
        c = model.coef[:, k]
        jac[k, 0] = c[1] + 2.0 * c[4] * x + c[5] * y + c[6] * z
        jac[k, 1] = c[2] + c[5] * x + 2.0 * c[7] * y + c[8] * z
        jac[k, 2] = c[3] + c[6] * x + c[8] * y + 2.0 * c[9] * z
    return jac


def trajectory_diagnostics(model: FlowModel, points: np.ndarray) -> dict:
    divergence = np.empty(len(points), dtype=float)
    eigvals = np.empty((len(points), 3), dtype=complex)
    for i, point in enumerate(points):
        jac = jacobian(model, point)
        divergence[i] = float(np.trace(jac))
        eigvals[i] = np.linalg.eigvals(jac)

    eig_real = np.sort(np.real(eigvals), axis=1)
    return {
        "divergence": divergence,
        "eigvals": eigvals,
        "eig_real_sorted": eig_real,
        "max_real": eig_real[:, 2],
        "mid_real": eig_real[:, 1],
        "min_real": eig_real[:, 0],
    }


def classify_fixed_point(eigvals: np.ndarray, eps: float = 1e-5) -> str:
    real_parts = np.real(eigvals)
    pos = int(np.sum(real_parts > eps))
    neg = int(np.sum(real_parts < -eps))
    imag = np.max(np.abs(np.imag(eigvals))) > eps

    if pos and neg:
        return "saddle-focus" if imag else "saddle"
    if pos == 0 and neg >= 1:
        return "spiral-sink" if imag else "sink"
    if neg == 0 and pos >= 1:
        return "spiral-source" if imag else "source"
    return "center"


def build_seed_points(points: np.ndarray, days_vecs: np.ndarray, data: dict) -> np.ndarray:
    seeds = [np.mean(points, axis=0), np.zeros(3)]

    for key in ("halving_days", "used_peak_days", "used_bottom_days"):
        for day in data.get(key, []):
            idx = int(np.argmin(np.abs(days_vecs - day)))
            seeds.append(points[idx])

    idx_sparse = np.linspace(0, len(points) - 1, min(20, len(points)), dtype=int)
    seeds.extend(points[idx_sparse])

    stacked = np.vstack(seeds)
    rounded = np.round(stacked, decimals=6)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return stacked[np.sort(unique_idx)]


def find_fixed_points(model: FlowModel, points: np.ndarray, seeds: np.ndarray) -> list[dict]:
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    pad = 0.25 * (bbox_max - bbox_min)
    merge_tol = 0.02 * np.linalg.norm(bbox_max - bbox_min)
    merge_tol = max(merge_tol, 1e-3)

    found: list[dict] = []
    for seed in seeds:
        sol = root(lambda vec: vector_field(model, vec), seed, method="hybr")
        if not sol.success:
            continue
        point = np.asarray(sol.x, dtype=float)
        resid = float(np.linalg.norm(vector_field(model, point)))
        if resid > 1e-6:
            continue
        if np.any(point < (bbox_min - pad)) or np.any(point > (bbox_max + pad)):
            continue

        merged = False
        for item in found:
            if np.linalg.norm(point - item["point"]) <= merge_tol:
                if resid < item["residual"]:
                    item["point"] = point
                    item["residual"] = resid
                merged = True
                break
        if not merged:
            found.append({"point": point, "residual": resid})

    for item in found:
        eigvals = np.linalg.eigvals(jacobian(model, item["point"]))
        item["eigvals"] = eigvals
        item["kind"] = classify_fixed_point(eigvals)
        item["distance_to_cloud"] = float(np.min(np.linalg.norm(points - item["point"], axis=1)))

    found.sort(key=lambda item: (item["distance_to_cloud"], item["residual"]))
    return found
