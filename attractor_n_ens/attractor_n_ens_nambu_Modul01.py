"""Wang/Nambu-Zerlegung des rekonstruierten Ensemble-n-Makroflusses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from attractor_n_ens.attractor_n_ens_flow_Modul01 import FlowModel, vector_field


@dataclass
class ScalarFieldModel:
    labels: tuple[str, ...]
    exponents: tuple[tuple[int, int, int], ...]
    coef: np.ndarray
    values: np.ndarray
    gradients: np.ndarray


@dataclass
class NambuModel:
    flow: FlowModel
    center: np.ndarray
    points_centered: np.ndarray
    divergence_coef: np.ndarray
    potential_values: np.ndarray
    dissipative_field: np.ndarray
    nondissipative_field: np.ndarray
    h1: ScalarFieldModel
    h2: ScalarFieldModel
    singular_values: np.ndarray
    nambu_cross: np.ndarray
    nambu_scale: np.ndarray
    nambu_field: np.ndarray
    align_cos: np.ndarray
    rel_residual: np.ndarray
    drift_h1_full: np.ndarray
    drift_h2_full: np.ndarray
    drift_h1_nd: np.ndarray
    drift_h2_nd: np.ndarray
    drift_h1_diss: np.ndarray
    drift_h2_diss: np.ndarray
    drift_score_h1_full: float
    drift_score_h2_full: float
    drift_score_h1_nd: float
    drift_score_h2_nd: float
    drift_score_h1_diss: float
    drift_score_h2_diss: float


def divergence_coefficients(flow: FlowModel) -> np.ndarray:
    coef = flow.coef
    return np.array(
        [
            coef[1, 0] + coef[2, 1] + coef[3, 2],
            2.0 * coef[4, 0] + coef[5, 1] + coef[6, 2],
            coef[5, 0] + 2.0 * coef[7, 1] + coef[8, 2],
            coef[6, 0] + coef[8, 1] + 2.0 * coef[9, 2],
        ],
        dtype=float,
    )


def evaluate_dissipative_potential(points_centered: np.ndarray, div_coef: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = points_centered[:, 0]
    y = points_centered[:, 1]
    z = points_centered[:, 2]
    d0, d1, d2, d3 = div_coef

    potential = (
        (d0 / 6.0) * (x * x + y * y + z * z)
        + (d1 / 6.0) * x * x * x
        + (d2 / 6.0) * y * y * y
        + (d3 / 6.0) * z * z * z
    )
    gradient = np.column_stack(
        [
            (d0 / 3.0) * x + 0.5 * d1 * x * x,
            (d0 / 3.0) * y + 0.5 * d2 * y * y,
            (d0 / 3.0) * z + 0.5 * d3 * z * z,
        ]
    )
    return potential, gradient


def _monomial_exponents(max_degree: int) -> list[tuple[int, int, int]]:
    exponents: list[tuple[int, int, int]] = []
    for total in range(1, max_degree + 1):
        for px in range(total, -1, -1):
            for py in range(total - px, -1, -1):
                pz = total - px - py
                exponents.append((px, py, pz))
    return exponents


def _label_for_exponent(exp: tuple[int, int, int], coord_labels: tuple[str, str, str] = ("u", "v", "w")) -> str:
    parts: list[str] = []
    for power, label in zip(exp, coord_labels):
        if power <= 0:
            continue
        if power == 1:
            parts.append(label)
        else:
            parts.append(f"{label}^{power}")
    return "".join(parts) if parts else "1"


def build_scalar_basis(points_centered: np.ndarray, degree: int = 3) -> tuple[tuple[str, ...], tuple[tuple[int, int, int], ...], np.ndarray, np.ndarray]:
    x = points_centered[:, 0]
    y = points_centered[:, 1]
    z = points_centered[:, 2]
    exponents = _monomial_exponents(degree)

    values = np.empty((len(points_centered), len(exponents)), dtype=float)
    gradients = np.zeros((len(points_centered), len(exponents), 3), dtype=float)
    labels: list[str] = []

    for idx, (px, py, pz) in enumerate(exponents):
        labels.append(_label_for_exponent((px, py, pz)))

        x_px = x**px
        y_py = y**py
        z_pz = z**pz
        values[:, idx] = x_px * y_py * z_pz

        if px > 0:
            gradients[:, idx, 0] = px * (x ** (px - 1)) * y_py * z_pz
        if py > 0:
            gradients[:, idx, 1] = py * x_px * (y ** (py - 1)) * z_pz
        if pz > 0:
            gradients[:, idx, 2] = pz * x_px * y_py * (z ** (pz - 1))

    return tuple(labels), tuple(exponents), values, gradients


def build_scalar_field(
    labels: tuple[str, ...],
    exponents: tuple[tuple[int, int, int], ...],
    values_basis: np.ndarray,
    gradients_basis: np.ndarray,
    coef: np.ndarray,
) -> ScalarFieldModel:
    values = values_basis @ coef
    gradients = np.tensordot(gradients_basis, coef, axes=([1], [0]))
    return ScalarFieldModel(
        labels=labels,
        exponents=exponents,
        coef=np.asarray(coef, dtype=float),
        values=values,
        gradients=gradients,
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std < 1e-12 or b_std < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _flip_field(field: ScalarFieldModel, sign: float) -> ScalarFieldModel:
    return ScalarFieldModel(
        labels=field.labels,
        exponents=field.exponents,
        coef=sign * field.coef,
        values=sign * field.values,
        gradients=sign * field.gradients,
    )


def orient_invariant_pair(candidates: list[ScalarFieldModel], points_centered: np.ndarray) -> tuple[ScalarFieldModel, ScalarFieldModel]:
    z_scores = [abs(_safe_corr(field.values, points_centered[:, 2])) for field in candidates]
    h1_idx = int(np.argmax(z_scores))
    h2_idx = 1 - h1_idx

    h1 = candidates[h1_idx]
    h2 = candidates[h2_idx]

    if _safe_corr(h1.values, points_centered[:, 2]) < 0.0:
        h1 = _flip_field(h1, -1.0)

    h2_ref = points_centered[:, 0] - 0.5 * points_centered[:, 1]
    if _safe_corr(h2.values, h2_ref) < 0.0:
        h2 = _flip_field(h2, -1.0)

    return h1, h2


def fit_invariant_pair(points_centered: np.ndarray, field: np.ndarray, degree: int = 3) -> tuple[ScalarFieldModel, ScalarFieldModel, np.ndarray]:
    labels, exponents, values_basis, gradients_basis = build_scalar_basis(points_centered, degree=degree)
    matrix = np.einsum("npk,nk->np", gradients_basis, field)
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)

    candidates = [
        build_scalar_field(labels, exponents, values_basis, gradients_basis, vt[-1]),
        build_scalar_field(labels, exponents, values_basis, gradients_basis, vt[-2]),
    ]
    h1, h2 = orient_invariant_pair(candidates, points_centered)
    return h1, h2, singular_values


def _drift_score(gradients: np.ndarray, field: np.ndarray, drift: np.ndarray) -> float:
    scale = np.linalg.norm(gradients, axis=1) * np.linalg.norm(field, axis=1)
    rms_drift = float(np.sqrt(np.mean(drift * drift)))
    rms_scale = float(np.sqrt(np.mean(scale * scale)))
    return rms_drift / max(rms_scale, 1e-12)


def build_nambu_model(flow: FlowModel, points: np.ndarray, degree: int = 3) -> NambuModel:
    points_centered = points - flow.center
    full_field = vector_field(flow, points)

    div_coef = divergence_coefficients(flow)
    potential_values, dissipative_field = evaluate_dissipative_potential(points_centered, div_coef)
    nondissipative_field = full_field - dissipative_field

    h1, h2, singular_values = fit_invariant_pair(points_centered, nondissipative_field, degree=degree)

    nambu_cross = np.cross(h1.gradients, h2.gradients)
    cross_norm_sq = np.sum(nambu_cross * nambu_cross, axis=1)
    nambu_scale = np.sum(nambu_cross * nondissipative_field, axis=1) / np.maximum(cross_norm_sq, 1e-12)
    nambu_field = nambu_scale[:, None] * nambu_cross

    nd_norm = np.linalg.norm(nondissipative_field, axis=1)
    nambu_norm = np.linalg.norm(nambu_cross, axis=1)
    valid = (nd_norm > 1e-12) & (nambu_norm > 1e-12)

    align_cos = np.full(len(points), np.nan, dtype=float)
    rel_residual = np.full(len(points), np.nan, dtype=float)

    align_cos[valid] = np.sum(nambu_cross[valid] * nondissipative_field[valid], axis=1) / (nambu_norm[valid] * nd_norm[valid])
    residual = nondissipative_field[valid] - nambu_field[valid]
    rel_residual[valid] = np.linalg.norm(residual, axis=1) / np.maximum(nd_norm[valid], 1e-12)

    drift_h1_full = np.einsum("ij,ij->i", h1.gradients, full_field)
    drift_h2_full = np.einsum("ij,ij->i", h2.gradients, full_field)
    drift_h1_nd = np.einsum("ij,ij->i", h1.gradients, nondissipative_field)
    drift_h2_nd = np.einsum("ij,ij->i", h2.gradients, nondissipative_field)
    drift_h1_diss = drift_h1_full - drift_h1_nd
    drift_h2_diss = drift_h2_full - drift_h2_nd

    return NambuModel(
        flow=flow,
        center=flow.center.copy(),
        points_centered=points_centered,
        divergence_coef=div_coef,
        potential_values=potential_values,
        dissipative_field=dissipative_field,
        nondissipative_field=nondissipative_field,
        h1=h1,
        h2=h2,
        singular_values=singular_values,
        nambu_cross=nambu_cross,
        nambu_scale=nambu_scale,
        nambu_field=nambu_field,
        align_cos=align_cos,
        rel_residual=rel_residual,
        drift_h1_full=drift_h1_full,
        drift_h2_full=drift_h2_full,
        drift_h1_nd=drift_h1_nd,
        drift_h2_nd=drift_h2_nd,
        drift_h1_diss=drift_h1_diss,
        drift_h2_diss=drift_h2_diss,
        drift_score_h1_full=_drift_score(h1.gradients, full_field, drift_h1_full),
        drift_score_h2_full=_drift_score(h2.gradients, full_field, drift_h2_full),
        drift_score_h1_nd=_drift_score(h1.gradients, nondissipative_field, drift_h1_nd),
        drift_score_h2_nd=_drift_score(h2.gradients, nondissipative_field, drift_h2_nd),
        drift_score_h1_diss=_drift_score(h1.gradients, dissipative_field, drift_h1_diss),
        drift_score_h2_diss=_drift_score(h2.gradients, dissipative_field, drift_h2_diss),
    )


def dominant_terms(field: ScalarFieldModel, top_n: int = 6) -> list[tuple[str, float]]:
    order = np.argsort(np.abs(field.coef))[::-1]
    terms: list[tuple[str, float]] = []
    for idx in order:
        coef = float(field.coef[idx])
        if abs(coef) < 1e-5:
            continue
        terms.append((field.labels[idx], coef))
        if len(terms) >= top_n:
            break
    return terms


def format_scalar_expression(
    field: ScalarFieldModel,
    coord_labels: tuple[str, str, str] = ("u", "v", "w"),
    top_n: int = 6,
    digits: int = 4,
) -> str:
    replacements = {_label_for_exponent(exp): _label_for_exponent(exp, coord_labels) for exp in field.exponents}
    chunks: list[str] = []
    for label, coef in dominant_terms(field, top_n=top_n):
        chunks.append(f"{coef:+.{digits}f}{replacements.get(label, label)}")
    return " ".join(chunks) if chunks else "0"


def format_potential_expression(div_coef: np.ndarray, digits: int = 4) -> str:
    d0, d1, d2, d3 = div_coef
    return (
        f"{d0/6.0:+.{digits}f}(u^2+v^2+w^2) "
        f"{d1/6.0:+.{digits}f}u^3 "
        f"{d2/6.0:+.{digits}f}v^3 "
        f"{d3/6.0:+.{digits}f}w^3"
    )
