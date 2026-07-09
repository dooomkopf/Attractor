"""SSM/res/geometry.py — SSM-Geometrie-Fit (Phase 1 Schritt 3).

Mathematik (mit Codex abgestimmt, Iteration 1):
    SSM-Annahme: Die Mannigfaltigkeit ist tangential zur (PC1, PC2)-Ebene
                 und nichtlinear (Polynom Grad 2..N) gekrümmt.
    Tangentialitätsbedingung: KEIN konstanter und KEIN linearer Term
                 → Polynom-Basis startet bei Grad 2.
    Skalierung:  u = PC1/std(PC1), v = PC2/std(PC2)  (ddof=0)
                 → Monome auf vergleichbarer Größenordnung.
    Solver:      np.linalg.lstsq(X, Y, rcond=None) — SVD-basiert, multi-output.
    Fit-Quality: Frobenius-R² = 1 - ||Y - Y_hat||²_F / ||Y||²_F
                 Kein Re-Centering von Y (PC-Spalten sind bereits zentriert).
    Diagnostik:  Konditionszahl von X. Warnung > 1e8, fragil > 1e10.
    Order-Scan:  Jeder Order separat fitten (NICHT als Truncation des
                 Maximalfits, die Polynom-Basis ist nicht orthogonal).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GeometryFit:
    order:         int
    master_idx:    tuple                 # (i, j) — z.B. (0, 1) für (PC1, PC2)
    slaved_idx:    np.ndarray            # Array der slaved-PC-Indizes
    u_std:         float                 # Skalierung PC_master[0]
    v_std:         float                 # Skalierung PC_master[1]
    feature_names: list                  # ['u^2', 'uv', 'v^2', 'u^3', ...]
    B:             np.ndarray            # (n_features, K) Polynom-Koeffizienten
    Y_hat:         np.ndarray            # (N, K) Rekonstruktion der slaved PCs
    R2_total:      float                 # Frobenius-R² über alle slaved zusammen
    R2_per_pc:     np.ndarray            # (K,) R² pro slaved PC
    rank:          int                   # Rang von X
    cond:          float                 # Konditionszahl von X
    sing_min:      float                 # min Singularwert von X
    sing_max:      float                 # max Singularwert von X


# ── Polynom-Basis ────────────────────────────────────────────────────────────
def poly_features_2d(u, v, max_order, start_deg=2):
    """Polynom-Features in (u, v) vom Grad start_deg..max_order.

    Tangentialitätsbedingung: start_deg=2 → kein Konstant- und kein Linearterm.

    Anzahl Features: sum_{d=start_deg}^{max_order} (d+1).
    Z.B. start_deg=2, max_order=3: (2+1) + (3+1) = 7 Features.
         start_deg=2, max_order=5: 3+4+5+6 = 18 Features.

    Returns:
        X:             (N, n_features) Design-Matrix
        feature_names: list der Feature-Bezeichnungen
    """
    cols = []
    names = []
    for deg in range(start_deg, max_order + 1):
        for i in range(deg, -1, -1):
            j = deg - i
            cols.append((u ** i) * (v ** j))
            names.append(f'u^{i}v^{j}' if (i > 0 and j > 0)
                         else (f'u^{i}' if j == 0 else f'v^{j}'))
    if not cols:
        return np.empty((len(u), 0)), []
    return np.column_stack(cols), names


# ── Geometrie-Fit ────────────────────────────────────────────────────────────
def fit_geometry(pc, master_idx=(0, 1), slaved_idx=None, K=5, order=3,
                 start_deg=2, eps=1e-12) -> GeometryFit:
    """Polynom-Fit der slaved PCs als Funktion der master PCs.

    Args:
        pc:         (N, M) PCA-Scores (zentriert), aus PCAResult.pc
        master_idx: 2-Tupel der master-PC-Spalten, default (0, 1)
        slaved_idx: Array der slaved-PC-Spalten. Wenn None: die K Spalten
                    direkt nach den master-PCs (häufiger Default).
        K:          Anzahl slaved PCs (nur relevant wenn slaved_idx None ist)
        order:      Maximaler Polynom-Grad
        start_deg:  Tangentialitätsbedingung — Default 2
        eps:        Schutz gegen Division durch 0

    Returns:
        GeometryFit dataclass.
    """
    pc = np.asarray(pc, dtype=float)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"pc must have shape (N, M>=3), got {pc.shape}")

    if slaved_idx is None:
        i_master_max = max(master_idx)
        slaved_idx = np.arange(i_master_max + 1, i_master_max + 1 + K)
    slaved_idx = np.asarray(slaved_idx, dtype=int)
    if slaved_idx.max() >= pc.shape[1]:
        raise ValueError(f"slaved_idx {slaved_idx} exceeds pc.shape[1]={pc.shape[1]}")

    # Master-Koordinaten extrahieren und skalieren
    U = pc[:, master_idx[0]]
    V = pc[:, master_idx[1]]
    u_std = float(U.std(ddof=0))
    v_std = float(V.std(ddof=0))
    if u_std < eps:
        u_std = 1.0
    if v_std < eps:
        v_std = 1.0
    u = U / u_std
    v = V / v_std

    # Slaved Koordinaten
    Y = pc[:, slaved_idx].copy()
    K_eff = Y.shape[1]

    # NaN-Maskierung
    finite = np.isfinite(u) & np.isfinite(v) & np.all(np.isfinite(Y), axis=1)
    if (~finite).any():
        u = u[finite]; v = v[finite]
        Y = Y[finite, :]

    # Polynom-Features (Tangentialität: start_deg=2 default)
    X, feature_names = poly_features_2d(u, v, order, start_deg=start_deg)
    n_feat = X.shape[1]

    if n_feat == 0 or Y.shape[0] <= n_feat:
        return GeometryFit(
            order=order, master_idx=master_idx, slaved_idx=slaved_idx,
            u_std=u_std, v_std=v_std, feature_names=feature_names,
            B=np.zeros((n_feat, K_eff)), Y_hat=np.zeros_like(Y),
            R2_total=float('nan'),
            R2_per_pc=np.full(K_eff, np.nan),
            rank=0, cond=float('inf'), sing_min=0.0, sing_max=0.0,
        )

    # Multi-Output Lstsq (SVD-basiert)
    B, _resid, rank, sv = np.linalg.lstsq(X, Y, rcond=None)
    Y_hat = X @ B

    # Frobenius-R² ohne Re-Centering (Y ist schon zentriert via PCA)
    ss_res = float(np.sum((Y - Y_hat) ** 2))
    ss_tot = float(np.sum(Y ** 2))
    R2_total = 1.0 - ss_res / ss_tot if ss_tot > eps else float('nan')

    # R² pro slaved PC
    R2_per_pc = np.empty(K_eff)
    for i in range(K_eff):
        sr = float(np.sum((Y[:, i] - Y_hat[:, i]) ** 2))
        st = float(np.sum(Y[:, i] ** 2))
        R2_per_pc[i] = 1.0 - sr / st if st > eps else float('nan')

    sing_max = float(sv[0])  if len(sv) > 0 else 0.0
    sing_min = float(sv[-1]) if len(sv) > 0 else 0.0
    cond = sing_max / sing_min if sing_min > eps else float('inf')

    return GeometryFit(
        order=order, master_idx=master_idx, slaved_idx=slaved_idx,
        u_std=u_std, v_std=v_std, feature_names=feature_names,
        B=B, Y_hat=Y_hat,
        R2_total=R2_total, R2_per_pc=R2_per_pc,
        rank=int(rank), cond=cond,
        sing_min=sing_min, sing_max=sing_max,
    )


def scan_orders(pc, orders=(2, 3, 4, 5), master_idx=(0, 1),
                slaved_idx=None, K=5, start_deg=2):
    """Wiederholt fit_geometry für jeden Order separat.

    Codex-Hinweis: niedrigere Orders NICHT durch Truncation aus dem Max-Order-Fit
    ablesen, weil die Polynom-Basis nicht orthogonal ist und sich die
    quadratischen Koeffizienten ändern, sobald höhere Terme mitfitten.

    Returns dict[order] -> GeometryFit.
    """
    out = {}
    for order in orders:
        out[int(order)] = fit_geometry(
            pc, master_idx=master_idx, slaved_idx=slaved_idx,
            K=K, order=int(order), start_deg=start_deg)
    return out


def evaluate_surface(fit: GeometryFit, u_grid, v_grid, slaved_pos=0):
    """Evaluiert eine slaved-PC-Komponente W_k(u, v) auf einem (u, v)-Grid.

    Args:
        fit:        GeometryFit aus fit_geometry/scan_orders
        u_grid:     1D-Array Werte u (bereits skaliert: U/u_std)
        v_grid:     1D-Array Werte v (bereits skaliert: V/v_std)
        slaved_pos: Index der slaved PC innerhalb fit.slaved_idx (default 0 = erste)

    Returns:
        Z: 2D-Array shape (len(v_grid), len(u_grid)), W_k(u, v)
    """
    UU, VV = np.meshgrid(u_grid, v_grid)
    u_flat = UU.ravel()
    v_flat = VV.ravel()
    X_grid, _ = poly_features_2d(u_flat, v_flat, fit.order, start_deg=2)
    z_flat = X_grid @ fit.B[:, slaved_pos]
    return z_flat.reshape(UU.shape)
