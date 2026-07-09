"""SSM/res/local_pca.py — Lokale PCA an jedem Embedding-Punkt.

Idee (modellfrei):
    Statt eine globale Polynom-Form W(PC1, PC2) zu erzwingen, schätze die
    Mannigfaltigkeit LOKAL aus den k nächsten Nachbarn jedes Punkts.

Pro Punkt x_i ∈ ℝ^M:
    1) k-NN  →  N_k(x_i)  ⊂ ℝ^M
    2) Y_i = N_k(x_i) - mean(N_k(x_i))     (zentrieren)
    3) SVD: Y_i = U_i Σ_i V_i^T
    4) lokale Eigenwerte λ_{i,j} = σ_{i,j}^2
    5) lokale '2D-ness' = (λ_1 + λ_2) / Σ λ_j
       → 1.0 wenn Mannigfaltigkeit lokal exakt 2D
       → kleiner wenn höherdimensional (oder gekrümmt → off-manifold-Streuung)

Keine globale Funktionsform, keine Tangentialitätsbedingung, keine
vorgegebene Polynom-Ordnung. Nur Distanzen und lokale Hauptachsen.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class LocalPCAResult:
    k:           int               # Anzahl Nachbarn pro Punkt
    eigenvalues: np.ndarray        # (N, p) lokale σ² (sortiert absteigend)
    two_d_ness:  np.ndarray        # (N,)  (λ1 + λ2) / Σλ  ∈ [0, 1]
    intrinsic_local_dim: np.ndarray  # (N,) Schätzung lokal aus Eigenwert-Knick


def local_pca(X, k=50):
    """Lokale PCA an jedem Punkt mit k nächsten Nachbarn.

    Args:
        X:  (N, M) Embedding-Vektoren
        k:  Anzahl Nachbarn (Default 50, heuristisch — Codex bestätigt vernünftig)

    Returns:
        LocalPCAResult dataclass.
    """
    X = np.asarray(X, dtype=float)
    N, M = X.shape
    if k >= N:
        raise ValueError(f"k={k} muss kleiner als N={N} sein")

    tree = cKDTree(X)
    # Index 0 = sich selbst, also k+1 abfragen
    _, idx_neighbors = tree.query(X, k=k + 1)

    p = min(k, M)        # max. Anzahl Eigenwerte
    eigenvalues = np.zeros((N, p))
    two_d_ness  = np.zeros(N)
    intrinsic_local_dim = np.zeros(N)

    for i in range(N):
        nbrs = X[idx_neighbors[i, 1:]]   # k Nachbarn ohne sich selbst
        Y = nbrs - nbrs.mean(axis=0)
        # SVD ist robust und gibt sortierte Singularwerte
        s = np.linalg.svd(Y, compute_uv=False)
        sig2 = s ** 2
        # auf p Komponenten begrenzen + auffüllen
        m_eff = min(len(sig2), p)
        eigenvalues[i, :m_eff] = sig2[:m_eff]

        total = sig2.sum()
        if total > 1e-30 and len(sig2) >= 2:
            two_d_ness[i] = (sig2[0] + sig2[1]) / total
        else:
            two_d_ness[i] = np.nan

        # Lokale intrinsische Dim aus Eigenwert-Verteilung:
        # Anzahl signifikanter Komponenten via partizipativer Ratio
        if total > 1e-30:
            p_norm = sig2 / total
            intrinsic_local_dim[i] = float(np.exp(
                -(p_norm * np.log(p_norm + 1e-30)).sum()))
        else:
            intrinsic_local_dim[i] = np.nan

    return LocalPCAResult(
        k=int(k),
        eigenvalues=eigenvalues,
        two_d_ness=two_d_ness,
        intrinsic_local_dim=intrinsic_local_dim,
    )
