"""SSM/raw/embedding.py — Time Delay Embedding, PCA, Smoothing.

Wiederverwendbar für SSM/res/ und SSM/n/ ohne Änderungen.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class PCAResult:
    Vt:     np.ndarray   # (M, M)  rechtsseitige Eigenvektoren (Moden)
    pc:     np.ndarray   # (N-W, M) PCA-Scores (zentriert)
    var:    np.ndarray   # (M,) normalisierte Varianzen sigma_k^2/sum
    s:      np.ndarray   # (M,) Singularwerte
    D_mean: np.ndarray   # (M,) Spaltenmittel von D


def build_embedding(signal, M, TAU):
    """Klassisches Time Delay Embedding (Takens), wie attractor.py:146-148.

    Returns D ∈ R^{(N-W) × M}, mit W=(M-1)*TAU.
    Spalte j greift signal[W-j*TAU : N-j*TAU].
    Convention identisch zu attractor.py: Spalte 0 = "jetzt", j↑ = weiter zurück.
    """
    N = len(signal)
    W = (M - 1) * TAU
    if N - W < M:
        raise ValueError(f"Signal zu kurz: N={N}, W={W}, brauche N-W>={M}")
    D = np.empty((N - W, M))
    for j in range(M):
        D[:, j] = signal[W - j * TAU : N - j * TAU]
    return D, W


def pca(D):
    """Centered SVD-PCA wie attractor.py:152-155, mit s zusätzlich.

    Returns:
        PCAResult mit Vt, pc, var, s, D_mean.
    """
    D_mean = D.mean(axis=0)
    D_c    = D - D_mean
    _, s, Vt = np.linalg.svd(D_c, full_matrices=False)
    pc  = D_c @ Vt.T
    var = s**2 / (s**2).sum()
    return PCAResult(Vt=Vt, pc=pc, var=var, s=s, D_mean=D_mean)


def smooth_pcs(pc, sigma):
    """Gaussian-Filter pro PC-Spalte, wie attractor.py:158-160.

    Bei sigma <= 0 wird eine Kopie ohne Filter zurückgegeben.
    """
    if sigma <= 0:
        return pc.copy()
    pc_s = pc.copy()
    for j in range(pc_s.shape[1]):
        pc_s[:, j] = gaussian_filter1d(pc[:, j], sigma=sigma)
    return pc_s
