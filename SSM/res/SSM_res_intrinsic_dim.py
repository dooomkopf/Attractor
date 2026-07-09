"""SSM/res/intrinsic_dim.py — TWO-NN intrinsische Dimensionsschätzung.

Methode: Facco et al. 2017
    "Estimating the intrinsic dimension of datasets by a minimal
    neighborhood information", Sci. Rep. 7:12140.

Idee:
    Für jeden Datenpunkt x_i die Distanzen zu den zwei nächsten Nachbarn
    r1 < r2 berechnen. Definiere
        μ_i = r2 / r1   ∈ [1, ∞)
    Unter der Annahme lokal uniformer Dichte hat μ_i die theoretische
    Verteilung
        P(μ) = d · μ^{-(d+1)}     für μ ≥ 1
    mit d = intrinsische Dimension.
    Die kumulative Verteilung ist
        F(μ) = 1 - μ^{-d}
    →  log(1 - F(μ)) = -d · log(μ)
    Lineare Regression von log(1 - F̂(μ_i)) gegen log(μ_i) ergibt die
    Steigung -d, also d = -slope.

Wichtige Eigenschaften:
    - keine Funktionsform der Mannigfaltigkeit angenommen
    - keine Glattheit angenommen
    - keine Dimension vorgegeben
    - nur Distanzen zwischen Punkten benötigt
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class TwoNNResult:
    d_intr:        float          # geschätzte intrinsische Dimension
    d_intr_std:    float          # Bootstrap-Std (über sub-samples)
    mu:            np.ndarray     # (N',) μ_i = r2/r1 nach Filterung
    cdf_x:         np.ndarray     # log(μ) für die Plot-Anpassung
    cdf_y:         np.ndarray     # -log(1 - F(μ))
    n_used:        int            # Anzahl gültiger Punkte nach Filter
    n_total:       int            # Anzahl Eingangsdatenpunkte
    bootstrap_d:   np.ndarray     # (n_boot,) Bootstrap-Schätzungen


def two_nn(X, n_boot=20, frac_use=0.9, eps=1e-12, rng_seed=42):
    """TWO-NN intrinsische Dimensionsschätzung.

    Args:
        X:        (N, D) Eingangsdaten (z.B. TDE-Vektoren)
        n_boot:   Anzahl Bootstrap-Wiederholungen für Std-Schätzung
        frac_use: Anteil der μ-Verteilung der für den Linear-Fit verwendet wird
                  (Facco empfiehlt 0.9 — die obersten 10% sind durch Endeffekte
                  verzerrt und werden weggeschnitten)
        eps:      Schutz gegen Duplikate (μ_i muss > 1+eps sein)
        rng_seed: Bootstrap-Seed

    Returns:
        TwoNNResult dataclass.
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    if N < 5:
        raise ValueError(f"TWO-NN braucht mindestens 5 Punkte, bekam {N}")

    # k-NN mit k=2 Nachbarn (ohne Selbst)
    tree = cKDTree(X)
    dists, _ = tree.query(X, k=3)   # Index 0 = sich selbst (dist 0)
    r1 = dists[:, 1]
    r2 = dists[:, 2]

    # μ_i = r2/r1, Schutz gegen r1=0 (Duplikate)
    valid = r1 > eps
    mu = np.full(N, np.nan)
    mu[valid] = r2[valid] / r1[valid]
    mu_clean = mu[np.isfinite(mu) & (mu > 1.0 + eps)]

    if len(mu_clean) < 10:
        raise ValueError(f"Zu wenige gültige μ-Werte: {len(mu_clean)}")

    # Bootstrap für Std-Schätzung
    rng = np.random.default_rng(rng_seed)
    bootstrap_d = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(mu_clean, size=len(mu_clean), replace=True)
        bootstrap_d[i] = _two_nn_fit(sample, frac_use)

    # Hauptschätzung auf Original-Daten
    d_intr = _two_nn_fit(mu_clean, frac_use)

    # Plot-Daten: log(μ) vs -log(1-F)
    mu_sorted = np.sort(mu_clean)
    F = np.arange(1, len(mu_sorted) + 1) / (len(mu_sorted) + 1)
    cdf_x = np.log(mu_sorted)
    cdf_y = -np.log(1.0 - F)

    return TwoNNResult(
        d_intr=float(d_intr),
        d_intr_std=float(np.std(bootstrap_d, ddof=1)),
        mu=mu_clean,
        cdf_x=cdf_x,
        cdf_y=cdf_y,
        n_used=int(len(mu_clean)),
        n_total=int(N),
        bootstrap_d=bootstrap_d,
    )


def _two_nn_fit(mu_clean, frac_use):
    """Linear-Fit log(1-F̂(μ)) ≈ -d · log(μ) auf den unteren frac_use Anteil."""
    mu_sorted = np.sort(mu_clean)
    n = len(mu_sorted)
    F = np.arange(1, n + 1) / (n + 1)
    n_use = int(frac_use * n)
    if n_use < 5:
        return float('nan')
    log_mu = np.log(mu_sorted[:n_use])
    log_oneminusF = np.log(1.0 - F[:n_use])
    # slope = -d  →  d = -slope
    A = np.column_stack([log_mu, np.ones_like(log_mu)])
    sol, *_ = np.linalg.lstsq(A, log_oneminusF, rcond=None)
    slope = sol[0]
    return -slope
