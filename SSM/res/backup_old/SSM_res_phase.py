"""SSM/res/phase.py — Intrinsische Phase aus den Master-Koordinaten.

Mathematik:
    θ(t)        = atan2(PC2(t), PC1(t))           ← Phase
    ρ(t)        = sqrt(PC1(t)² + PC2(t)²)         ← Amplitude
    θ_unwrap(t) = unwrap(θ(t))                    ← stetig fortgesetzt
    Cycle-Anfänge = Stellen, an denen θ_unwrap die 2π-Vielfachen kreuzt

Die Cycle-Bestimmung ist damit komplett intrinsisch: sie kommt aus
der Geometrie der Master-Koordinaten und ist unabhängig von externen
Markern wie BTC-Halvings. Halvings werden später nur zur Anker-Setzung
(Phasen-Offset) und als visuelle Marker verwendet — sie definieren
keine Cycle-Grenzen.

Konvention (analog attractor.py:166-172):
    Wenn θ_unwrap fallend statt steigend ist, wird das Vorzeichen geflippt,
    sodass der Limit-Cycle counterclockwise (CCW) durchlaufen wird.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PhaseResult:
    theta:        np.ndarray   # (N,) — atan2(PC2,PC1) in [-π, π] (oder [0, 2π) nach Anchor)
    theta_unwrap: np.ndarray   # (N,) — unwrapped, stetig
    rho:          np.ndarray   # (N,) — Amplitude sqrt(PC1²+PC2²)
    bounds:       np.ndarray   # (n_cycles+1,) — Indizes der Cycle-Anfänge (incl. 0 und N-1)
    direction:    int          # +1 oder -1 (vor Flip)
    anchor_shift: float        # rad — Verschiebung beim Anchoring (0 wenn nicht verankert)


def compute_phase(pc, master_idx=(0, 1)):
    """Phase, Amplitude, Unwrap aus den Master-Koordinaten.

    Args:
        pc:         (N, M) PCA-Scores
        master_idx: 2-Tupel der master-PC-Spalten, default (0, 1)

    Returns:
        theta, theta_unwrap, rho, direction
            direction: +1 wenn θ_unwrap insgesamt steigt, -1 wenn fallend
                       (im 2. Fall werden θ und θ_unwrap geflippt)
    """
    pc = np.asarray(pc, dtype=float)
    u = pc[:, master_idx[0]]
    v = pc[:, master_idx[1]]

    theta    = np.arctan2(v, u)
    rho      = np.sqrt(u * u + v * v)
    theta_uw = np.unwrap(theta)

    if theta_uw[-1] < theta_uw[0]:
        # CW → flip auf CCW (Konvention attractor.py:166-172)
        theta    = -theta
        theta_uw = -theta_uw
        direction = -1
    else:
        direction = +1

    return theta, theta_uw, rho, direction


def anchor_phase_to_halving(theta, theta_uw, days_vecs, halving_days,
                            halving_idx_for_anchor=1, max_dist=200):
    """Verschiebt die Phase, sodass θ = 0 am gewählten Halving liegt.

    Wie attractor.py:179-183. Standard: Halving 2 (Index 1) als Anker —
    funktioniert für BTC-Daten ab ~2012, weil Halving 1 am Anfang ist und
    Halving 2 in der Mitte liegt → robuste Verankerung.

    Args:
        theta:                  (N,) Phase aus compute_phase
        theta_uw:               (N,) unwrapped Phase
        days_vecs:              (N,) Tag-Achse der Embedding-Vektoren
        halving_days:           int-Array der Halving-Tage
        halving_idx_for_anchor: welches Halving als Anker (default 1)
        max_dist:               nur verankern wenn das gewählte Halving
                                tatsächlich nahe an einem Datenpunkt liegt

    Returns:
        theta_a, theta_uw_a, anchor_shift
    """
    if halving_idx_for_anchor >= len(halving_days):
        return theta.copy(), theta_uw.copy(), 0.0
    hday = halving_days[halving_idx_for_anchor]
    hidx = int(np.argmin(np.abs(days_vecs - hday)))
    if abs(days_vecs[hidx] - hday) > max_dist:
        return theta.copy(), theta_uw.copy(), 0.0

    shift = float(theta[hidx])
    theta_a    = (theta - shift) % (2 * np.pi)
    theta_uw_a = theta_uw - shift
    return theta_a, theta_uw_a, shift


def cycle_bounds_from_phase(theta_uw):
    """Cycle-Anfänge als 2π-Crossings von θ_unwrap.

    Returns:
        bounds: int-Array mit den Indizes [0, c1, c2, ..., N-1].
                Jedes Intervall [bounds[k], bounds[k+1]] ist ein vollständiger Cycle
                (außer am Rand: erster und letzter Cycle ggf. unvollständig).
    """
    cross_idx = []
    for k in range(1, 30):
        level = k * 2 * np.pi
        idx = np.where((theta_uw[:-1] < level) & (theta_uw[1:] >= level))[0]
        if len(idx) > 0:
            cross_idx.append(int(idx[0]))
    bounds = [0] + cross_idx + [len(theta_uw) - 1]
    return np.array(bounds, dtype=int)


def compute_phase_full(pc, days_vecs, halving_days,
                       master_idx=(0, 1), anchor_idx=1):
    """Convenience: alle Phasen-Schritte zusammen → PhaseResult.

    1) compute_phase(pc, master_idx)
    2) anchor_phase_to_halving(... , halving_days, anchor_idx)
    3) cycle_bounds_from_phase(theta_uw)
    """
    theta, theta_uw, rho, direction = compute_phase(pc, master_idx=master_idx)
    theta_a, theta_uw_a, shift = anchor_phase_to_halving(
        theta, theta_uw, days_vecs, halving_days,
        halving_idx_for_anchor=anchor_idx)
    bounds = cycle_bounds_from_phase(theta_uw_a)
    return PhaseResult(
        theta=theta_a, theta_unwrap=theta_uw_a, rho=rho,
        bounds=bounds, direction=direction, anchor_shift=shift,
    )
