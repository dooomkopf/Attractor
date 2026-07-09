#!/usr/bin/env python3
"""Schnelltest: LPPL als reines 2D-System (kein z, kein Wang).
dy1 = y2
dy2 = alpha*y2 - gamma*y1^3
Vergleich mit dem 3D-System (mit Wang-Kopplung)."""

import numpy as np
from scipy.optimize import fsolve

ALPHA = -0.00074
GAMMA = 0.003


def rhs_2d(y):
    y1, y2 = y
    return np.array([y2, ALPHA * y2 - GAMMA * y1**3])


def jac_2d(y):
    y1, y2 = y
    return np.array([
        [0.0, 1.0],
        [-3.0 * GAMMA * y1**2, ALPHA],
    ])


def find_eq_2d():
    found = [np.zeros(2)]
    for guess in [np.array([0.5, 0]), np.array([-0.5, 0]), np.array([1, 0]), np.array([-1, 0])]:
        sol = fsolve(rhs_2d, guess, full_output=True)
        x, _, ier, _ = sol
        if ier == 1 and np.linalg.norm(rhs_2d(x)) < 1e-10:
            if all(np.linalg.norm(x - f) > 1e-6 for f in found):
                found.append(x.copy())
    return found


print("=" * 72)
print("2D LPPL (kein z, kein Wang)")
print(f"dy1 = y2")
print(f"dy2 = {ALPHA}*y2 - {GAMMA}*y1^3")
print("=" * 72)

eqs = find_eq_2d()
for i, eq in enumerate(eqs):
    J = jac_2d(eq)
    eigvals = np.linalg.eigvals(J)
    eigvals = sorted(eigvals, key=lambda z: (z.real, abs(z.imag)))
    print(f"\nE{i+1}: ({eq[0]:+.6f}, {eq[1]:+.6f})")
    for ev in eigvals:
        if abs(ev.imag) > 1e-8:
            T = 2 * np.pi / abs(ev.imag)
            print(f"  lambda = {ev.real:+.6e} +/- {abs(ev.imag):.6e}i  T={T:.2f}d")
        else:
            print(f"  lambda = {ev.real:+.6e}  (real)")

print("\n" + "=" * 72)
print("VERGLEICH")
print("=" * 72)
print("2D: Gleichgewichte existieren bei y1=0 (Ursprung)")
print("    Jacobian am Ursprung: [[0, 1], [0, alpha]]")
print(f"    Eigenwerte: 0, {ALPHA}")
print("    -> KEIN oszillatorisches Paar (beide reell)")
print("    -> System ist nicht-hyperbolisch (lambda=0)")
print()
print("Aber: bei y1 != 0 gibt es KEINE weiteren Gleichgewichte")
print("(dy1=0 -> y2=0, dy2=0 -> -gamma*y1^3=0 -> y1=0)")
print("-> einziges Gleichgewicht ist der Ursprung")
print()
print("Die Schwingung im 2D-System kommt NICHT aus einem Fokus,")
print("sondern aus der nichtlinearen Wechselwirkung alpha*y2 vs gamma*y1^3")
print("(Duffing-artig). Es ist ein Grenzzyklus, kein Spiral-Fokus.")
print()
print("3D (mit Wang): Z_A/Z_B-Kopplung erzeugt neue Gleichgewichte E2/E3")
print("die oszillatorische Eigenwertpaare haben -> SSM-Analyse moeglich.")
print("=" * 72)
