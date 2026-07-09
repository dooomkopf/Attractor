"""Wang 2-scroll equations, Jacobian, and equilibria."""

import numpy as np


def wang_rhs(state, a, b, c, d):
    x, y, z = state
    return np.array([
        a * (x - y) - y * z,
        -b * y + x * z,
        -c * z + d * x + x * y,
    ], dtype=float)


def wang_jacobian(state, a, b, c, d):
    x, y, z = state
    return np.array([
        [a, -a - z, -y],
        [z, -b, x],
        [d + y, x, -c],
    ], dtype=float)


def equilibria(a, b, c, d):
    pts = [('S1', np.zeros(3, dtype=float))]
    disc_z = a * a + 4.0 * a * b
    if disc_z < 0:
        return pts
    for sign_z in (+1.0, -1.0):
        z = (-a + sign_z * np.sqrt(disc_z)) / 2.0
        a2 = z / b
        b2 = d
        c2 = -c * z
        disc_x = b2 * b2 - 4.0 * a2 * c2
        if disc_x < 0:
            continue
        root_disc_x = np.sqrt(disc_x)
        for sign_x in (+1.0, -1.0):
            x = (-b2 + sign_x * root_disc_x) / (2.0 * a2)
            y = x * z / b
            pts.append((f'S{len(pts)+1}', np.array([x, y, z], dtype=float)))
    return pts


def shifted_quadratic_terms():
    return {
        'du': '-v*w',
        'dv': '+u*w',
        'dw': '+u*v',
    }
