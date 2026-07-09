"""LPPL attractor ODE (M=1 polynomial approximation) for SSM analysis.

With M=1 the system is polynomial (degree 3):
    dy1/dt = Z_MIX*y1 + (1-Z_MIX)*y2 - Z_A*y2*z
    dy2/dt = alpha*y2 - gamma*y1^3 + Z_B*y1*z
    dz/dt  = -Z_C*z + Z_D*y1 + Z_E*y1*y2

Parameters from lpplattr02_params.py (LPPL-attractor/).
"""

import numpy as np
from scipy.optimize import fsolve


DEFAULT_PARAMS = {
    'alpha': -0.00074,
    'gamma': 0.003,
    'Z_A': 8e-3,
    'Z_B': 8e-3,
    'Z_C': 0.0039,
    'Z_D': 1e-6,
    'Z_E': 2.0,
    'Z_MIX': 0.0002,
}

WANG_OFF_PARAMS = {
    'Z_A': 0.0,
    'Z_B': 0.0,
    'Z_MIX': 8e-5,
}


def build_params(wang_off=False, **overrides):
    """Build parameter dict. --wang-off sets Z_A=0, Z_B=0, Z_MIX=8e-5 (wins over overrides)."""
    p = dict(DEFAULT_PARAMS)
    p.update(overrides)
    if wang_off:
        p.update(WANG_OFF_PARAMS)
    return p


def lppl_rhs(state, p=None):
    if p is None:
        p = DEFAULT_PARAMS
    y1, y2, z = state
    dy1 = p['Z_MIX'] * y1 + (1.0 - p['Z_MIX']) * y2 - p['Z_A'] * y2 * z
    dy2 = p['alpha'] * y2 - p['gamma'] * y1**3 + p['Z_B'] * y1 * z
    dz = -p['Z_C'] * z + p['Z_D'] * y1 + p['Z_E'] * y1 * y2
    return np.array([dy1, dy2, dz], dtype=float)


def lppl_jacobian(state, p=None):
    if p is None:
        p = DEFAULT_PARAMS
    y1, y2, z = state
    return np.array([
        [p['Z_MIX'],             1.0 - p['Z_MIX'] - p['Z_A'] * z,   -p['Z_A'] * y2],
        [-3.0 * p['gamma'] * y1**2 + p['Z_B'] * z,   p['alpha'],     p['Z_B'] * y1],
        [p['Z_D'] + p['Z_E'] * y2,                    p['Z_E'] * y1,  -p['Z_C']],
    ], dtype=float)


def find_equilibria(p=None, n_trials=200, tol=1e-10):
    if p is None:
        p = DEFAULT_PARAMS
    rng = np.random.default_rng(42)
    found = [np.zeros(3)]

    for _ in range(n_trials):
        guess = rng.uniform(-5, 5, size=3)
        try:
            sol = fsolve(lambda y: lppl_rhs(y, p), guess, full_output=True)
            x, info, ier, _ = sol
            if ier == 1 and np.linalg.norm(lppl_rhs(x, p)) < tol:
                is_new = all(np.linalg.norm(x - f) > 1e-6 for f in found)
                if is_new:
                    found.append(x.copy())
        except Exception:
            continue

    found.sort(key=lambda x: (x[0]**2 + x[1]**2 + x[2]**2))
    return [(f'E{i+1}', pt) for i, pt in enumerate(found)]


def shifted_first_order(eq_point, p=None):
    if p is None:
        p = DEFAULT_PARAMS
    eq = np.asarray(eq_point, dtype=float)
    A = lppl_jacobian(eq, p)

    y1e, y2e, ze = eq
    H0 = np.zeros((3, 3))
    H0[1, 2] = -p['Z_A'] / 2.0
    H0[2, 1] = -p['Z_A'] / 2.0

    H1 = np.zeros((3, 3))
    H1[0, 2] = p['Z_B'] / 2.0
    H1[2, 0] = p['Z_B'] / 2.0

    H2 = np.zeros((3, 3))
    H2[0, 1] = p['Z_E'] / 2.0
    H2[1, 0] = p['Z_E'] / 2.0

    T = np.zeros((3, 3, 3))
    T[1, 0, 0] = -p['gamma'] * 3.0

    return {
        'A': A,
        'B': np.eye(3),
        'H': [H0, H1, H2],
        'T': T,
        'eq': eq,
        'params': p,
        'nl_degree': 3,
    }


def eval_quadratic(H, z):
    z = np.asarray(z, dtype=float)
    return np.array([float(z @ Hi @ z) for Hi in H])


def eval_cubic(T, z):
    z = np.asarray(z, dtype=float)
    result = np.zeros(3)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i] += T[i, j, k] * z[j] * z[k] * z[j]
    return result


def verify_shifted_ode(sys_data, test_perturbation=None):
    eq = sys_data['eq']
    A = sys_data['A']
    H = sys_data['H']
    p = sys_data['params']

    if test_perturbation is None:
        rng = np.random.default_rng(42)
        test_perturbation = rng.standard_normal(3) * 0.01

    dz = np.asarray(test_perturbation, dtype=float)
    full_point = eq + dz

    rhs_original = lppl_rhs(full_point, p)
    rhs_lin = A @ dz
    rhs_quad = eval_quadratic(H, dz)
    y1e = eq[0]
    cubic_contrib = np.array([
        0.0,
        -p['gamma'] * (3.0 * y1e * dz[0]**2 + dz[0]**3),
        0.0,
    ])
    rhs_shifted = rhs_lin + rhs_quad + cubic_contrib
    error = np.linalg.norm(rhs_original - rhs_shifted)
    return {
        'dz': dz,
        'rhs_original': rhs_original,
        'rhs_shifted': rhs_shifted,
        'error': float(error),
    }
