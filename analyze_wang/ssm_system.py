"""First-order form of the Wang system shifted to an equilibrium for SSM computation."""

import numpy as np

from .system import equilibria, wang_jacobian


def shifted_first_order(eq_point, a, b, c, d):
    """Build A matrix and quadratic Hessians for Wang ODE shifted to eq_point.

    Returns dict with:
        A       : (3,3) Jacobian at equilibrium (linear part of shifted ODE)
        B       : (3,3) identity (first-order form: B dz/dt = A z + F2(z))
        H       : list of three (3,3) symmetric matrices (quadratic Hessians)
                  equation i gets quadratic contribution z^T @ H[i] @ z
        eq      : the equilibrium point (3,)
        params  : dict of a, b, c, d
    """
    eq = np.asarray(eq_point, dtype=float)
    x_eq, y_eq, z_eq = eq

    A = wang_jacobian(eq, a, b, c, d)
    B = np.eye(3)

    H0 = np.zeros((3, 3))
    H0[1, 2] = -0.5
    H0[2, 1] = -0.5
    H1 = np.zeros((3, 3))
    H1[0, 2] = 0.5
    H1[2, 0] = 0.5
    H2 = np.zeros((3, 3))
    H2[0, 1] = 0.5
    H2[1, 0] = 0.5

    return {
        'A': A,
        'B': B,
        'H': [H0, H1, H2],
        'eq': eq,
        'params': {'a': a, 'b': b, 'c': c, 'd': d},
    }


def eval_quadratic(H, z):
    """Evaluate quadratic vector field: F2_i(z) = z^T @ H[i] @ z."""
    z = np.asarray(z, dtype=float)
    return np.array([float(z @ Hi @ z) for Hi in H])


def verify_shifted_ode(sys_data, test_perturbation=None):
    """Verify shifted ODE by comparing A*dz + F2(dz) to original wang_rhs at eq + dz."""
    from .system import wang_rhs

    eq = sys_data['eq']
    A = sys_data['A']
    H = sys_data['H']
    p = sys_data['params']

    if test_perturbation is None:
        rng = np.random.default_rng(42)
        test_perturbation = rng.standard_normal(3) * 0.1

    dz = np.asarray(test_perturbation, dtype=float)
    full_point = eq + dz

    rhs_original = wang_rhs(full_point, p['a'], p['b'], p['c'], p['d'])
    rhs_shifted = A @ dz + eval_quadratic(H, dz)
    error = np.linalg.norm(rhs_original - rhs_shifted)
    return {
        'dz': dz,
        'rhs_original': rhs_original,
        'rhs_shifted': rhs_shifted,
        'error': float(error),
    }


def format_system(sys_data, label=''):
    """Return list of print lines describing the shifted system."""
    eq = sys_data['eq']
    A = sys_data['A']
    H = sys_data['H']
    lines = []
    if label:
        lines.append(f"SHIFTED SYSTEM AT {label}: ({eq[0]:+.6f}, {eq[1]:+.6f}, {eq[2]:+.6f})")
    else:
        lines.append(f"SHIFTED SYSTEM AT ({eq[0]:+.6f}, {eq[1]:+.6f}, {eq[2]:+.6f})")
    lines.append("  form: B dz/dt = A z + F2(z),  B = I")
    lines.append("  coordinates: u = x-x_eq, v = y-y_eq, w = z-z_eq")
    lines.append("")
    lines.append("  A (Jacobian):")
    for i in range(3):
        row_txt = "    [" + "  ".join(f"{A[i,j]:+10.6f}" for j in range(3)) + " ]"
        lines.append(row_txt)
    lines.append("")
    names = ['u', 'v', 'w']
    lines.append("  quadratic terms F2(z) = z^T H_i z:")
    for i, Hi in enumerate(H):
        nz = [(j, k) for j in range(3) for k in range(j, 3) if abs(Hi[j, k]) > 1e-12]
        if not nz:
            lines.append(f"    d{names[i]}: (none)")
        else:
            terms = []
            for j, k in nz:
                coeff = Hi[j, k] + (Hi[k, j] if j != k else 0.0)
                sign = '+' if coeff > 0 else '-'
                if j == k:
                    terms.append(f"{sign}{abs(coeff):.0f}*{names[j]}^2")
                else:
                    terms.append(f"{sign}{abs(coeff):.0f}*{names[j]}*{names[k]}")
            lines.append(f"    d{names[i]}: {' '.join(terms)}")
    return lines
