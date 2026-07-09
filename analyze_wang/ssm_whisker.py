"""Order-2 autonomous SSM computation (cohomological equation)."""

import numpy as np

from .constants import IMAG_TOL


MULTI_IDX_ORDER2 = [(2, 0), (1, 1), (0, 2)]


def _eigenvalue_sum(k, Lambda_E):
    return k[0] * Lambda_E[0] + k[1] * Lambda_E[1]


def _f2_bilinear(H, a, b):
    """Evaluate quadratic vector field bilinear form: component i = a^T H_i b."""
    return np.array([complex(a.conj() @ Hi @ b) for Hi in H])


def _check_resonance(Lk, Lambda_E, reltol=0.05):
    ref = min(abs(ev) for ev in Lambda_E)
    if ref < 1e-10:
        ref = max(abs(ev) for ev in Lambda_E)
    abstol = reltol * ref
    for j, lam_j in enumerate(Lambda_E):
        if abs(Lk - lam_j) < abstol:
            return True, j
    return False, None


def compute_whisker_order2(sys_data, E):
    """Solve the cohomological equation at order 2.

    Returns dict with:
        W2         : dict {(k1,k2): (3,) complex} SSM geometry coefficients
        R2         : dict {(k1,k2): (2,) complex} reduced dynamics coefficients
        Lambda_k   : dict {(k1,k2): complex} eigenvalue sums per multi-index
        L2         : dict {(k1,k2): (3,) complex} RHS vectors
        resonant   : dict {(k1,k2): bool}
        cond       : dict {(k1,k2): float} condition number of cohomological operator
    """
    A = sys_data['A'].astype(complex)
    H = [Hi.astype(complex) for Hi in sys_data['H']]
    V_E = E['V_E']
    Lambda_E = E['Lambda_E']
    n = A.shape[0]

    v1 = V_E[:, 0]
    v2 = V_E[:, 1]

    Lambda_k = {k: _eigenvalue_sum(k, Lambda_E) for k in MULTI_IDX_ORDER2}

    L2 = {
        (2, 0): _f2_bilinear(H, v1, v1),
        (1, 1): _f2_bilinear(H, v1, v2) + _f2_bilinear(H, v2, v1),
        (0, 2): _f2_bilinear(H, v2, v2),
    }

    W2 = {}
    R2 = {}
    resonant_map = {}
    cond_map = {}

    for k in MULTI_IDX_ORDER2:
        Lk = Lambda_k[k]
        Ck = Lk * np.eye(n, dtype=complex) - A

        is_res, res_idx = _check_resonance(Lk, Lambda_E)
        resonant_map[k] = is_res

        if is_res:
            W_E_adj = E['W_E']
            r2_k = np.zeros(2, dtype=complex)
            r2_k[res_idx] = W_E_adj[:, res_idx].conj() @ L2[k]
            R2[k] = r2_k
            rhs = L2[k] - V_E @ r2_k
            W2[k] = np.linalg.lstsq(Ck, rhs, rcond=None)[0]
        else:
            R2[k] = np.zeros(2, dtype=complex)
            W2[k] = np.linalg.solve(Ck, L2[k])

        cond_map[k] = float(np.linalg.cond(Ck))

    return {
        'W2': W2,
        'R2': R2,
        'Lambda_k': Lambda_k,
        'L2': L2,
        'resonant': resonant_map,
        'cond': cond_map,
    }


def format_whisker(result, Lambda_E):
    """Return list of print lines for the order-2 whisker computation."""
    lines = []
    lines.append("ORDER-2 COHOMOLOGICAL EQUATION (AUTONOMOUS SSM)")
    lines.append("")
    lines.append("  invariance equation: (Lambda_k I - A) W2_k = L2_k  (non-resonant)")
    lines.append("                       R2_k via adjoint projection    (resonant)")
    lines.append("")

    freq_labels = {
        (2, 0): '2w  (second harmonic)',
        (1, 1): 'DC  (mean shift)',
        (0, 2): '-2w (conjugate of (2,0))',
    }

    for k in MULTI_IDX_ORDER2:
        Lk = result['Lambda_k'][k]
        w2 = result['W2'][k]
        r2 = result['R2'][k]
        is_res = result['resonant'][k]
        cond = result['cond'][k]

        lines.append(f"  MULTI-INDEX {k}  ->  {freq_labels[k]}")
        lines.append(f"    Lambda_k       = {Lk.real:+.6f}{Lk.imag:+.6f}i")
        lines.append(f"    resonant       : {'YES' if is_res else 'no'}")
        lines.append(f"    cond(C_k)      : {cond:.2e}")
        lines.append(f"    R2_k           = [{', '.join(f'{v.real:+.4e}{v.imag:+.4e}i' for v in r2)}]")

        coord_names = ['u', 'v', 'w']
        lines.append(f"    W2_k (SSM geometry at this frequency):")
        for i in range(len(w2)):
            amp = abs(w2[i])
            phase_deg = np.degrees(np.angle(w2[i])) if amp > 1e-12 else 0.0
            lines.append(f"      {coord_names[i]}: {w2[i].real:+.6e}{w2[i].imag:+.6e}i"
                         f"  |{amp:.4e}|  phase={phase_deg:+.1f} deg")
        lines.append("")

    lines.append("  INTERPRETATION")

    w2_20 = result['W2'][(2, 0)]
    w2_11 = result['W2'][(1, 1)]
    amps_2w = np.abs(w2_20)
    amps_dc = np.abs(w2_11)
    coord_names = ['u (->x)', 'v (->y)', 'w (->z)']

    lines.append("    2w content per observable (from W2[(2,0)]):")
    for i in range(3):
        bar = '#' * int(min(40, amps_2w[i] / (max(amps_2w) + 1e-30) * 40))
        lines.append(f"      {coord_names[i]:10s}: |W2|={amps_2w[i]:.4e}  {bar}")

    lines.append("    DC content per observable (from W2[(1,1)]):")
    for i in range(3):
        bar = '#' * int(min(40, amps_dc[i] / (max(amps_dc) + 1e-30) * 40))
        lines.append(f"      {coord_names[i]:10s}: |W2|={amps_dc[i]:.4e}  {bar}")

    any_2w = any(amps_2w[i] > 1e-10 for i in range(3))
    if any_2w:
        dominant_2w = coord_names[np.argmax(amps_2w)]
        lines.append("")
        lines.append(f"  RESULT: 2w IS generated on the 2D SSM via quadratic NL")
        lines.append(f"  strongest 2w channel: {dominant_2w}")
        lines.append(f"  R2 = 0 at order 2 -> 2w is purely geometric (SSM shape), not in reduced dynamics")
    else:
        lines.append("")
        lines.append(f"  RESULT: no 2w content found at order 2")

    return lines
