"""Spectral analysis and master/slave mode selection (SSMTool_jain style)."""

import numpy as np

from .constants import IMAG_TOL


def spectral_analysis(A):
    """Eigendecomposition of A with SSMTool_jain conventions.

    Returns dict with:
        Lambda : (n,) eigenvalues sorted by descending Re, pos Im first in pairs
        V      : (n,n) right eigenvectors (columns), normalized
        W      : (n,n) left eigenvectors (columns), W^H V = I
    """
    eigvals, V = np.linalg.eig(A)

    idx = _sort_eigenvalues(eigvals)
    eigvals = eigvals[idx]
    V = V[:, idx]

    W = np.linalg.inv(V).conj().T
    mu = np.diag(W.conj().T @ V)
    W = W @ np.diag(1.0 / mu.conj())

    ortho_err = np.linalg.norm(W.conj().T @ V - np.eye(len(eigvals)))

    return {
        'Lambda': eigvals,
        'V': V,
        'W': W,
        'ortho_error': float(ortho_err),
    }


def _sort_eigenvalues(eigvals):
    n = len(eigvals)
    order = sorted(range(n), key=lambda k: (-np.real(eigvals[k]), abs(np.imag(eigvals[k]))))

    result = []
    used = set()
    for k in order:
        if k in used:
            continue
        if abs(np.imag(eigvals[k])) > IMAG_TOL:
            conj_idx = None
            for j in order:
                if j != k and j not in used and abs(eigvals[j] - np.conj(eigvals[k])) < 1e-8:
                    conj_idx = j
                    break
            if conj_idx is not None:
                if np.imag(eigvals[k]) > 0:
                    result.extend([k, conj_idx])
                else:
                    result.extend([conj_idx, k])
                used.update([k, conj_idx])
                continue
        result.append(k)
        used.add(k)
    return result


def choose_E(spec, master_indices=None):
    """Select master spectral subspace E (SSMTool_jain choose_E style).

    If master_indices is None, auto-selects the complex conjugate pair
    with largest Re (slowest decay / fastest growth).

    Returns dict with:
        master_idx   : list of master eigenvalue indices
        slave_idx    : list of slave eigenvalue indices
        Lambda_E     : master eigenvalues
        V_E          : master right eigenvectors
        W_E          : master left eigenvectors (adjoint basis)
        Lambda_S     : slave eigenvalues
        sigma_in     : inner spectral quotient
        sigma_out    : outer spectral quotient
        resonances   : list of near-resonance hits
    """
    eigvals = spec['Lambda']
    V = spec['V']
    W = spec['W']
    n = len(eigvals)

    if master_indices is None:
        pos_imag = [(k, eigvals[k]) for k in range(n) if np.imag(eigvals[k]) > IMAG_TOL]
        if not pos_imag:
            raise RuntimeError("no oscillatory mode found")
        pos_imag.sort(key=lambda item: -np.real(item[1]))
        k_main = pos_imag[0][0]
        k_conj = None
        for j in range(n):
            if j != k_main and abs(eigvals[j] - np.conj(eigvals[k_main])) < 1e-8:
                k_conj = j
                break
        if k_conj is None:
            raise RuntimeError("conjugate partner not found")
        master_indices = sorted([k_main, k_conj])

    slave_indices = [k for k in range(n) if k not in master_indices]
    lam_M = eigvals[master_indices]
    lam_S = eigvals[slave_indices]

    re_M_max = float(np.max(np.real(lam_M)))
    re_S_min = float(np.min(np.real(lam_S))) if len(lam_S) > 0 else None
    lam_all = np.concatenate([lam_M, lam_S])

    sigma_in = None
    sigma_out = None
    if abs(re_M_max) > 1e-12 and re_S_min is not None:
        sigma_in = int(np.fix(np.min(np.real(lam_all)) / re_M_max))
        sigma_out = int(np.fix(re_S_min / re_M_max))

    resonances = _scan_resonances(lam_M, lam_S)

    return {
        'master_idx': list(master_indices),
        'slave_idx': list(slave_indices),
        'Lambda_E': lam_M,
        'V_E': V[:, master_indices],
        'W_E': W[:, master_indices],
        'Lambda_S': lam_S,
        'V_S': V[:, slave_indices],
        'sigma_in': sigma_in,
        'sigma_out': sigma_out,
        'resonances': resonances,
    }


def _scan_resonances(lam_M, lam_S, reltol=0.05, max_order=4):
    if len(lam_M) != 2 or len(lam_S) == 0:
        return []
    ref = min(abs(ev) for ev in lam_M)
    if ref < 1e-10:
        ref = max(abs(ev) for ev in lam_M)
    abstol = reltol * ref
    hits = []
    for order in range(2, max_order + 1):
        for a in range(order + 1):
            b = order - a
            combo = a * lam_M[0] + b * lam_M[1]
            for si, sev in enumerate(lam_S):
                mismatch = abs(combo - sev)
                if mismatch < abstol:
                    hits.append({
                        'combo': (a, b),
                        'order': order,
                        'slave_idx': si,
                        'mismatch': float(mismatch),
                    })
    return hits


def format_spectral(spec, E, eq_label=''):
    """Return list of print lines for spectral analysis + mode choice."""
    lines = []
    header = "SPECTRAL ANALYSIS"
    if eq_label:
        header += f" AT {eq_label}"
    lines.append(header)
    lines.append(f"  orthonormality ||W^H V - I|| = {spec['ortho_error']:.2e}")
    lines.append("")
    lines.append("  eigenvalues (sorted: descending Re, pos Im first in pair):")
    for k, ev in enumerate(spec['Lambda']):
        role = 'master' if k in E['master_idx'] else 'slave'
        re_txt = f"Re={ev.real:+.6f}"
        if abs(ev.imag) > IMAG_TOL:
            T = 2.0 * np.pi / abs(ev.imag)
            im_txt = f"Im={ev.imag:+.6f}  T={T:.4f}"
        else:
            im_txt = "real"
        lines.append(f"    [{k}] {re_txt}  {im_txt}  ({role})")
    lines.append("")
    lines.append("  MASTER SUBSPACE E (2D, complex pair):")
    lines.append(f"    indices        : {E['master_idx']}")
    lines.append(f"    eigenvalues    : {', '.join(f'{ev.real:+.6f}{ev.imag:+.6f}i' for ev in E['Lambda_E'])}")
    lines.append(f"    sigma_in       : {E['sigma_in']}")
    lines.append(f"    sigma_out      : {E['sigma_out']}")
    if E['resonances']:
        lines.append(f"    resonances     : {len(E['resonances'])} hit(s):")
        for hit in E['resonances']:
            lines.append(f"      ({hit['combo'][0]},{hit['combo'][1]}) order {hit['order']}"
                         f"  mismatch={hit['mismatch']:.2e}")
    else:
        lines.append(f"    resonances     : none (orders 2..4, tol=5%)")
    lines.append("")
    lines.append("  RIGHT EIGENVECTORS V_E (columns = master modes):")
    for i in range(E['V_E'].shape[0]):
        parts = []
        for j in range(E['V_E'].shape[1]):
            v = E['V_E'][i, j]
            if abs(v.imag) > 1e-10:
                parts.append(f"{v.real:+.6f}{v.imag:+.6f}i")
            else:
                parts.append(f"{v.real:+.6f}")
        lines.append("    [" + "  " .join(parts) + " ]")
    lines.append("")
    lines.append("  SLAVE EIGENVALUES:")
    for k, ev in zip(E['slave_idx'], E['Lambda_S']):
        if abs(ev.imag) > IMAG_TOL:
            lines.append(f"    [{k}] {ev.real:+.6f}{ev.imag:+.6f}i")
        else:
            lines.append(f"    [{k}] {ev.real:+.6f} (real)")
    return lines
