"""Reduced dynamics in polar form and backbone curve from SSM coefficients."""

import numpy as np

from .constants import IMAG_TOL


def polar_reduced_dynamics(E, whisker_result):
    """Convert reduced dynamics R(p) to polar form: rho_dot, theta_dot.

    For a 2D SSM with complex pair lambda = alpha +/- i*omega:
      p1 = rho * exp(i*theta),  p2 = rho * exp(-i*theta)
      rho_dot = a(rho) * rho
      theta_dot = omega(rho)

    At order 1 (R2=0): a(rho) = Re(lambda), omega(rho) = Im(lambda)
    At order 3: a(rho) = Re(lambda) + Re(gamma1)*rho^2
                omega(rho) = Im(lambda) + Im(gamma1)*rho^2

    Returns dict with alpha, omega, gamma coefficients.
    """
    lam = E['Lambda_E'][0]
    alpha_0 = float(lam.real)
    omega_0 = float(abs(lam.imag))

    R2 = whisker_result['R2']
    gamma_coeffs = []

    r2_21 = R2.get((2, 1))
    if r2_21 is not None and abs(r2_21[0]) > 1e-15:
        gamma_coeffs.append(complex(r2_21[0]))

    return {
        'alpha_0': alpha_0,
        'omega_0': omega_0,
        'gamma': gamma_coeffs,
        'n_gamma': len(gamma_coeffs),
        'R2_all_zero': all(np.max(np.abs(v)) < 1e-12 for v in R2.values()),
    }


def backbone_curve(polar, rho_max=1.0, n_points=100):
    """Compute backbone omega(rho) and growth rate alpha(rho)."""
    rho = np.linspace(0, rho_max, n_points)
    omega = np.full_like(rho, polar['omega_0'])
    alpha = np.full_like(rho, polar['alpha_0'])

    for k, gamma_k in enumerate(polar['gamma']):
        omega += np.imag(gamma_k) * rho ** (2 * (k + 1))
        alpha += np.real(gamma_k) * rho ** (2 * (k + 1))

    return {'rho': rho, 'omega': omega, 'alpha': alpha}


def format_backbone(polar):
    """Return list of print lines for polar reduced dynamics."""
    lines = []
    lines.append("REDUCED DYNAMICS IN POLAR FORM")
    lines.append("")
    lines.append("  p1 = rho*exp(i*theta),  p2 = conj(p1)")
    lines.append("  rho_dot   = alpha(rho) * rho")
    lines.append("  theta_dot = omega(rho)")
    lines.append("")
    lines.append(f"  alpha_0 (growth rate)   : {polar['alpha_0']:+.6f}   {'[unstable, Re>0]' if polar['alpha_0'] > 0 else '[stable, Re<0]'}")
    lines.append(f"  omega_0 (frequency)     : {polar['omega_0']:.6f} rad/t   T={2*np.pi/polar['omega_0']:.4f}")
    lines.append(f"  gamma coefficients      : {polar['n_gamma']}")
    lines.append(f"  R2 identically zero     : {polar['R2_all_zero']}")
    lines.append("")

    if polar['R2_all_zero']:
        lines.append("  BACKBONE: flat (omega independent of amplitude)")
        lines.append("    omega(rho) = omega_0 = const")
        lines.append("    alpha(rho) = alpha_0 = const")
        lines.append("")
        lines.append("  WHY FLAT:")
        lines.append("    R2 = 0 means the reduced dynamics is purely linear at order 2.")
        lines.append("    All nonlinear content (incl. 2w) lives in the SSM geometry W2,")
        lines.append("    not in the dynamics on the manifold.")
        lines.append("    The trajectory spirals outward at constant frequency omega_0.")
        lines.append("    Amplitude-dependent corrections would appear at order 3+")
        lines.append("    (from quadratic NL composed with W2).")
    else:
        for k, g in enumerate(polar['gamma']):
            lines.append(f"  gamma_{k+1} = {g.real:+.6e} {g.imag:+.6e}i")
            lines.append(f"    -> freq shift : Im(gamma_{k+1})*rho^{2*(k+1)}")
            lines.append(f"    -> growth shift: Re(gamma_{k+1})*rho^{2*(k+1)}")

    return lines
