"""Module for 15_cli_resample_control.py — plot helpers."""
import numpy as np
from scipy.optimize import curve_fit

from analyze_residuals.constants import DAYS_PER_YEAR

HALVING_DAYS = [1425, 2744, 4146, 5586]
HALVING_LABELS = ['H1', 'H2', 'H3', 'H4']

COLOR_PC1 = '#FFB04A'
COLOR_PC2 = '#B197FC'
COLOR_HALV = '#A0A0A0'
LEGEND_KW = dict(loc='upper left', fontsize=9, facecolor='#1A1A1A',
                 edgecolor='#808080', labelcolor='#E0E0E0', framealpha=0.85)


def frame(ax, color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2.0)


def halving_marks(ax, days_range, x_log10=False, label=None):
    """Vertical halving lines + H1..H4 text labels. label=None -> no legend entry."""
    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    first = True
    for h_day, h_lab in zip(HALVING_DAYS, HALVING_LABELS):
        if h_day < days_range[0] or h_day > days_range[-1]:
            continue
        x_h = np.log10(h_day) if x_log10 else h_day
        ax.axvline(x_h, color=COLOR_HALV, lw=0.9, alpha=0.7, ls='--',
                   label=(label if first else None))
        ax.text(x_h, 0.92, h_lab, transform=trans, color=COLOR_HALV,
                fontsize=9, fontweight='bold', ha='center', va='top',
                bbox=dict(facecolor='#1A1A1A', edgecolor='none', alpha=0.7, pad=1.5))
        first = False


def plot_residual_ts(ax, days, y, color, title, x_log10=False):
    x = np.log10(days) if x_log10 else days
    ax.plot(x, y, color=color, lw=0.6, alpha=0.9)
    halving_marks(ax, days, x_log10=x_log10)
    xl = (r'$\log_{10}(\mathrm{day})$ since Genesis' if x_log10
          else r'day since Genesis')
    ax.set_xlabel(xl, fontsize=10)
    ax.set_ylabel('residual', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25)


def plot_psd_linear(ax, f_y, psd, color, title):
    ax.loglog(f_y, psd, color=color, lw=0.9)
    ax.set_xlabel('frequency [1/year]', fontsize=10)
    ax.set_ylabel('PSD', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, which='both', alpha=0.25)


def plot_psd_log(ax, f, psd, color, title):
    ax.loglog(f, psd, color=color, lw=0.9)
    ax.set_xlabel(r'frequency [cycles/$\log_{10}$d]', fontsize=10)
    ax.set_ylabel('PSD', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, which='both', alpha=0.25)


def _ssm_mode_fit_pc1(ctx, ext, tc_offset=0.0):
    """Free-omega LPPL overlay for PC1.

    Fits

        y(t) = A + B*t^m + C1*t^m*cos(omega*ln(t)) + C2*t^m*sin(omega*ln(t))

    with tc=0 in the past and t = ctx['days_vecs'] > 0. The fit is independent
    of the SSM eigenvalue estimate; ext is accepted for API compatibility only.

    Returns (fit_pc1, T_log10, lam, r2) or (None, None, None, None).
    """
    _ = ext
    pc1_input = ctx.get('_pc1_smoothed_for_fit', None)
    pc1 = (np.asarray(pc1_input, dtype=float) if pc1_input is not None
           else np.asarray(ctx['pc'][:, 0], dtype=float))
    days_raw = np.asarray(ctx['days_vecs'], dtype=float)
    # tc-Shift: dt = days - tc_offset (tc_offset=0 -> Genesis Konvention)
    days = days_raw - tc_offset
    if pc1.ndim != 1 or days.ndim != 1 or len(pc1) != len(days) or len(pc1) < 8:
        return None, None, None, None
    if np.any(~np.isfinite(pc1)) or np.any(~np.isfinite(days)) or np.any(days <= 0):
        return None, None, None, None

    log_days = np.log(days)

    def _r2_score(y_true, y_fit):
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        ss_res = float(np.sum((y_true - y_fit) ** 2))
        return 1.0 - ss_res / (ss_tot + 1e-30)

    def _lppl_model(t, A, B, m, omega, C1, C2):
        tm = np.power(t, m)
        phase = omega * np.log(t)
        return A + B * tm + C1 * tm * np.cos(phase) + C2 * tm * np.sin(phase)

    # omega-Range fix: lambda komplett frei, kein SSM-Bezug
    omega_grid = np.linspace(30.0, 50.0, 10000)
    # Coarse omega-sweep mit FIX m=0.5 -> linearer LSQ
    m0 = 0.5
    tm0 = np.power(days, m0)
    best_omega = None
    best_r2 = -np.inf
    for omega0 in omega_grid:
        phase0 = omega0 * log_days
        X = np.column_stack([
            np.ones_like(pc1),
            tm0,
            tm0 * np.cos(phase0),
            tm0 * np.sin(phase0),
        ])
        coeffs, _, _, _ = np.linalg.lstsq(X, pc1, rcond=None)
        if np.any(~np.isfinite(coeffs)):
            continue
        fit_lin = X @ coeffs
        r2 = _r2_score(pc1, fit_lin)
        if r2 > best_r2:
            best_r2 = r2
            best_omega = omega0

    if best_omega is None:
        return None, None, None, None

    # Refine: full curve_fit mit FREIEM m + omega aus best init
    om_lo, om_hi = float(omega_grid[0]), float(omega_grid[-1])
    bounds = (
        [-np.inf, -np.inf, -1.0, om_lo, -np.inf, -np.inf],
        [np.inf, np.inf, 2.0, om_hi, np.inf, np.inf],
    )
    phase0 = best_omega * log_days
    X = np.column_stack([
        np.ones_like(pc1), tm0,
        tm0 * np.cos(phase0), tm0 * np.sin(phase0),
    ])
    coeffs, _, _, _ = np.linalg.lstsq(X, pc1, rcond=None)
    p0 = np.array([coeffs[0], coeffs[1], m0, best_omega, coeffs[2], coeffs[3]],
                  dtype=float)
    try:
        popt, _ = curve_fit(_lppl_model, days, pc1, p0=p0, bounds=bounds,
                            method='trf', max_nfev=20000)
        fit = _lppl_model(days, *popt)
        if np.any(~np.isfinite(fit)):
            raise RuntimeError('non-finite fit')
        r2 = _r2_score(pc1, fit)
        omega_fit = float(popt[3])
    except Exception:
        # Fallback: linearer LSQ mit best_omega
        fit = X @ coeffs
        r2 = best_r2
        omega_fit = best_omega

    T_log10 = 2.0 * np.pi / omega_fit
    lam = 10.0 ** T_log10
    # N_osc analog bf_filt.py:370 (tc=0 -> ratio = days[-1]/days[0])
    n_osc = (omega_fit / (2.0 * np.pi)) * np.log(days[-1] / days[0])
    return fit, T_log10, lam, r2, omega_fit, n_osc


def plot_pc(ax, ctx, smooth_days=180, legend_loc='upper left', ext=None,
            ctx_unfilt=None):
    """PC1/PC2 + SG smoothed overlay.
    Linear-clock: smooth_days converted to samples directly.
    Log-clock:    smooth_days converted to constant log10-window centered on geo-mean.
    """
    from scipy.signal import savgol_filter
    pc1 = ctx['pc'][:, 0]
    pc2 = ctx['pc'][:, 1]
    days = ctx['days_vecs']
    fit_time = np.asarray(ctx['fit_time_vecs'], dtype=float)
    dt = float(np.median(np.diff(fit_time)))
    if ctx['clock_x_log10']:
        geo_mean = float(np.sqrt(days.min() * days.max()))
        half = smooth_days * 0.5
        win_log = np.log10(geo_mean + half) - np.log10(max(geo_mean - half, 1.0))
        sw = max(1, int(round(win_log / max(dt, 1e-12))))
    else:
        sw = max(1, int(round(smooth_days / max(dt, 1e-12))))
    win = sw if sw % 2 == 1 else sw + 1
    polyorder = 3
    if win < polyorder + 2:
        win = polyorder + 3
    if win >= len(pc1):
        win = (len(pc1) // 2) * 2 - 1
    pc1_s = savgol_filter(pc1, window_length=win, polyorder=polyorder)
    pc2_s = savgol_filter(pc2, window_length=win, polyorder=polyorder)
    x = np.log10(days) if ctx['clock_x_log10'] else days
    if ctx['clock_x_log10']:
        sg_label = f'SG {smooth_days:.0f}d (geo-mean)'
    else:
        sg_label = f'SG {smooth_days:.0f}d'
    show_pc2 = (ext is None)
    ax.plot(x, pc1, color=COLOR_PC1, lw=0.4, alpha=0.35)
    if show_pc2:
        ax.plot(x, pc2, color=COLOR_PC2, lw=0.4, alpha=0.30)
    ax.plot(x, pc1_s, color=COLOR_PC1, lw=1.1, label=f'PC1 ({sg_label})')
    if show_pc2:
        ax.plot(x, pc2_s, color=COLOR_PC2, lw=0.9, alpha=0.85,
                label=f'PC2 ({sg_label})')
    if ctx_unfilt is not None:
        # zusätzliche Linie: gesmoothes UNFILTERED PC1 als Vergleichs-Übersicht
        # (zeigt 'Kollaps' nach pre-whitening), fast schwarz
        pc1_u = np.asarray(ctx_unfilt['pc'][:, 0], dtype=float)
        days_u = np.asarray(ctx_unfilt['days_vecs'], dtype=float)
        fit_time_u = np.asarray(ctx_unfilt['fit_time_vecs'], dtype=float)
        dt_u = float(np.median(np.diff(fit_time_u)))
        if ctx_unfilt['clock_x_log10']:
            geo_u = float(np.sqrt(days_u.min() * days_u.max()))
            half_u = smooth_days * 0.5
            win_log_u = np.log10(geo_u + half_u) - np.log10(max(geo_u - half_u, 1.0))
            sw_u = max(1, int(round(win_log_u / max(dt_u, 1e-12))))
        else:
            sw_u = max(1, int(round(smooth_days / max(dt_u, 1e-12))))
        win_u = sw_u if sw_u % 2 == 1 else sw_u + 1
        if win_u < polyorder + 2:
            win_u = polyorder + 3
        if win_u >= len(pc1_u):
            win_u = (len(pc1_u) // 2) * 2 - 1
        pc1_u_s = savgol_filter(pc1_u, window_length=win_u, polyorder=polyorder)
        x_u = np.log10(days_u) if ctx_unfilt['clock_x_log10'] else days_u
        ax.plot(x_u, pc1_u_s, color='#E0E0E0', lw=1.0, alpha=0.85,
                label=f'PC1 unfiltered (SG {smooth_days:.0f}d)')
    if ext is not None:
        # Single LPPL-Fit ueber alle Daten, tc=0 (Genesis)
        ctx_for_fit = dict(ctx)
        ctx_for_fit['_pc1_smoothed_for_fit'] = pc1_s
        fit_result = _ssm_mode_fit_pc1(ctx_for_fit, ext)
        if fit_result is not None and fit_result[0] is not None:
            fit_pc1, T_fit, lam_fit, r2_fit, omega_fit, n_osc = fit_result
            fit_lbl = (rf'Fit ($\omega$={omega_fit:.2f}, '
                       rf'T={T_fit:.3f}, $\lambda$={lam_fit:.3f}, '
                       rf'$N_{{osc}}$={n_osc:.1f}, $R^2$={r2_fit:.3f})')
            ax.plot(x, fit_pc1, color='#FF6B6B', lw=1.4, alpha=0.9,
                    ls='-', label=fit_lbl)
    halving_marks(ax, days, x_log10=ctx['clock_x_log10'])
    ax.set_xlabel(ctx['clock_label'], fontsize=10)
    ax.set_ylabel('PC amplitude', fontsize=10)
    ax.tick_params(labelsize=9)
    legend_kw = dict(LEGEND_KW); legend_kw['loc'] = legend_loc
    ax.legend(**legend_kw)


def plot_eigs(ax, ext, ctx, scatter_color):
    eigs = ext['eigvals']
    re = np.array([e.real for e in eigs])
    im = np.array([e.imag for e in eigs])
    ax.axhline(0, color='#808080', lw=0.5, alpha=0.6)
    ax.axvline(0, color='#808080', lw=0.5, alpha=0.6)
    ax.scatter(re, im, s=45, c=scatter_color, edgecolors='#FFFFFF',
               linewidth=1.0, zorder=10)
    if ext['lam_main'] is not None:
        a = ext['lam_main'].real
        w = ext['lam_main'].imag
        T = ext['T_main']
        if ctx['clock_x_log10']:
            txt = (rf'$\alpha={a:+.2e}$' + '\n' +
                   rf'$\omega={w:+.2e}$' + '\n' +
                   rf'$T={T:.3f}\;\log_{{10}}\mathrm{{d}}$' + '\n' +
                   rf'$\lambda=10^T={10 ** T:.3f}$')
        else:
            txt = (rf'$\alpha={a:+.2e}$' + '\n' +
                   rf'$\omega={w:+.2e}$' + '\n' +
                   rf'$T={T:.0f}\,\mathrm{{d}}={T / DAYS_PER_YEAR:.2f}\,\mathrm{{y}}$')
        ax.text(0.98, 0.02, txt, transform=ax.transAxes, va='bottom', ha='right',
                fontsize=8, color='#E0E0E0',
                bbox=dict(facecolor='#1A1A1A', edgecolor='#808080', alpha=0.85))
    ax.set_xlabel(r'$\mathrm{Re}(\mu)$', fontsize=10)
    ax.set_ylabel(r'$\mathrm{Im}(\mu)$', fontsize=10)
    ax.tick_params(labelsize=9)
    if len(re) > 0:
        mx = max(np.max(np.abs(re)), 1e-30) * 1.5
        my = max(np.max(np.abs(im)), 1e-30) * 1.5
        ax.set_xlim(-mx, mx)
        ax.set_ylim(-my, my)
