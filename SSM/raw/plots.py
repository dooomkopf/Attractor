"""SSM/raw/plots.py — Plot-Funktionen für Phase 1.

Phase 1, Schritte 1-2 (jetzt):
    plot_modes      — POD-Moden V[k,:] über die Lag-Achse
    plot_variance   — Scree (log) + kumulative Varianz

Phase 1, Schritt 3 (kommt nach User-Review):
    plot_manifold   — 3D-Datenwolke + Polynom-Surface
"""

import numpy as np
import matplotlib.pyplot as plt


def _flip_modes_for_visualization(Vt, K_show):
    """Vorzeichen pro Mode konsistent flippen, sodass Maximum positiv ist.

    SVD-Vorzeichen ist beliebig, aber für die Optik ist konsistente
    Orientierung schöner. Hat keinen mathematischen Effekt auf den Fit.
    """
    Vt_out = Vt.copy()
    for k in range(min(K_show, Vt.shape[0])):
        i = int(np.argmax(np.abs(Vt_out[k])))
        if Vt_out[k, i] < 0:
            Vt_out[k] = -Vt_out[k]
    return Vt_out


def plot_modes(pca_res, M, TAU, K_show, mode_label='log p'):
    """Fig 1 — POD-Moden V[k, :] über die Lag-Achse j·TAU in Tagen.

    K_show Subachsen, ein Modus pro Achse, sortiert nach Singularwert.
    """
    Vt_flipped = _flip_modes_for_visualization(pca_res.Vt, K_show)

    n_cols = 2
    n_rows = (K_show + 1) // 2
    fig = plt.figure(figsize=(11, 1.6 * n_rows + 1.4))
    lag_days = np.arange(M) * TAU

    for k in range(K_show):
        ax = fig.add_subplot(n_rows, n_cols, k + 1)
        v = Vt_flipped[k]
        ax.plot(lag_days, v, color='#FFAA33', linewidth=1.6)
        ax.fill_between(lag_days, 0, v, color='#FFAA33', alpha=0.18)
        ax.axhline(0, color='#666666', linewidth=0.6, linestyle='--')
        ax.set_title(f'Mode {k+1}  ($\\sigma_k^2/\\Sigma$ = {pca_res.var[k]*100:.2f}\\%)',
                     fontsize=10, color='#E0E0E0')
        ax.set_xlim(lag_days[0], lag_days[-1])
        ax.tick_params(labelsize=9)
        if k >= K_show - 2:
            ax.set_xlabel(r'Lag $j\cdot\tau$ [days]', fontsize=10)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(f'$V_{{{k+1}}}$', fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(f'SSM_raw — POD modes  (signal: {mode_label}, M={M}, τ={TAU}d)',
                     color='#CCCCCC', fontsize=13, y=0.985,
                     fontname='Comfortaa', fontweight='bold')
    plt.subplots_adjust(top=0.92, hspace=0.55, left=0.08, right=0.97, bottom=0.08)
    return fig


def plot_variance(pca_res, K_show, signal_label='log p'):
    """Fig 2 — Varianz-Spektrum (Scree, log-y) + kumulative Varianz."""
    var = pca_res.var
    M   = len(var)
    k   = np.arange(1, M + 1)
    cum = np.cumsum(var)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Linker Plot: per-Mode Varianz
    ax1.bar(k, var * 100, color='#4488FF', alpha=0.85, edgecolor='#2266CC')
    ax1.axvline(K_show + 0.5, color='#FFDD44', linestyle='--', linewidth=1.0,
                label=f'$K={K_show}$ slaved cut')
    ax1.set_yscale('log')
    ax1.set_xlabel('Mode index $k$', fontsize=11)
    ax1.set_ylabel(r'$\sigma_k^2 / \Sigma$  [\%]', fontsize=11)
    ax1.set_title('Per-mode variance (log)', fontsize=11, color='#E0E0E0')
    ax1.set_xlim(0.5, min(M, 25) + 0.5)
    ax1.grid(True, alpha=0.4, linewidth=0.5, which='both')
    ax1.legend(fontsize=9, facecolor='#1A1A1A', edgecolor='#808080',
               labelcolor='#E0E0E0')

    # Rechter Plot: kumulativ
    ax2.plot(k, cum * 100, color='#44FF88', linewidth=2.0,
             marker='o', markersize=4)
    ax2.axhline(99, color='#FFDD44', linestyle=':', linewidth=0.9,
                alpha=0.7, label=r'$99\%$')
    ax2.axvline(K_show + 0.5, color='#FFDD44', linestyle='--', linewidth=1.0)
    ax2.set_xlabel('Mode index $k$', fontsize=11)
    ax2.set_ylabel(r'Cumulative variance [\%]', fontsize=11)
    ax2.set_title('Cumulative variance', fontsize=11, color='#E0E0E0')
    ax2.set_xlim(0.5, min(M, 25) + 0.5)
    ax2.set_ylim(0, 102)
    ax2.grid(True, alpha=0.4, linewidth=0.5)
    ax2.legend(fontsize=9, facecolor='#1A1A1A', edgecolor='#808080',
               labelcolor='#E0E0E0')

    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(f'SSM_raw — variance spectrum  (signal: {signal_label})',
                     color='#CCCCCC', fontsize=13, y=0.985,
                     fontname='Comfortaa', fontweight='bold')
    plt.subplots_adjust(top=0.88, wspace=0.30, left=0.08, right=0.97, bottom=0.13)
    return fig
