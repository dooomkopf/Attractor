#!/usr/bin/env python3
"""surrogate_ssm.py — IAAFT-Surrogate-Test für SSM_res (BTC-Residuen).

Engste Analogie zu surrogates/surrogates_theiler_test.py, aber:
    * verwendet die SSM/res/-Pipeline (data, embedding, geometry, phase)
    * primäre Teststatistik Q = R²_total des Polynom-Fits slaved PCs ≈
      W(PC1, PC2)  (geometry.fit_geometry, default order=3, K=5)
    * sekundär: phase_pc3 = R²(PC3 ~ sin 2θ + cos 2θ) zur Vergleichbarkeit
    * --loc segmentiert IAAFT an den Halvings (Stationarität pro Cycle)
    * --norm verwendet das peak-decay-normierte Signal aus data.load_data
    * Kontextlogik in surrogate_compute.compute_original_only +
      compute_surrogate_payload — surrogate_ssm.py ist nur Entry-Point.

Aufruf:
    python3 surrogate_ssm.py [--loc] [--norm] [--n_surr 999]
                             [--M 35] [--K 5] [--order 3]
                             [--no-plot]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# SSM/res/ Geschwister verfügbar machen
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from SSM_res_surrogate_compute import (
    compute_original_only, compute_surrogate_payload, _phase_pc3_corr,
)
from SSM_res_geometry import fit_geometry, poly_features_2d


_STYLEFILE = '/home/hz/Data/Attractor/hz.mplstyle'
_ZIEL_DEF  = '/home/hz/Data/Attractor/ziel.csv'

if os.path.exists(_STYLEFILE):
    plt.style.use(_STYLEFILE)
# Comfortaa wird NUR in Titeln verwendet (fontname='Comfortaa' am set_title /
# suptitle), nicht global. Achsen, Legenden etc. behalten den Style-Font.


def _p_rank(q_orig, q_surr):
    """Einseitiger Rang-p-Wert (oberer Tail). p = (1 + #{surr >= orig}) / (n+1).
    Identisch zu surrogates_theiler_test._p_rank.
    """
    q_s = q_surr[np.isfinite(q_surr)]
    rank = int(np.sum(q_s >= q_orig)) + 1
    return rank / (len(q_s) + 1)


def _p_rank_two_sided(q_orig, q_surr):
    """Zweiseitiger Rang-p-Wert: 2 * min(p_upper, p_lower), gekappt auf 1.
    Sinnvoll für Statistiken die in beide Richtungen abweichen können.
    """
    q_s = q_surr[np.isfinite(q_surr)]
    n   = len(q_s)
    if n == 0:
        return float('nan')
    p_up = (int(np.sum(q_s >= q_orig)) + 1) / (n + 1)
    p_lo = (int(np.sum(q_s <= q_orig)) + 1) / (n + 1)
    return min(1.0, 2.0 * min(p_up, p_lo))


def main():
    ap = argparse.ArgumentParser(description='SSM Surrogate Test (IAAFT)')
    ap.add_argument('--loc',    action='store_true', help='IAAFT pro Halving-Segment')
    ap.add_argument('--norm',   action='store_true', help='Peak-decay-Normierung des Signals')
    ap.add_argument('--n_surr', type=int, default=999)
    ap.add_argument('--M',      type=int, default=35)
    ap.add_argument('--tau',    type=int, default=41,
                    help='Default 41d (konsistent mit SSM_res.py)')
    ap.add_argument('--K',      type=int, default=3,
                    help='Anzahl slaved PCs (default 3; SSM-Signal sitzt in PC3..PC5)')
    ap.add_argument('--order',  type=int, default=3, help='Polynom-Grad (default 3)')
    ap.add_argument('--phase-pcs', type=str, default='4,5',
                    help='Slaved PCs für die PRIMÄRE Q-Statistik phase_R2_sum '
                         '(comma-sep, default "4,5" — Variante C). '
                         'Beispiel: "5" für single-PC PC5, "3,4,5" für alle drei.')
    ap.add_argument('--diag-pcs',  type=str, default='3,4,5',
                    help='Slaved PCs für phase_R2-Diagnose-Spalten (default "3,4,5")')
    ap.add_argument('--phase-pc', type=int, default=4,
                    help='Welche slaved PC in Fig 2 vs θ plotten (default 4)')
    ap.add_argument('--sigma',  type=float, default=60, help='Smoothing σ für Plots')
    ap.add_argument('--n_iter', type=int, default=50, help='IAAFT Iterationen')
    ap.add_argument('--start-idx', type=int, default=1164)
    ap.add_argument('--filename',  type=str, default=_ZIEL_DEF)
    ap.add_argument('--no-plot',   action='store_true')
    args = ap.parse_args()

    _M   = args.M
    _TAU = args.tau
    PHASE_PCS = tuple(int(s) for s in args.phase_pcs.split(',') if s.strip())
    DIAG_PCS  = tuple(int(s) for s in args.diag_pcs.split(',')  if s.strip())

    params = {
        'PERCENTILE':   0.01,
        'START_IDX':    args.start_idx,
        'M':            _M,
        'TAU':          _TAU,
        'K':            args.K,
        'ORDER':        args.order,
        'PHASE_PCS':    PHASE_PCS,
        'DIAG_PCS':     DIAG_PCS,
        'SMOOTH_SIGMA': args.sigma,
        'N_ITER':       args.n_iter,
        'loc':          args.loc,
        'norm':         args.norm,
        'filename':     args.filename,
    }

    mode = 'loc' if args.loc else 'glob'
    norm_tag = '  +norm' if args.norm else ''
    pcs_tag  = ','.join(f'PC{j}' for j in PHASE_PCS)
    print(f"SSM Surrogate Test  [{mode}{norm_tag}]"
          f"  M={_M}  τ={_TAU}  K={args.K}  order={args.order}"
          f"  Q=R²_F({{{pcs_tag}}}|sin2θ,cos2θ)"
          f"  n_surr={args.n_surr}")
    print()

    # ── Original ──────────────────────────────────────────────────────────────
    print("Computing original...")
    orig = compute_original_only(params)
    Q_orig = orig['Q_orig']

    q_orig_frob_own  = Q_orig['phase_R2_frob']
    q_orig_frob_orig = Q_orig['phase_R2_frob_orig']
    q_orig_sum_own   = Q_orig['phase_R2_sum']
    q_orig_sum_orig  = Q_orig['phase_R2_sum_orig']

    print(f"  PRIMARY phase_R2_frob (Σ over {pcs_tag}, multivariate Frobenius)")
    print(f"    own basis  = {q_orig_frob_own:.4f}")
    print(f"  diagnose phase_R2_sum  = {q_orig_sum_own:.4f}")
    print(f"  phase_R2 per PC (own basis):")
    for j in DIAG_PCS:
        v = Q_orig['phase_R2_per_pc'].get(j, float('nan'))
        marker = '  ← in PRIMARY frob' if j in PHASE_PCS else ''
        print(f"    PC{j}: {v:.4f}{marker}")
    print(f"  R²_total polynom fit (diag) = {Q_orig['R2_total']:.4f}")
    print(f"  R² per slaved PC fit (PC3..PC{2 + args.K}) = "
          f"[{'  '.join(f'{r:.3f}' for r in Q_orig['R2_per_pc'])}]")
    print()

    # ── Surrogate ─────────────────────────────────────────────────────────────
    surr_frob_own, surr_frob_orig = [], []
    surr_sum_own,  surr_sum_orig  = [], []
    surr_per_pc_own  = {j: [] for j in DIAG_PCS}
    surr_per_pc_orig = {j: [] for j in DIAG_PCS}
    surr_r2_total_own  = []
    surr_r2_total_orig = []

    for seed in range(args.n_surr):
        pl = compute_surrogate_payload(orig, seed, params)
        Qs = pl['Q_surr']
        surr_frob_own.append (Qs['phase_R2_frob'])
        surr_frob_orig.append(Qs['phase_R2_frob_orig'])
        surr_sum_own.append (Qs['phase_R2_sum'])
        surr_sum_orig.append(Qs['phase_R2_sum_orig'])
        for j in DIAG_PCS:
            surr_per_pc_own[j].append (Qs['phase_R2_per_pc'].get(j, np.nan))
            surr_per_pc_orig[j].append(Qs['phase_R2_per_pc_orig'].get(j, np.nan))
        surr_r2_total_own.append (Qs['R2_total'])
        surr_r2_total_orig.append(Qs['R2_total_orig'])
        if (seed + 1) % 100 == 0 or seed == args.n_surr - 1:
            print(f"  {seed+1}/{args.n_surr}", flush=True)

    surr_frob_own = np.array(surr_frob_own)
    surr_frob_orig = np.array(surr_frob_orig)
    surr_sum_own  = np.array(surr_sum_own)
    surr_sum_orig = np.array(surr_sum_orig)
    surr_per_pc_own  = {j: np.array(v) for j, v in surr_per_pc_own.items()}
    surr_per_pc_orig = {j: np.array(v) for j, v in surr_per_pc_orig.items()}
    surr_r2_total_own  = np.array(surr_r2_total_own)
    surr_r2_total_orig = np.array(surr_r2_total_orig)

    # ── Ergebnisse ────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"SSM Surrogate Test  n={args.n_surr}  [{mode}{norm_tag}]")
    print(f"PRIMARY Q = Frobenius-R² der Regression Y={{{pcs_tag}}} ~ sin 2θ + cos 2θ")
    print(f"           (rotations-invariant in der ({pcs_tag})-Ebene, upper-tail)")
    print(f"{'='*72}\n")

    def _summary(name, q_o, q_s, two_sided=False):
        q_clean = q_s[np.isfinite(q_s)]
        mu    = float(np.mean(q_clean))
        sigma = float(np.std(q_clean))
        S     = (q_o - mu) / sigma if sigma > 0 else 0.0
        p     = _p_rank_two_sided(q_o, q_s) if two_sided else _p_rank(q_o, q_s)
        tag   = '2s' if two_sided else '1s'
        print(f"  {name:34s}  orig={q_o:.4f}  μ={mu:.4f}  σ={sigma:.4f}"
              f"  S={S:+.2f}σ  p_{tag}={p:.4f}")
        return mu, sigma, S, p

    print(f"  ── PRIMARY phase_R2_frob (Frobenius über {{{pcs_tag}}}, upper-tail) ──")
    _summary('Own basis',           q_orig_frob_own,  surr_frob_own)
    _summary('Original basis [diag]', q_orig_frob_orig, surr_frob_orig)
    print()
    print(f"  ── DIAGNOSE: phase_R2_sum (achsenabhängig, vergleich zu frob) ──")
    _summary('Own basis (sum)',           q_orig_sum_own,  surr_sum_own)
    _summary('Original basis (sum)',      q_orig_sum_orig, surr_sum_orig)
    print()
    print(f"  ── DIAGNOSE: phase_R2 per PC (own basis, upper-tail) ──")
    for j in DIAG_PCS:
        mark = '  ← in PRIMARY frob' if j in PHASE_PCS else ''
        q_o  = Q_orig['phase_R2_per_pc'].get(j, float('nan'))
        _summary(f'PC{j}{mark}', q_o, surr_per_pc_own[j])
    print()
    print(f"  ── DIAGNOSE: phase_R2 per PC (orig basis) ──")
    for j in DIAG_PCS:
        q_o = Q_orig['phase_R2_per_pc_orig'].get(j, float('nan'))
        _summary(f'PC{j}', q_o, surr_per_pc_orig[j])
    print()
    print(f"  ── DIAGNOSE: Polynom-Manifold R²_total ──")
    _summary('Own basis (R²_total)',          Q_orig['R2_total'], surr_r2_total_own)
    _summary('Original basis (R²_total)',     Q_orig['R2_total_orig'], surr_r2_total_orig)
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.no_plot:
        return

    opts = f'--{mode}'
    if args.norm:
        opts += ' --norm'

    # ── Fig 1: Histogramme (phase_R2_frob primär = Variante C-frob) ─────────
    # Konvention: Achsenlabels = LaTeX (Formeln). Titel = plain text (kein $).
    # hz.mplstyle hat global text.usetex=True; Titel werden lokal auf False
    # gewrapped, damit darin freier Text + Sonderzeichen funktionieren.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, name, q_o, q_s in [
        (ax1, 'Own PCA Basis',      q_orig_frob_own,  surr_frob_own),
        (ax2, 'Original PCA Basis', q_orig_frob_orig, surr_frob_orig),
    ]:
        q_clean = q_s[np.isfinite(q_s)]
        p = _p_rank(q_o, q_s)

        ax.hist(q_clean, bins=40, color='#4488FF', alpha=0.7, edgecolor='#2266CC',
                label=f'Surrogate ($n$={len(q_clean)})')
        ax.axvline(q_o, color='#FF4444', linewidth=2.5, linestyle='-',
                   label=r'$\mu_{\mathrm{orig}}$ = ' + f'{q_o:.3f}')
        ax.axvline(np.mean(q_clean), color='#888888', linewidth=1, linestyle='--',
                   label=r'$\mu_{\mathrm{surr}}$ = ' + f'{np.mean(q_clean):.3f}')
        ax.plot([], [], ' ', label=r'$\sigma_{\mathrm{surr}}$ = ' + f'{np.std(q_clean):.3f}',
                color='#666666')

        pcs_math = ','.join(f'\\mathrm{{PC{j}}}' for j in PHASE_PCS)
        ax.set_xlabel(rf'$R^2_F\left(\{{{pcs_math}\}} \sim \sin 2\theta + \cos 2\theta\right)$')
        ax.set_ylabel('Count')

        mu = np.mean(q_clean)
        sigma = np.std(q_clean)
        S = abs(q_o - mu) / sigma if sigma > 0 else 0
        N_clean = len(q_clean) + 1
        rank = int(np.sum(q_clean >= q_o)) + 1
        y_arrow = ax.get_ylim()[1] * 0.70
        ax.annotate('', xy=(q_o, y_arrow), xytext=(mu, y_arrow),
                    arrowprops=dict(arrowstyle='<->', color='#FFDD44',
                                   linewidth=0.8))
        with plt.rc_context({'text.usetex': False}):
            ax.text((mu + q_o) / 2, y_arrow * 1.05,
                    f'rank {rank}/{N_clean}',
                    ha='center', color='#FFDD44', fontsize=10, fontweight='bold')
            ax.text((mu + q_o) / 2, y_arrow * 0.85,
                    f'({S:.1f} sigma Gauss approx.)',
                    ha='center', color='#666666', fontsize=9)
            ax.set_title(f'{name}    p_rank = {p:.3f}',
                         fontname='Comfortaa')
        ax.legend(fontsize=9, facecolor='#1A1A1A', edgecolor='#808080',
                  labelcolor='#E0E0E0')

    pcs_str = ','.join(f'PC{j}' for j in PHASE_PCS)
    with plt.rc_context({'text.usetex': False}):
        plt.suptitle(f'SSM Residuals: Surrogate Test (IAAFT)  '
                     f'[{opts} --n_surr {args.n_surr} --phase-pcs {args.phase_pcs}]',
                     color='#CCCCCC', fontsize=13, y=0.98,
                     fontname='Comfortaa', fontweight='bold')
    pcs_math2 = ','.join(f'\\mathrm{{PC{j}}}' for j in PHASE_PCS)
    fig.text(0.5, 0.90,
             rf'$Q = R^2_F\,\left(Y \sim \sin 2\theta + \cos 2\theta\right),'
             rf'\ \ Y = \{{{pcs_math2}\}},'
             rf'\ \ \theta = \mathrm{{atan2}}(\mathrm{{PC2}}, \mathrm{{PC1}})$',
             ha='center', color='#AAAAAA', fontsize=10)
    plt.subplots_adjust(top=0.84)

    # ── Fig 2: PC{phase_pc} = f(θ) — Original + 7 Surrogate ──────────────────
    # Wichtig (Codex 2026-04-07):
    #  * Original und Surrogate verwenden BEIDE die anchored intrinsic phase
    #    aus phase.compute_phase_full → Panels sind direkt vergleichbar.
    #  * Bounds (Cycle-Färbung) werden pro Panel aus seiner EIGENEN Phase
    #    abgeleitet — die Surrogate haben i.A. andere intrinsische Bounds.
    #  * Default --phase-pc 4 weil das SSM-Signal in PC4/PC5 sitzt; --phase-pc 3
    #    reproduziert die direkte Vorlagentreue.
    _CYC_COLORS     = ['#888888', '#FF4444', '#44FF44', '#4488FF', '#FFAA44']
    _CYC_COLORS_DIM = ['#444444', '#772222', '#227722', '#223388', '#775522']
    _CYC_LABELS     = ['Cycle 0', 'Cycle 1', 'Cycle 2', 'Cycle 3', 'Cycle 4']

    pc_idx = args.phase_pc - 1   # PC4 → Index 3
    pc_lbl = f'PC{args.phase_pc}'

    def _fit_curve(theta, y):
        th_grid = np.linspace(0, 2 * np.pi, 200)
        X = np.column_stack([np.sin(2 * theta), np.cos(2 * theta)])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        X_grid = np.column_stack([np.sin(2 * th_grid), np.cos(2 * th_grid)])
        return th_grid, X_grid @ beta

    def _plot_panel(ax, theta, y, bounds_loc, R2, title, title_col,
                    colors, alpha_pts, show_legend=False, is_surrogate=False):
        n_segs_loc = max(1, len(bounds_loc) - 1)
        for k in range(n_segs_loc):
            s0 = bounds_loc[k]
            s1 = min(bounds_loc[k + 1] + 1, len(theta))
            col = colors[k] if k < len(colors) else '#CCCCCC'
            lbl = _CYC_LABELS[k] if k < len(_CYC_LABELS) else f'Cycle {k}'
            ax.scatter(theta[s0:s1], y[s0:s1],
                       s=2, alpha=alpha_pts, color=col, label=lbl)
        th_g, fit_g = _fit_curve(theta, y)
        fit_col = '#888888' if is_surrogate else 'white'
        ax.plot(th_g, fit_g, color=fit_col, linewidth=2.5,
                label=(r'$Q_{H_0}$' if is_surrogate else r'$Q$'))
        with plt.rc_context({'text.usetex': False}):
            tag = 'Q_H0' if is_surrogate else 'Q'
            ax.set_title(f'{title}    {tag} = {R2:.3f}',
                         color=title_col, fontname='Comfortaa')
        ax.set_xlabel(r'$\theta$')
        if show_legend:
            ax.legend(fontsize=7, facecolor='#1A1A1A', edgecolor='#808080',
                      labelcolor='#E0E0E0', markerscale=3, loc='best')

    fig2, axes = plt.subplots(2, 4, figsize=(18, 8))

    # ── Original (oben links) ─────────────────────────────────────────────────
    pc_raw       = orig['pc']
    th_orig_anc  = orig['theta_orig']         # geankert via compute_phase_full
    bounds_orig  = orig['bounds']
    y_orig       = pc_raw[:, pc_idx]
    R2_o = _phase_pc3_corr(th_orig_anc, y_orig)

    _plot_panel(axes[0, 0], th_orig_anc, y_orig, bounds_orig, R2_o,
                'Original', '#44FF44', _CYC_COLORS, 0.4, show_legend=True)
    axes[0, 0].set_ylabel(f'$\\mathrm{{{pc_lbl}}}$')

    # ── 7 Surrogate ──────────────────────────────────────────────────────────
    rng_plot = np.random.default_rng(42)
    n_pick = min(7, args.n_surr)
    seeds_plot = rng_plot.choice(args.n_surr, size=n_pick, replace=False)

    surr_axes = [axes[0, 1], axes[0, 2], axes[0, 3],
                 axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]]

    for i, seed in enumerate(seeds_plot):
        pl = compute_surrogate_payload(orig, int(seed), params)
        th_s     = pl['theta']            # eigene anchored phase
        bnds_s   = pl['bounds']           # eigene intrinsic bounds
        y_s      = pl['pcs_own'][:, pc_idx]
        R2_s     = _phase_pc3_corr(th_s, y_s)

        _plot_panel(surr_axes[i], th_s, y_s, bnds_s, R2_s,
                    f'Surrogate \\#{seed}', '#888888', _CYC_COLORS_DIM, 0.3,
                    is_surrogate=True)
        if i in (0, 3):
            surr_axes[i].set_ylabel(f'$\\mathrm{{{pc_lbl}}}$')

    with plt.rc_context({'text.usetex': False}):
        fig2.suptitle(f'Discriminating Statistic: {pc_lbl} = f(\u03b8)  '
                      f'[{opts} --n_surr {args.n_surr} --phase-pc {args.phase_pc}]',
                      color='#CCCCCC', fontsize=13, y=0.98,
                      fontname='Comfortaa', fontweight='bold')
    plt.subplots_adjust(top=0.88, hspace=0.50)

    fig.savefig (os.path.join(_HERE, 'surrogate_ssm_fig1.png'), dpi=150, facecolor='#0a0a0a')
    fig2.savefig(os.path.join(_HERE, 'surrogate_ssm_fig2.png'), dpi=150, facecolor='#0a0a0a')
    print(f"Saved: surrogate_ssm_fig1.png, surrogate_ssm_fig2.png")
    plt.show()


if __name__ == '__main__':
    main()
