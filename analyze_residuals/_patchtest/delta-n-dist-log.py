#!/usr/bin/env python3
"""
Daily n Distribution: Laplace vs Student-t Comparison
Gio-style: Monte-Carlo sampling of n = log(p(t+Δ)/p(t-Δ)) / log((t+Δ)/(t-Δ))
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.stats import t as student_t
from statsmodels.robust.scale import huber

# Style
plt.style.use('hz.mplstyle')
mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']

# ======================== CYCLES (from gold/cycles.json) ========================

HALVINGS = [1425, 2744, 4146, 5586, 7044]  # 1st, 2nd, 3rd, 4th, 5th (est.)

CYCLES = {
    "'13": (1425, 2744),   # 1st-to-2nd halving
    "'17": (2744, 4146),   # 2nd-to-3rd halving
    "'21": (4146, 5586),   # 3rd-to-4th halving
    "'25": (5586, 7044),   # 4th-to-5th halving (current)
}

CYCLE_COLORS = {
    "'13": '#0000FF',   # Blue
    "'17": '#90EE90',   # Light green
    "'21": '#FF69B4',   # Hot pink
    "'25": '#FFD700',   # Gold
}

# Phasen innerhalb eines Zyklus (Tage seit Halving)
# P1 (Hype): 100-550, P2 (Suppression): 550-950, P3 (Relaxation): 0-100 ∪ 950-1400
PHASE_COLORS = {
    'P1': 'green',   # Hype
    'P2': 'red',     # Suppression
    'P3': 'blue',    # Relaxation
    None: '#808080'  # Neutral/grau
}

def get_phase(day):
    """Bestimmt Phase basierend auf Tag seit letztem Halving."""
    # Finde letztes Halving
    halving_day = None
    for h in HALVINGS:
        if day >= h:
            halving_day = h
        else:
            break

    if halving_day is None:
        return None

    days_since = day - halving_day

    if 100 <= days_since < 550:
        return 'P1'  # Hype
    elif 550 <= days_since < 950:
        return 'P2'  # Suppression
    elif days_since < 100 or days_since >= 950:
        return 'P3'  # Relaxation

    return None

# ======================== DATA ========================

DATA_FILE = '/home/hz/Data/ziel.csv'

# Load data: day, price, date
df = pd.read_csv(DATA_FILE, sep=r'\s+', header=None, names=['day', 'price', 'date'])
df = df.sort_values('day').reset_index(drop=True)

days_all = df['day'].values
prices_all = df['price'].values

# Filter: nur ab Tag 1164 (tägliche Daten ohne Lücken)
mask = days_all >= 1164
days = days_all[mask]
prices = prices_all[mask]
N_data = len(days)

print(f"Loaded {N_data} data points (ab Tag 1164)")
print(f"Day range: {days[0]} to {days[-1]}")

# ======================== MONTE-CARLO SAMPLING ========================

N_SAMPLES = 1000000
B_MAX = 365  # |Δ| < 365 days (1 Jahr)

np.random.seed(42)

n_samples = []
sample_phases = []  # Phase für jeden Sample (oder None wenn nicht gleich)

for _ in range(N_SAMPLES * 3):  # oversample to account for rejections
    if len(n_samples) >= N_SAMPLES:
        break

    # Random center index
    i = np.random.randint(0, N_data)

    # Random delta (pos OR neg), exclude 0
    delta = np.random.randint(-B_MAX + 1, B_MAX)
    if delta == 0:
        delta = 1

    # Check bounds for i+delta
    j = i + delta
    if j < 0 or j >= N_data:
        continue  # reject, out of bounds

    t_center = days[i]
    t_delta = days[j]
    p_center = prices[i]
    p_delta = prices[j]

    # Avoid invalid values
    if p_center <= 0 or p_delta <= 0 or t_center <= 0 or t_delta <= 0 or t_delta == t_center:
        continue

    # n = log(p(t+Δ)/p(t)) / log((t+Δ)/t)  -- VZ identisch!
    n = np.log(p_delta / p_center) / np.log(t_delta / t_center)

    if np.isfinite(n):
        n_samples.append(n)
        # Phase: nur wenn beide Punkte in gleicher Phase
        phase1 = get_phase(t_center)
        phase2 = get_phase(t_delta)
        if phase1 == phase2 and phase1 is not None:
            sample_phases.append(phase1)
        else:
            sample_phases.append(None)

n_samples = np.array(n_samples)
sample_phases = np.array(sample_phases)
print(f"\nMonte-Carlo samples: {len(n_samples)}")
print(f"Mean: {np.mean(n_samples):.4f}")
print(f"Std:  {np.std(n_samples):.4f}")

# ======================== HISTOGRAM ========================

# Gio-style: 12 bins per unit, range -100 to 100
n_bins = 200 * 12  # 2400 bins
bins = np.linspace(-100, 100, n_bins + 1)
hist, bin_edges = np.histogram(n_samples, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Filter to range for fitting
in_range = (n_samples >= -100) & (n_samples <= 100)
n_fit = n_samples[in_range]
print(f"In range [-100, 100]: {len(n_fit)} values")

def fit_laplace_robust(x):
    """Huber location + median absolute deviation calibrated for Laplace."""
    x = np.asarray(x)
    mu_hat, _ = huber(x)
    b_hat = np.median(np.abs(x - mu_hat)) / np.log(2)
    return mu_hat, max(b_hat, 1e-12)


def fit_ald_robust(x):
    """Huber location + side-specific median residuals calibrated for ALD."""
    x = np.asarray(x)
    mu_hat, _ = huber(x)
    left = mu_hat - x[x < mu_hat]
    right = x[x >= mu_hat] - mu_hat

    if len(left) < 10 or len(right) < 10:
        b_hat = np.median(np.abs(x - mu_hat)) / np.log(2)
        return {
            'mu': mu_hat,
            'b_left': max(b_hat, 1e-12),
            'b_right': max(b_hat, 1e-12),
            'kappa': 1.0,
            'scale': max(b_hat, 1e-12),
        }

    b_left = max(np.median(left) / np.log(2), 1e-12)
    b_right = max(np.median(right) / np.log(2), 1e-12)
    kappa = np.sqrt(b_left / b_right)
    scale = np.sqrt(b_left * b_right)
    return {
        'mu': mu_hat,
        'b_left': b_left,
        'b_right': b_right,
        'kappa': kappa,
        'scale': scale,
    }


# ======================== LAPLACE FIT (robust) ========================

print("\n" + "="*60)
print("LAPLACE FIT (Huber location + robust Laplace scale)")
print("="*60)

mu_lap, b_lap = fit_laplace_robust(n_fit)
N = len(n_fit)
se_mu = 1.25 * b_lap / np.sqrt(N)
se_b = b_lap / (np.sqrt(N) * np.log(2))

print(f"μ (Huber)       = {mu_lap:.4f} ± {se_mu:.4f}")
print(f"b (robust MAD)  = {b_lap:.4f} ± {se_b:.4f}")

# Laplace PDF
x_pdf = np.linspace(-100, 100, 1000)
laplace_pdf = stats.laplace.pdf(x_pdf, loc=mu_lap, scale=b_lap)

# ======================== STUDENT-T FIT (MLE) ========================

print("\n" + "="*60)
print("STUDENT-T FIT (MLE)")
print("="*60)

# MLE fit: returns (df, loc, scale)
nu_mle, mu_mle, sigma_mle = student_t.fit(n_fit)

print(f"ν (df)    = {nu_mle:.4f}")
print(f"μ (loc)   = {mu_mle:.4f}")
print(f"σ (scale) = {sigma_mle:.4f}")

# Student-t PDF (eigener Fit)
student_pdf_mle = student_t.pdf(x_pdf, df=nu_mle, loc=mu_mle, scale=sigma_mle)

# ======================== PLOT ========================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

bin_width = bins[1] - bins[0]
bin_indices_raw = np.digitize(n_samples, bins) - 1
# KEINE Clip - Werte außerhalb [-100, 100] wegschmeißen
valid_bin_mask = (bin_indices_raw >= 0) & (bin_indices_raw < len(bins) - 1)
bin_indices = bin_indices_raw  # Behalte raw, filtere später mit valid_bin_mask

phase_labels = {
    'P1': r'P1 (Hype): $100 \leq d < 550$',
    'P2': r'P2 (Suppression): $550 \leq d < 950$',
    'P3': r'P3 (Relaxation): $0 \leq d < 100 \;\cup\; 950 \leq d \leq 1400$',
}

# ==================== PLOT 1: Phasen getrennt ====================
BIN_THRESHOLD = 20  # Nur Bins mit > 10 Samples

phase_raw_counts = {phase: np.zeros(len(bin_centers)) for phase in ['P1', 'P2', 'P3']}
phase_totals = {'P1': 0, 'P2': 0, 'P3': 0}

for i, (phase, bin_idx) in enumerate(zip(sample_phases, bin_indices)):
    if valid_bin_mask[i] and phase in phase_raw_counts:
        phase_raw_counts[phase][bin_idx] += 1
        phase_totals[phase] += 1

# Normiere jede Phase zu eigener PDF
phase_counts = {}
for phase in phase_raw_counts:
    if phase_totals[phase] > 0:
        phase_counts[phase] = phase_raw_counts[phase] / (phase_totals[phase] * bin_width)
    else:
        phase_counts[phase] = phase_raw_counts[phase]

# Scatter für jede Phase (eigene PDFs)
# Zuerst: Bins <= BIN_THRESHOLD in grau (niedriges alpha)
for phase in ['P1', 'P2', 'P3']:
    mask_weak = (phase_raw_counts[phase] > 0) & (phase_raw_counts[phase] <= BIN_THRESHOLD)
    if np.any(mask_weak):
        ax1.scatter(bin_centers[mask_weak], phase_counts[phase][mask_weak],
                    s=1, c='#808080', alpha=0.2)

# Dann: Bins > BIN_THRESHOLD in Phasen-Farbe
for phase in ['P1', 'P2', 'P3']:
    mask = phase_raw_counts[phase] > BIN_THRESHOLD
    if np.any(mask):
        ax1.scatter(bin_centers[mask], phase_counts[phase][mask],
                    s=1, c=PHASE_COLORS[phase], alpha=0.7)

# Auch Neutral (grau) plotten
neutral_raw_counts = np.zeros(len(bin_centers))
neutral_total = 0
for i, (phase, bin_idx) in enumerate(zip(sample_phases, bin_indices)):
    if valid_bin_mask[i] and phase is None:
        neutral_raw_counts[bin_idx] += 1
        neutral_total += 1
if neutral_total > 0:
    neutral_counts = neutral_raw_counts / (neutral_total * bin_width)
    # Schwache Bins in grau (niedriges alpha)
    mask_weak = (neutral_raw_counts > 0) & (neutral_raw_counts <= BIN_THRESHOLD)
    ax1.scatter(bin_centers[mask_weak], neutral_counts[mask_weak], s=1, c='#808080', alpha=0.15)
    # Starke Bins in grau (höheres alpha)
    mask = neutral_raw_counts > BIN_THRESHOLD
    ax1.scatter(bin_centers[mask], neutral_counts[mask], s=1, c='#808080', alpha=0.5)

# Separate ALD-Fits für jede Phase - NUR Samples aus Bins > BIN_THRESHOLD
phase_fits = {}
for phase in ['P1', 'P2', 'P3']:
    # Nur Samples aus Bins mit genug Counts
    valid_bins_phase = phase_raw_counts[phase] > BIN_THRESHOLD
    phase_n_fit = []
    for i, (p, bin_idx) in enumerate(zip(sample_phases, bin_indices)):
        if valid_bin_mask[i] and p == phase and valid_bins_phase[bin_idx]:
            phase_n_fit.append(n_samples[i])
    phase_n_fit = np.array(phase_n_fit)

    if len(phase_n_fit) > 100:
        fit_p = fit_ald_robust(phase_n_fit)
        phase_fits[phase] = fit_p

        laplace_p = stats.laplace_asymmetric.pdf(
            x_pdf, fit_p['kappa'], loc=fit_p['mu'], scale=fit_p['scale']
        )
        ax1.plot(
            x_pdf, laplace_p, color=PHASE_COLORS[phase], linewidth=2.5,
            label=(
                f"{phase}: "
                + r'$\mu$=' + f"{fit_p['mu']:.1f}, "
                + r'$b_l$=' + f"{fit_p['b_left']:.1f}, "
                + r'$b_r$=' + f"{fit_p['b_right']:.1f}"
            )
        )

# Neutral Laplace-Fit (grau, hauchfein) - NUR Samples aus Bins > BIN_THRESHOLD
valid_bins_neutral = neutral_raw_counts > BIN_THRESHOLD
neutral_n_fit = []
for i, (p, bin_idx) in enumerate(zip(sample_phases, bin_indices)):
    if valid_bin_mask[i] and p is None and valid_bins_neutral[bin_idx]:
        neutral_n_fit.append(n_samples[i])
neutral_n_fit = np.array(neutral_n_fit)

if len(neutral_n_fit) > 100:
    mu_neut, b_neut = fit_laplace_robust(neutral_n_fit)
    laplace_neut = stats.laplace.pdf(x_pdf, loc=mu_neut, scale=b_neut)
    ax1.plot(x_pdf, laplace_neut, color='#808080', linewidth=1.0, alpha=0.6,
             label=f'Cross-Phase: ' + r'$\mu$=' + f'{mu_neut:.1f}, ' + r'$b$=' + f'{b_neut:.1f}')

ax1.set_xlim(-100, 100)
ax1.set_yscale('log')
ax1.set_ylim(1e-3, 3e-1)
ax1.set_xlabel('n')
ax1.set_ylabel('PDF')
ax1.set_title('Phase-dependent Binning', color='#CCCCCC', fontsize=11)
ax1.legend(loc='upper right', fontsize=9, facecolor='#1A1A1A',
           edgecolor='#808080', labelcolor='#E0E0E0')
ax1.grid(True, alpha=0.3, linestyle='--')

# ==================== PLOT 2: Global mit Phasen-Farben ====================
# Globale PDF, aber jeder Bin-Punkt nach dominanter Phase gefärbt
global_phase_counts = {phase: np.zeros(len(bin_centers)) for phase in ['P1', 'P2', 'P3', None]}

for i, (phase, bin_idx) in enumerate(zip(sample_phases, bin_indices)):
    if valid_bin_mask[i]:
        global_phase_counts[phase][bin_idx] += 1

# Für jeden Bin: welche Phase dominiert?
bin_colors = []
for i in range(len(bin_centers)):
    counts = {p: global_phase_counts[p][i] for p in ['P1', 'P2', 'P3', None]}
    dominant = max(counts, key=counts.get)
    bin_colors.append(PHASE_COLORS[dominant])

# Scatter mit globaler PDF-Höhe, aber Phasen-Farben (KEINE Labels)
# P3 (blau) zuletzt zeichnen, damit es oben liegt
for phase in [None, 'P1', 'P2', 'P3']:
    mask = np.array([bin_colors[i] == PHASE_COLORS[phase] for i in range(len(bin_centers))])
    mask &= hist > 0
    if np.any(mask):
        ax2.scatter(bin_centers[mask], hist[mask], s=1, c=PHASE_COLORS[phase], alpha=0.7)

# Fits mit LaTeX-Labels (Farben wie daily-n-prob-v3.py)
ax2.plot(x_pdf, laplace_pdf, color='#CCFF00', linewidth=1.3, alpha=0.7,
         label=r'Laplace: $\mu$=' + f'{mu_lap:.2f}, ' + r'$b$=' + f'{b_lap:.2f}')
ax2.plot(x_pdf, student_pdf_mle, color='#FFFFFF', linewidth=1.5, linestyle='--', alpha=0.9,
         label=r'Student-t: $\mu$=' + f'{mu_mle:.2f}, ' + r'$\sigma$=' + f'{sigma_mle:.2f}, ' + r'$\nu$=' + f'{nu_mle:.2f}')
ax2.set_xlim(-100, 100)
ax2.set_yscale('log')
ax2.set_ylim(1e-3, 3e-1)
ax2.set_xlabel('n')
ax2.set_ylabel('PDF')
ax2.set_title('Global Binning', color='#CCCCCC', fontsize=11)
ax2.legend(loc='upper right', fontsize=9, facecolor='#1A1A1A',
           edgecolor='#808080', labelcolor='#E0E0E0')
ax2.grid(True, alpha=0.3, linestyle='--')

# LINKS: Info für Plot 2 (OHNE Formel - die ist nur oben)
info_text2 = (
    f"Samples: {N_SAMPLES:,}\n"
    r"$|\Delta| < " + f"{B_MAX}$" + " Tage"
)
ax2.text(0.02, 0.98, info_text2, transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='#1A1A1A', edgecolor='#808080', alpha=0.9),
         color='#E0E0E0')

# LINKS OBEN: Cycle-Phases
phase_text = (
    "Cycle-Phases  ($d$ = days since last halving)\n"
    r"P1 (Hype): $100 \leq d < 550$" + "\n"
    r"P2 (Suppression): $550 \leq d < 950$" + "\n"
    r"P3 (Relaxation): $0 \leq d < 100 \;\cup\; 950 \leq d \leq 1400$"
)
ax1.text(0.02, 0.98, phase_text, transform=ax1.transAxes, fontsize=8,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='#1A1A1A', edgecolor='#808080', alpha=0.9),
         color='#E0E0E0')

# LINKS MITTE: Formel (GROSS)
ax1.text(0.02, 0.55, r"$n = \frac{\log(p(t+\Delta)/p(t))}{\log((t+\Delta)/t)}$",
         transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='#1A1A1A', edgecolor='#808080', alpha=0.9),
         color='#E0E0E0')

# LINKS UNTEN: Infos
info_text = (
    f"Samples: {N_SAMPLES:,}\n"
    r"$|\Delta| < " + f"{B_MAX}$" + " Tage"
)
ax1.text(0.02, 0.35, info_text, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='#1A1A1A', edgecolor='#808080', alpha=0.9),
         color='#E0E0E0')

# Title
with plt.rc_context({'text.usetex': False}):
    plt.suptitle("Distribution of Bitcoin's Power-Law Exponent",
                 color='#CCCCCC', fontsize=13, y=0.975,
                 fontname='Comfortaa', fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('delta-n-dist-log.png', dpi=300, facecolor='#0a0a0a')
plt.show()

print(f"\nSaved: delta-n-dist-log.png")
