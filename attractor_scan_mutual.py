#!/usr/bin/env python3
"""
attractor_scan_mutual.py — Optimale Embedding-Parameter via Mutual Information.

1) Zeitverzögerte MI: I(τ) = MI(x(t), x(t+τ)) → erstes Minimum = optimales τ
2) Bei festem W (3.77y): M = W/τ + 1
3) Optional: False Nearest Neighbors für M unabhängig von W

Aufruf:
    python3 attractor_scan_mutual.py [--tau_max 200] [--W_years 3.77]
"""

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

plt.style.use('hz.mplstyle')
mpl.rcParams['font.sans-serif'] = ['Comfortaa', 'DejaVu Sans', 'Arial']

# ── PARAMETERS ────────────────────────────────────────────────────────────────
PERCENTILE = 0.01
START_IDX  = 1164
HALVINGS   = [datetime(2012, 11, 28), datetime(2016, 7, 9),
              datetime(2020, 5, 11),  datetime(2024, 4, 20)]


def read_btc_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                data.append((int(parts[0]), float(parts[1]),
                             datetime.strptime(parts[2], "%d.%m.%Y")))
    days   = np.array([d[0] for d in data])
    prices = np.array([d[1] for d in data])
    dates  = np.array([d[2] for d in data])
    return days, prices, dates


def _mutual_info_hist(x, y, n_bins=64):
    """MI via Histogramm-Schätzer (schnell, ausreichend für erste Minimum-Suche)."""
    c_xy = np.histogram2d(x, y, bins=n_bins)[0]
    c_xy = c_xy / c_xy.sum()
    c_x  = c_xy.sum(axis=1)
    c_y  = c_xy.sum(axis=0)
    # MI = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if c_xy[i, j] > 0 and c_x[i] > 0 and c_y[j] > 0:
                mi += c_xy[i, j] * np.log(c_xy[i, j] / (c_x[i] * c_y[j]))
    return mi


def time_delayed_mutual_info(x, tau_max=200, n_bins=32):
    """I(τ) für τ = 1...tau_max."""
    taus = np.arange(1, tau_max + 1)
    mi_vals = np.empty(len(taus))
    for i, tau in enumerate(taus):
        mi_vals[i] = _mutual_info_hist(x[:-tau], x[tau:], n_bins=n_bins)
    return taus, mi_vals


def find_first_minimum(taus, mi_vals, smooth_sigma=3):
    """Erstes lokales Minimum von I(τ) (leicht geglättet).
    Gibt (tau_opt, mi_smooth, found) zurück.  found=False wenn monoton.
    """
    mi_s = gaussian_filter1d(mi_vals, sigma=smooth_sigma)
    for i in range(1, len(mi_s) - 1):
        if mi_s[i] < mi_s[i - 1] and mi_s[i] < mi_s[i + 1]:
            return int(taus[i]), mi_s, True
    return None, mi_s, False


def main():
    ap = argparse.ArgumentParser(description='Embedding via Mutual Information')
    ap.add_argument('--tau_max',  type=int,   default=500)
    ap.add_argument('--W_years',  type=float, default=3.77)
    ap.add_argument('--n_bins',   type=int,   default=32)
    args = ap.parse_args()

    days_all, prices_all, dates_all = read_btc_data('ziel.csv')

    # Quantilregression
    X  = sm.add_constant(np.log(days_all))
    qr = QuantReg(np.log(prices_all), X).fit(q=PERCENTILE)
    log_res_all = np.log(prices_all / np.exp(qr.predict(X)))

    mask_emb = days_all >= START_IDX
    log_res  = log_res_all[mask_emb]
    days_emb = days_all[mask_emb]

    # ── 1) Zeitverzögerte MI ─────────────────────────────────────────────────
    print(f"Berechne MI(τ) für τ = 1...{args.tau_max} ...")
    taus, mi_vals = time_delayed_mutual_info(log_res, tau_max=args.tau_max, n_bins=args.n_bins)
    tau_opt, mi_smooth, found = find_first_minimum(taus, mi_vals)

    # ── 2) M aus festem W ────────────────────────────────────────────────────
    W_days = round(args.W_years * 365)

    print(f"\nErgebnis:")
    if found:
        M_from_W = round(W_days / tau_opt) + 1
        print(f"  τ_opt (erstes MI-Minimum) = {tau_opt} Tage")
        print(f"  M = W/τ + 1 = {M_from_W}")
    else:
        M_from_W = None
        print(f"  KEIN MI-Minimum gefunden (monoton fallend bis τ={args.tau_max})")
    print(f"  W = {W_days} Tage ({args.W_years} Jahre)")
    print(f"  Aktuell: τ=40, M=35 (W=1360)")

    # ── 3) Vergleich: ACF erstes Minimum ─────────────────────────────────────
    acf_vals = np.array([np.corrcoef(log_res[:-t], log_res[t:])[0, 1]
                         for t in range(1, args.tau_max + 1)])
    acf_smooth = gaussian_filter1d(acf_vals, sigma=3)
    tau_acf = None
    for i in range(1, len(acf_smooth) - 1):
        if acf_smooth[i] < acf_smooth[i - 1] and acf_smooth[i] < acf_smooth[i + 1]:
            tau_acf = i + 1
            break
    if tau_acf is None:
        # Kein Minimum → ersten Nulldurchgang nehmen
        zero_cross = np.where(np.diff(np.sign(acf_smooth)))[0]
        tau_acf = int(zero_cross[0] + 1) if len(zero_cross) > 0 else args.tau_max

    print(f"  τ_ACF (erstes ACF-Minimum/Nulldurchgang) = {tau_acf} Tage")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # MI(τ)
    ax1.plot(taus, mi_vals, color='#4488FF', alpha=0.4, linewidth=0.8, label='MI raw')
    ax1.plot(taus, mi_smooth, color='#4488FF', linewidth=2, label='MI smoothed')
    if found:
        ax1.axvline(tau_opt, color='#FF4444', linewidth=2,
                    label=r'$\tau_{{opt}}$ = ' + f'{tau_opt}d')
    else:
        ax1.text(0.5, 0.5, 'no minimum (monotonic)', transform=ax1.transAxes,
                 ha='center', color='#FF4444', fontsize=14)
    ax1.axvline(40, color='#888888', linewidth=1, linestyle='--',
                label=r'current $\tau$ = 40d')
    ax1.set_xlabel(r'$\tau$ [days]')
    ax1.set_ylabel(r'$I(\tau)$ [nats]')
    ax1.set_title('Time-Delayed Mutual Information')
    ax1.legend(fontsize=9, facecolor='#1A1A1A', edgecolor='#808080',
               labelcolor='#E0E0E0')

    # ACF(τ) zum Vergleich
    ax2.plot(taus, acf_vals, color='#44FF44', alpha=0.4, linewidth=0.8, label='ACF raw')
    ax2.plot(taus, acf_smooth, color='#44FF44', linewidth=2, label='ACF smoothed')
    ax2.axvline(tau_acf, color='#FF4444', linewidth=2,
                label=r'$\tau_{{ACF}}$ = ' + f'{tau_acf}d')
    ax2.axvline(40, color='#888888', linewidth=1, linestyle='--',
                label=r'current $\tau$ = 40d')
    ax2.axhline(0, color='#666666', linewidth=0.5)
    ax2.set_xlabel(r'$\tau$ [days]')
    ax2.set_ylabel(r'$A(\tau)$')
    ax2.set_title('Autocorrelation Function (for comparison)')
    ax2.legend(fontsize=9, facecolor='#1A1A1A', edgecolor='#808080',
               labelcolor='#E0E0E0')

    if found:
        _title = f'Embedding Parameter: MI vs ACF  [τ_opt={tau_opt}d → M={M_from_W} at W={args.W_years}y]'
    else:
        _title = f'Embedding Parameter: MI vs ACF  [no MI minimum found, τ_max={args.tau_max}d]'
    with plt.rc_context({'text.usetex': False}):
        fig.suptitle(_title, color='#CCCCCC', fontsize=13, y=0.98,
                     fontname='Comfortaa', fontweight='bold')
    plt.subplots_adjust(top=0.91, hspace=0.35)
    plt.show()


if __name__ == '__main__':
    main()
