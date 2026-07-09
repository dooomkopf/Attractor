"""SSM/res/data.py — Daten laden + QuantReg-Detrending → log-Residuen.

Pipeline-Schritte (analog attractor.py:105-112):
    1) read_btc_data  → days, prices, dates
    2) Quantilregression auf log(price) vs log(day) bei q=PERCENTILE
    3) residuals_all = price / exp(qr_fit)
    4) log_res_all   = log(residuals_all)
    5) Cut ab start_idx
"""

from datetime import datetime
import sys

import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg


# BTC Halvings — Datum → wird auf den nächsten Tagesindex aus ziel.csv gemappt.
_HALVINGS = [datetime(2012, 11, 28), datetime(2016,  7,  9),
             datetime(2020,  5, 11), datetime(2024,  4, 20)]

# Cycle-Tops aus gold/cycles.json — Labels Tk = Halving-Cycle-Index:
#   T0 = pre-1st-halving '11-Peak (liegt vor start_idx, im Embedding nicht sichtbar)
#   T1 = '13-Cycle (1st→2nd halving)
#   T2 = '17-Cycle (2nd→3rd halving)
#   T3 = '21-Cycle (3rd→4th halving)
#   T4 = '25-Cycle (4th→5th halving)
_CYCLE_TOPS = [
    (datetime(2011, 11,  9), "Top0"),
    (datetime(2013, 12,  4), "Top1"),
    (datetime(2017, 12, 16), "Top2"),
    (datetime(2021, 11,  9), "Top3"),
    (datetime(2025, 10,  7), "Top4"),
]


def read_btc_data(filename):
    """Liest ziel.csv: <day> <price> <dd.mm.yyyy>. (aus attractor.py:63-74)"""
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


def _peak_decay_fit(days_all, residuals_all, halving_days):
    """Fit exp. Decay durch die Cycle-Peaks zwischen Halvings.

    Wie surrogates_n_IAAFT_compute.py:155-166. Liefert exp_fit-Funktion.
    """
    halv_bounds = np.concatenate([[days_all[0]], halving_days, [days_all[-1]]])
    pk_days = []
    pk_vals = []
    for k in range(len(halv_bounds) - 1):
        m = (days_all >= halv_bounds[k]) & (days_all < halv_bounds[k + 1])
        if m.any():
            imax = int(np.argmax(residuals_all[m]))
            pk_days.append(int(days_all[m][imax]))
            pk_vals.append(float(residuals_all[m][imax]))
    pk_days = np.array(pk_days)
    pk_vals = np.array(pk_vals)
    coef = np.polyfit(pk_days, np.log(pk_vals), 1)   # log-linear
    def exp_fit(d, c=coef):
        return np.exp(c[1] + c[0] * d)
    return exp_fit, pk_days, pk_vals


def load_data(filename, start_idx, percentile=0.01, norm=False):
    """Lädt ziel.csv, wendet QuantReg-Detrending an, gibt Signal zurück.

    Args:
        filename:    absoluter Pfad zu ziel.csv
        start_idx:   Tag-Index ab dem ausgewertet wird (default 1164)
        percentile:  Quantil für QuantReg (default 0.01 wie attractor.py)
        norm:        Wenn True, normalisiere mit exp-Decay durch Cycle-Peaks
                     (analog surrogates --norm). Default False.

    Returns dict mit:
        days_all, prices_all, dates_all       — komplette Reihe
        residuals_all                         — komplette rel. Residuen (für Plot 0)
        days_emb, signal, dates_emb           — ab start_idx, signal = was im
                                                Embedding verwendet wird
        signal_label                          — Beschreibung für Plots
        halving_days                          — int-Array der 4 Halving-Tage
        cycle_top_days, cycle_top_labels      — Cycle-Tops für Marker
        qr_log_fit                            — QuantReg-Fit (volle Reihe, log)
        norm                                  — bool, übernommene Option
        peak_days, peak_vals, exp_decay       — nur wenn norm=True
    """
    days_all, prices_all, dates_all = read_btc_data(filename)

    # Quantilregression auf log(price) vs log(day)  (attractor.py:105-112)
    log_days_all = np.log(days_all)
    log_btc_all  = np.log(prices_all)
    X_all        = sm.add_constant(log_days_all)
    qr           = QuantReg(log_btc_all, X_all).fit(q=percentile)
    log_fit_all  = qr.predict(X_all)
    residuals_all = prices_all / np.exp(log_fit_all)
    log_res_all   = np.log(residuals_all)

    # Halving-Tagesindizes
    halving_days = []
    for hd in _HALVINGS:
        diffs = np.array([abs((d - hd).days) for d in dates_all])
        halving_days.append(int(days_all[np.argmin(diffs)]))
    halving_days = np.array(halving_days)

    # Cycle-Top-Tagesindizes
    cycle_top_days = []
    cycle_top_labels = []
    for ct, lbl in _CYCLE_TOPS:
        diffs = np.array([abs((d - ct).days) for d in dates_all])
        idx = np.argmin(diffs)
        if diffs[idx] < 200:
            cycle_top_days.append(int(days_all[idx]))
            cycle_top_labels.append(lbl)
    cycle_top_days = np.array(cycle_top_days)

    # Cut ab start_idx
    mask      = days_all >= start_idx
    days_emb  = days_all[mask]
    log_res   = log_res_all[mask]
    dates_emb = dates_all[mask]

    out = {
        'days_all':         days_all,
        'prices_all':       prices_all,
        'dates_all':        dates_all,
        'residuals_all':    residuals_all,
        'log_res_all':      log_res_all,
        'qr_log_fit':       log_fit_all,
        'days_emb':         days_emb,
        'dates_emb':        dates_emb,
        'halving_days':     halving_days,
        'cycle_top_days':   cycle_top_days,
        'cycle_top_labels': cycle_top_labels,
        'norm':             bool(norm),
    }

    # Signal: log_res direkt ODER normalisiert
    if norm:
        exp_fit, pk_days, pk_vals = _peak_decay_fit(
            days_all, residuals_all, halving_days)
        # surrogates-Konvention: norm = (residual - 1) / (exp_decay - 1)
        denom_emb = exp_fit(days_emb) - 1.0
        signal = (residuals_all[mask] - 1.0) / denom_emb
        # Zusätzlich: die volle Reihe normiert für die Plot-0-Sanity
        denom_all = exp_fit(days_all) - 1.0
        signal_all = (residuals_all - 1.0) / denom_all

        out['signal'] = signal
        out['signal_all'] = signal_all
        out['signal_label'] = r'$(p/\mathrm{QR}-1)/(D_\mathrm{exp}-1)$'
        out['exp_decay'] = exp_fit
        out['peak_days'] = pk_days
        out['peak_vals'] = pk_vals
    else:
        out['signal'] = log_res
        out['signal_all'] = log_res_all
        out['signal_label'] = r'$\log\,p / \mathrm{QR}$ (log-residual)'

    # Lückenkontrolle
    gaps = np.diff(days_emb)
    if (gaps != 1).any():
        n_gaps = int((gaps != 1).sum())
        max_gap = int(gaps.max())
        print(f"  WARN: {n_gaps} Lücken in days_emb (max {max_gap}d). "
              f"start_idx={start_idx} zu früh?", file=sys.stderr)

    return out
