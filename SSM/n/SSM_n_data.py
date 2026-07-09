"""SSM/n/data.py — Daten laden + tägliches n(t) mit festem 180d-Fenster.

Pipeline-Schritte (analog SSM_res_data.load_data, aber Signal = n(t)):
    1) read_btc_data → days, prices, dates
    2) n_180(t) = (log p(t+90) - log p(t-90)) / (log(t+90) - log(t-90))
       für alle t mit gültigem Fenster (Rand: half=90d NaN)
    3) Cut ab start_idx, dann trailing-NaN entfernen
       (letzte 90 Tage sind per Konstruktion NaN)
"""

from datetime import datetime
import sys

import numpy as np


# BTC Halvings — Datum → Tagesindex aus ziel.csv (analog SSM_res_data).
_HALVINGS = [datetime(2012, 11, 28), datetime(2016,  7,  9),
             datetime(2020,  5, 11), datetime(2024,  4, 20)]

# Cycle-Tops (analog SSM_res_data).
_CYCLE_TOPS = [
    (datetime(2013, 12,  4), "T0"),
    (datetime(2017, 12, 16), "T1"),
    (datetime(2021, 11,  9), "T2"),
    (datetime(2025, 10,  7), "T3"),
]


def read_btc_data(filename):
    """Liest ziel.csv: <day> <price> <dd.mm.yyyy>. (1:1 wie SSM_res_data)."""
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


def compute_daily_n(days_all, prices_all, half=90):
    """Tägliches n(t) mit festem 2-Punkt-log-log-Fenster.

    n(t) = [log p(t+half) - log p(t-half)] / [log(t+half) - log(t-half)]

    Erste und letzte `half` Tage sind NaN. Default half=90 → W=180d.
    """
    days_all = np.asarray(days_all, dtype=float)
    prices_all = np.asarray(prices_all, dtype=float)
    n = np.full(len(days_all), np.nan, dtype=float)
    for i in range(half, len(days_all) - half):
        t1 = float(days_all[i - half])
        t2 = float(days_all[i + half])
        p1 = prices_all[i - half]
        p2 = prices_all[i + half]
        if p1 > 0 and p2 > 0 and t2 > t1 > 0:
            denom = np.log(t2) - np.log(t1)
            if denom > 0:
                n[i] = (np.log(p2) - np.log(p1)) / denom
    return n


def load_data(filename, start_idx, half=90):
    """Lädt ziel.csv, berechnet n_180(t), gibt dict im SSM_res_data-Format zurück.

    Args:
        filename:  absoluter Pfad zu ziel.csv
        start_idx: Tag-Index ab dem ausgewertet wird (default 1164)
        half:      halbes Fenster für n(t) in Tagen (default 90 → W=180)

    Returns dict mit Keys analog SSM_res_data.load_data:
        days_all, prices_all, dates_all       — komplette Reihe
        daily_n_all                           — komplette n(t)-Reihe (mit NaN-Rand)
        days_emb, signal, dates_emb           — ab start_idx, trailing NaN entfernt
        signal_label                          — r'$n_{180}(t)$ (daily exponent)'
        halving_days                          — int-Array der 4 Halving-Tage
        cycle_top_days, cycle_top_labels      — Cycle-Tops für Marker
        half                                  — übernommenes half-Fenster
    """
    days_all, prices_all, dates_all = read_btc_data(filename)
    daily_n_all = compute_daily_n(days_all, prices_all, half=half)

    halving_days = []
    for hd in _HALVINGS:
        diffs = np.array([abs((d - hd).days) for d in dates_all])
        halving_days.append(int(days_all[np.argmin(diffs)]))
    halving_days = np.array(halving_days)

    cycle_top_days = []
    cycle_top_labels = []
    for ct, lbl in _CYCLE_TOPS:
        diffs = np.array([abs((d - ct).days) for d in dates_all])
        idx = np.argmin(diffs)
        if diffs[idx] < 200:
            cycle_top_days.append(int(days_all[idx]))
            cycle_top_labels.append(lbl)
    cycle_top_days = np.array(cycle_top_days)

    mask      = days_all >= start_idx
    days_emb  = days_all[mask].astype(float)
    n_emb     = daily_n_all[mask]
    dates_emb = dates_all[mask]

    finite = np.isfinite(n_emb)
    if not finite.any():
        raise ValueError(
            f'n(t) komplett NaN ab start_idx={start_idx} (half={half}). '
            f'Datenlänge zu kurz?'
        )
    last_valid = int(np.where(finite)[0].max()) + 1
    first_valid = int(np.where(finite)[0].min())
    if first_valid != 0:
        print(f"  INFO: erste {first_valid} Tage ab start_idx={start_idx} sind NaN, "
              f"start verschoben auf day={int(days_emb[first_valid])}",
              file=sys.stderr)
    days_emb  = days_emb[first_valid:last_valid]
    signal    = n_emb[first_valid:last_valid]
    dates_emb = dates_emb[first_valid:last_valid]

    if not np.all(np.isfinite(signal)):
        n_gaps = int((~np.isfinite(signal)).sum())
        raise ValueError(
            f'n(t) hat {n_gaps} NaN-Lücken innerhalb [start_idx, end-half]. '
            f'Sollte nicht passieren bei lückenlosen ziel.csv-Daten.'
        )

    gaps = np.diff(days_emb)
    if (gaps != 1).any():
        n_gaps = int((gaps != 1).sum())
        max_gap = int(gaps.max())
        print(f"  WARN: {n_gaps} Lücken in days_emb (max {max_gap}d).",
              file=sys.stderr)

    return {
        'days_all':         days_all,
        'prices_all':       prices_all,
        'dates_all':        dates_all,
        'daily_n_all':      daily_n_all,
        'days_emb':         days_emb,
        'dates_emb':        dates_emb,
        'signal':           signal,
        'signal_label':     r'$n_{180}(t)$ (daily exponent)',
        'halving_days':     halving_days,
        'cycle_top_days':   cycle_top_days,
        'cycle_top_labels': cycle_top_labels,
        'half':             int(half),
    }
