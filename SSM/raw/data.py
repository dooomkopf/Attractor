"""SSM/raw/data.py — Daten laden und Signal vorbereiten (raw: log p oder p).

Bei einer Kopie nach SSM/res/ wird hier QuantReg-Detrending eingebaut,
bei SSM/n/ die n(t)-Berechnung mit zentriertem Fenster.
"""

from datetime import datetime
import sys

import numpy as np


# BTC Halvings — Datum → wird auf den nächsten Tagesindex aus ziel.csv gemappt.
_HALVINGS = [datetime(2012, 11, 28), datetime(2016,  7,  9),
             datetime(2020,  5, 11), datetime(2024,  4, 20)]


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


def load_data(filename, start_idx, mode='log'):
    """Lädt ziel.csv, schneidet ab start_idx, gibt Signal in Modus log/lin zurück.

    Args:
        filename:  absoluter Pfad zu ziel.csv
        start_idx: Tag-Index ab dem ausgewertet wird (default 1164 = täglich kont.)
        mode:      'log' (default) oder 'lin'

    Returns:
        days_emb:     int-Array, Tage ab start_idx
        signal:       float-Array, log(price) oder price
        dates_emb:    datetime-Array
        halving_days: int-Array der 4 Halving-Tage (Original day-index)
    """
    days_all, prices_all, dates_all = read_btc_data(filename)

    if mode == 'log':
        sig_all = np.log(prices_all)
    elif mode == 'lin':
        sig_all = prices_all.astype(float)
    else:
        raise ValueError(f"mode must be 'log' or 'lin', got {mode!r}")

    # Halving-Tagesindizes (originale day-Werte, nicht Array-Indizes)
    halving_days = []
    for hd in _HALVINGS:
        diffs = np.array([abs((d - hd).days) for d in dates_all])
        halving_days.append(int(days_all[np.argmin(diffs)]))
    halving_days = np.array(halving_days)

    # Cut ab start_idx
    mask     = days_all >= start_idx
    days_emb = days_all[mask]
    sig_emb  = sig_all[mask]
    dates_emb= dates_all[mask]

    # Lückenkontrolle (ab 1164 sollte täglich sein)
    gaps = np.diff(days_emb)
    if (gaps != 1).any():
        n_gaps = int((gaps != 1).sum())
        max_gap = int(gaps.max())
        print(f"  WARN: {n_gaps} Lücken in days_emb (max {max_gap}d). "
              f"start_idx={start_idx} zu früh?", file=sys.stderr)

    return days_emb, sig_emb, dates_emb, halving_days
