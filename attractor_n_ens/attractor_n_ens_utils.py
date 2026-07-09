"""attractor_n_ens/utils.py — I/O und Segment-Hilfsfunktionen"""

import numpy as np
from datetime import datetime


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


def make_segments(x, y):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)


def make_segments3d(x, y, z):
    pts = np.array([x, y, z]).T.reshape(-1, 1, 3)
    return np.concatenate([pts[:-1], pts[1:]], axis=1)
