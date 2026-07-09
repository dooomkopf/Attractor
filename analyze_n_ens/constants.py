"""Constants for ensemble-n SSM analysis (analog to analyze_residuals)."""

import os

ATTRACTOR_DIR = '/home/hz/Data/Attractor'

# Window sizes for the n(t) ensemble: matches attractor_n_ens.py convention
WINDOW_SIZES = list(range(90, 181, 10))   # [90, 100, ..., 180]

# Embedding defaults -- identical to analyze_residuals so the analyses are comparable
DEFAULT_M = 35
DEFAULT_YEARS = 3.77
DAYS_PER_YEAR = 365.25

# Start index identical to residuals to keep day axis aligned
START_IDX = 1164

# Data file
DEFAULT_FILENAME = os.path.join(ATTRACTOR_DIR, 'ziel.csv')
