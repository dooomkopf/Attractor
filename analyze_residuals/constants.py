"""Constants and default parameters for residual analysis."""

from datetime import datetime
import os

ATTRACTOR_DIR = '/home/hz/Data/Attractor'
DEFAULT_CYCLES_JSON = '/home/hz/Data/gold/cycles.json'

# Time constants
DAYS_PER_YEAR = 365.25

# Data range
START_IDX = 1164

# Bitcoin halvings
HALVINGS = [
    datetime(2012, 11, 28),
    datetime(2016, 7, 9),
    datetime(2020, 5, 11),
    datetime(2024, 4, 20)
]

# Embedding defaults
DEFAULT_M = 35
DEFAULT_YEARS = 3.77
DEFAULT_SMOOTH_DAYS = 180

# Data file
DEFAULT_FILENAME = os.path.join(ATTRACTOR_DIR, 'ziel.csv')

# Quantile regression
PERCENTILE = 0.01
