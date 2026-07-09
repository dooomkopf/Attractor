"""Modular residual analysis package for BTC time series."""

from .data import build_residuals_and_embedding, read_btc_data
from .precheck import run_precheck
from .common import identify_modes, smooth_real_series, smooth_phase_series
from .constants import DAYS_PER_YEAR, START_IDX, HALVINGS

__all__ = [
    'build_residuals_and_embedding',
    'read_btc_data',
    'run_precheck',
    'identify_modes',
    'smooth_real_series',
    'smooth_phase_series',
    'DAYS_PER_YEAR',
    'START_IDX',
    'HALVINGS',
]