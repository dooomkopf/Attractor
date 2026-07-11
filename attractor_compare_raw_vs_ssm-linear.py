#!/usr/bin/env python3
"""Compare phase-space reconstructions of the SAME BTC power-law residuals.

2x2 grid: rows = embedding dimension (top M=6, bottom M=35),
          cols = raw delay coordinates  vs.  + PCA.

TIME_MODE selects the sampling clock:
  "log"    -> log10(age) resampling (Perrenod's clock; cycles compress -> inward spiral),
              trajectory coloured continuously by log10(day).
  "linear" -> calendar-time resampling (halving cycle repeats -> overlapping loops),
              trajectory coloured by HALVING CYCLE, halvings marked in cycle colours.

Residuals = QuantReg q=0.01 power-law bottom, identical to analyze_residuals/data.py.
Run it yourself:  ./attractor_compare_raw_vs_ssm.py
"""

import os
import sys
from datetime import datetime

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:
    plt.style.use(os.path.join(HERE, "hz.mplstyle"))
except Exception:
    pass
mpl.rcParams["font.sans-serif"] = ["Comfortaa", "DejaVu Sans", "Arial"]

# --- parameters -----------------------------------------------------------
FILENAME = os.path.join(HERE, "ziel.csv")
START_IDX = 1164
PERCENTILE = 0.01
N_GRID = 3000
P_M, P_TAU = 6, 50        # Perrenod raw embedding (dim, lag in grid steps)
S_M, S_TAU = 35, 40       # SSM embedding
TIME_MODE = "linear"

# --- data + power-law-bottom residuals ------------------------------------
from ssmlearn_res import read_btc_data  # noqa: E402

days_all, prices_all, dates_all = read_btc_data(FILENAME)
days_all = np.asarray(days_all, float)
prices_all = np.asarray(prices_all, float)

X = sm.add_constant(np.log(days_all))
qr = QuantReg(np.log(prices_all), X).fit(q=PERCENTILE)
log_res_all = np.log(prices_all / np.exp(qr.predict(X)))

mask = days_all >= START_IDX
days = days_all[mask]
res = log_res_all[mask]

# --- resampling onto N_GRID points (log10-age or linear calendar time) -----
if TIME_MODE == "log":
    _axis = np.log10(days)
    grid = np.linspace(_axis[0], _axis[-1], N_GRID)
    r = np.interp(grid, _axis, res)
    day_grid = 10.0 ** grid
else:  # linear
    grid = np.linspace(days[0], days[-1], N_GRID)
    r = np.interp(grid, days, res)
    day_grid = grid

# --- halving cycles: day index of each halving from the data --------------
_HALVINGS = [datetime(2012, 11, 28), datetime(2016, 7, 9),
             datetime(2020, 5, 11), datetime(2024, 4, 20)]
_hday = [float(days_all[np.argmin([abs((d - hd).days) for d in dates_all])])
         for hd in _HALVINGS]
CYCLES = [(_hday[0], _hday[1], "#0000FF", "'13 cycle"),
          (_hday[1], _hday[2], "#90EE90", "'17 cycle"),
          (_hday[2], _hday[3], "#FF69B4", "'21 cycle"),
          (_hday[3], 1e12,     "orange",  "'25 cycle")]

CMAP = "plasma"
norm = Normalize(vmin=float(np.log10(day_grid).min()),
                 vmax=float(np.log10(day_grid).max()))
_cmap = plt.get_cmap(CMAP)


def point_colors(day_slice):
    """log-mode: continuous plasma by log10(day); linear-mode: halving-cycle colour."""
    if TIME_MODE == "log":
        return _cmap(norm(np.log10(day_slice)))
    out = np.full(len(day_slice), "#8899AA", dtype=object)
    for lo, hi, c, _ in CYCLES:
        out[(day_slice >= lo) & (day_slice < hi)] = c
    return list(out)


def delay_embed(x, m, tau):
    W = (m - 1) * tau
    N = len(x)
    D = np.empty((N - W, m))
    for j in range(m):
        D[:, j] = x[W - j * tau: N - j * tau]
    return D, W


# --- top row data: M=6 ----------------------------------------------------
Dp, Wp = delay_embed(r, P_M, P_TAU)
p_xyz = Dp[:, :3]                                # raw
Dpc = Dp - Dp.mean(axis=0)
_, sp, Vtp = np.linalg.svd(Dpc, full_matrices=False)
pp_xyz = (Dpc @ Vtp.T)[:, :3]                    # + PCA
day_p = day_grid[Wp:]
col_p = point_colors(day_p)

# --- bottom row data: M=35 ------------------------------------------------
Ds, Ws = delay_embed(r, S_M, S_TAU)
r_xyz = Ds[:, :3]                                # raw
Dc = Ds - Ds.mean(axis=0)
_, s, Vt = np.linalg.svd(Dc, full_matrices=False)
s_xyz = (Dc @ Vt.T)[:, :3]                       # + PCA
day_s = day_grid[Ws:]
col_s = point_colors(day_s)

# --- plot: 2x2 ------------------------------------------------------------
fig = plt.figure(figsize=(15, 12))
ax_pr = fig.add_subplot(2, 2, 1, projection="3d")
ax_pp = fig.add_subplot(2, 2, 2, projection="3d")
ax_sr = fig.add_subplot(2, 2, 3, projection="3d")
ax_ss = fig.add_subplot(2, 2, 4, projection="3d")

for ax in (ax_pr, ax_pp, ax_sr, ax_ss):
    try:
        ax.set_box_aspect(None, zoom=0.85)
    except TypeError:
        pass
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#444444")


def _mark_halvings(ax, xyz, day_slice):
    """Mark each halving on the trajectory with a white-edged dot in its cycle colour."""
    if TIME_MODE != "linear":
        return
    for lo, hi, c, _ in CYCLES:
        idx = int(np.argmin(np.abs(day_slice - lo)))
        if abs(day_slice[idx] - lo) < 150:
            ax.scatter(xyz[idx, 0], xyz[idx, 1], xyz[idx, 2],
                       s=130, facecolor=c, edgecolor="white", linewidth=2.0, zorder=20)


def _panel(ax, xyz, col, day_slice, title, xlab, ylab, zlab):
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=0.4, color="#666666", alpha=0.4)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=col, s=2, alpha=0.7)
    ax.text2D(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top",
              fontsize=12, color="#CCCCCC")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)


_panel(ax_pr, p_xyz, col_p, day_p, rf"Raw Delay Coords  ($M={P_M}$, $\tau={P_TAU}$)",
       rf"$r(i)$", rf"$r(i+{P_TAU})$", rf"$r(i+{2 * P_TAU})$")
_panel(ax_pp, pp_xyz, col_p, day_p, rf"Delay Coords  ($M={P_M}$, $\tau={P_TAU}$) $+$ PCA",
       "PC1", "PC2", "PC3")
_panel(ax_sr, r_xyz, col_s, day_s, rf"Raw Delay Coords  ($M={S_M}$, $\tau={S_TAU}$)",
       rf"$r(i)$", rf"$r(i+{S_TAU})$", rf"$r(i+{2 * S_TAU})$")
_panel(ax_ss, s_xyz, col_s, day_s, rf"Delay Coords  ($M={S_M}$, $\tau={S_TAU}$) $+$ PCA",
       "PC1", "PC2", "PC3")

# --- colour key: colorbar (log) or halving-cycle legend (linear) ----------
if TIME_MODE == "log":
    sm_ = ScalarMappable(cmap=CMAP, norm=norm)
    sm_.set_array([])
    cax = fig.add_axes([0.93, 0.20, 0.011, 0.60])
    cb = fig.colorbar(sm_, cax=cax)
    cb.set_label(r"$\log_{10}(\mathrm{day})$", fontsize=plt.rcParams["font.size"] + 1)
else:
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=9,
                      markerfacecolor=c, markeredgecolor="none", label=name)
               for lo, hi, c, name in CYCLES]
    fig.legend(handles=handles, loc="center right", frameon=True,
               facecolor="#1A1A1A", edgecolor="#808080", labelcolor="#E0E0E0",
               fontsize=11)

_mode_word = "log10-time" if TIME_MODE == "log" else "linear-time"
with plt.rc_context({'text.usetex': False}):
    plt.suptitle(f"BTC-Residuals {_mode_word} sampled: Visualisation of different Time Delay Embeddings",
                 color='#CCCCCC', fontsize=13, y=0.98,
                 fontname='Comfortaa', fontweight='bold')

out = os.path.join(HERE, "compare_raw_vs_ssm-logtime.png" if TIME_MODE == "log"
                   else "compare_raw_vs_ssm-linear.png")
plt.subplots_adjust(top=0.90, bottom=0.03, left=0.02, right=0.91, wspace=0.05, hspace=0.14)
fig.savefig(out, dpi=300, facecolor='#0a0a0a')
print(f"saved {out}")
plt.show()
