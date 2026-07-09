#!/usr/bin/env python3
"""
attractor_2scroll_slider.py — Wang 2-Scroll mit interaktiven Schiebereglern.

Schieberegler: a, d, IC_x, IC_y, IC_z
Live-Update: H1-Fläche + Schnittkurve (ohne H2 — zu langsam für Echtzeit)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Startwerte ─────────────────────────────────────────────────────────────────
A0, D0 = 3.5, 0.06
B, C    = 9.0, 5.0
IC0     = np.array([4.84, 0.25, -5.44])

EQ_COLORS = ['white', 'cyan', 'magenta', 'lime', 'orange']

try:
    plt.style.use('hz.mplstyle')
except Exception:
    plt.rcParams.update({'figure.facecolor': 'black', 'axes.facecolor': '#0A0A0A',
                         'text.color': '#CCCCCC'})


# ── Funktionen ─────────────────────────────────────────────────────────────────
def h1(x, y, z, d):
    return z**2 - (y + d)**2


def h2(x, y, z, a, d):
    arg = np.abs(z + y + d)
    arg = np.maximum(arg, 1e-12)
    return 0.25*(x**2 + y**2) + 0.5*a*z - 0.5*a*d*np.log(arg)


def equilibria(a, b, c, d):
    pts = [('S1', np.zeros(3))]
    disc_z = a**2 + 4*a*b
    if disc_z < 0:
        return pts
    for sz in [+1, -1]:
        z = (-a + sz*np.sqrt(disc_z)) / 2
        disc_x = d**2 + 4*(z/b)*c*z
        if disc_x < 0:
            continue
        for sx in [+1, -1]:
            x = (-d + sx*np.sqrt(disc_x)) / (2*z/b)
            y = x*z / b
            pts.append((f'S{len(pts)+1}', np.array([x, y, z])))
    return pts


def build_loops(k1, k2, a, d, y_range=(-15, 15), n=4000):
    yg = np.linspace(*y_range, n)
    loops = []
    for sigma in (1.0, -1.0):
        rad = k1 + (yg + d)**2
        z_br = np.full_like(yg, np.nan)
        mask = rad >= 0
        z_br[mask] = sigma * np.sqrt(rad[mask])

        x_sq = np.full_like(yg, np.nan)
        safe = mask & (np.abs(z_br + yg + d) > 1e-10)
        x_sq[safe] = (4*k2 - yg[safe]**2
                      - 2*a*z_br[safe]
                      + 2*a*d*np.log(np.abs(z_br[safe] + yg[safe] + d)))

        valid = np.isfinite(x_sq) & (x_sq >= 0)
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        gaps = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[gaps + 1]]
        ends   = np.r_[idx[gaps], idx[-1]]
        for s, e in zip(starts, ends):
            y_s = yg[s:e+1]; z_s = z_br[s:e+1]
            xr  = np.sqrt(np.maximum(x_sq[s:e+1], 0))
            loops.append(np.column_stack([
                np.concatenate([ xr, -xr[::-1]]),
                np.concatenate([ y_s,  y_s[::-1]]),
                np.concatenate([ z_s,  z_s[::-1]]),
            ]))
    return loops


def build_h1(k1, d, xl, yl, n=80):
    xg = np.linspace(*xl, n); yg = np.linspace(*yl, n)
    XF, YF = np.meshgrid(xg, yg)
    rad = k1 + (YF + d)**2
    zp  = np.where(rad >= 0, np.sqrt(np.maximum(rad, 0)), np.nan)
    return XF, YF, zp, -zp


# ── Figure aufbauen ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 9))
fig.patch.set_facecolor('black')

ax = fig.add_axes([0.05, 0.30, 0.90, 0.66], projection='3d')
ax.set_facecolor('#0A0A0A')

# Schieberegler-Achsen
sl_ax = {}
params = [
    ('a',    0.1, 5.0,  A0,   'a (Kreis ← klein)'),
    ('d',   -0.5, 0.5,  D0,   'd'),
    ('ICx', -8.0, 8.0,  IC0[0], 'IC x'),
    ('ICy', -8.0, 8.0,  IC0[1], 'IC y'),
    ('ICz', -8.0, 8.0,  IC0[2], 'IC z'),
]
sliders = {}
for i, (name, vmin, vmax, v0, label) in enumerate(params):
    sax = fig.add_axes([0.12, 0.22 - i*0.04, 0.75, 0.025],
                       facecolor='#1A1A1A')
    sl = widgets.Slider(sax, label, vmin, vmax, valinit=v0,
                        color='#4488FF', track_color='#222222')
    sl.label.set_color('#CCCCCC')
    sl.valtext.set_color('#AAAAAA')
    sliders[name] = sl

# Plot-Objekte (werden in update() neu gesetzt)
surf_h1p = [None]; surf_h1n = [None]
orbit_lines = []
eq_scatters = []; eq_texts = []
info_text = [None]


def update(_=None):
    a  = sliders['a'].val
    d  = sliders['d'].val
    ic = np.array([sliders['ICx'].val, sliders['ICy'].val, sliders['ICz'].val])

    k1 = float(h1(*ic, d))
    k2 = float(h2(*ic, a, d))

    loops = build_loops(k1, k2, a, d)

    # Achsengrenzen aus Loops bestimmen
    if loops:
        all_pts = np.vstack(loops)
        xl = (all_pts[:,0].min()-2, all_pts[:,0].max()+2)
        yl = (all_pts[:,1].min()-2, all_pts[:,1].max()+2)
        zl = (all_pts[:,2].min()-2, all_pts[:,2].max()+2)
    else:
        xl = yl = zl = (-10, 10)

    ax.cla()
    ax.set_facecolor('#0A0A0A')

    # H1-Fläche
    XF, YF, zp, zn = build_h1(k1, d, xl, yl)
    ax.plot_surface(XF, YF, zp, color='#00DDDD', alpha=0.18,
                    linewidth=0, antialiased=True, shade=False)
    ax.plot_surface(XF, YF, zn, color='#00DDDD', alpha=0.18,
                    linewidth=0, antialiased=True, shade=False)

    # Schnittkurven
    colors = ['#FF8800', '#FF4444', '#FFFF00', '#FF88FF']
    for i, loop in enumerate(loops):
        ax.plot(loop[:,0], loop[:,1], loop[:,2],
                color=colors[i % len(colors)], linewidth=2.2, alpha=0.95)

    # Gleichgewichte
    eq_pts = equilibria(a, B, C, d)
    for i, (lbl, pt) in enumerate(eq_pts):
        col = EQ_COLORS[i % len(EQ_COLORS)]
        ax.scatter(*pt, color=col, s=60, zorder=10, depthshade=False)
        ax.text(pt[0], pt[1], pt[2]+0.3, f' {lbl}', color=col, fontsize=8)

    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor('#333333')
    ax.tick_params(colors='#888888', labelsize=7)
    ax.set_xlabel('x', color='#AAAAAA'); ax.set_ylabel('y', color='#AAAAAA')
    ax.set_zlabel('z', color='#AAAAAA')
    ax.set_title(f'Wang 2-Scroll  a={a:.3f}  d={d:.3f}  k1={k1:.3f}  k2={k2:.3f}  '
                 f'Loops={len(loops)}',
                 color='#CCCCCC', fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.canvas.draw_idle()


for sl in sliders.values():
    sl.on_changed(update)

update()
plt.show()
