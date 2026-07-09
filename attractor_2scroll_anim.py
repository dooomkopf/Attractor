#!/usr/bin/env python3
"""
attractor_2scroll_anim.py — Wang 2-Scroll animiert.

Trajektorie läuft durch den Attraktor; Gleichgewichtspunkte leuchten auf
wenn die Trajektorie in ihrer Nähe ist.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import solve_ivp

# ── Parameter ─────────────────────────────────────────────────────────────────
A, B, C, D = 3.5, 9.0, 5.0, 0.06

TRAIL       = 4000   # Länge der sichtbaren Trajektorie (Punkte)
STEP        = 3      # Punkte pro Frame
GLOW_DIST   = 4.0    # Schwelle: "nahe" Gleichgewicht (Einheiten im Phasenraum)
FPS         = 30

EQ_COLORS   = ['white', 'cyan', 'magenta', 'lime', 'orange']

try:
    plt.style.use('hz.mplstyle')
except Exception:
    plt.rcParams.update({
        'figure.facecolor': 'black', 'axes.facecolor': '#0A0A0A',
        'text.color': '#CCCCCC', 'font.size': 10,
    })


def wang_rhs(t, s):
    x, y, z = s
    return [A*(x-y) - y*z, -B*y + x*z, -C*z + D*x + x*y]


def equilibria():
    pts = [('S1', np.zeros(3))]
    disc_z = A**2 + 4*A*B
    for sz in [+1, -1]:
        z = (-A + sz*np.sqrt(disc_z)) / 2
        a2, b2, c2 = z/B, D, -C*z
        disc_x = b2**2 - 4*a2*c2
        if disc_x < 0:
            continue
        for sx in [+1, -1]:
            x = (-b2 + sx*np.sqrt(disc_x)) / (2*a2)
            y = x*z / B
            pts.append((f'S{len(pts) + 1}', np.array([x, y, z])))
    return pts


def main():
    # ── Trajektorie vorberechnen ───────────────────────────────────────────────
    print("Integriere Wang-ODE...")
    t_end  = 120
    n_pts  = 60000
    sol    = solve_ivp(wang_rhs, (0, t_end), [0.1, 0.1, 0.1],
                       method='RK45', t_eval=np.linspace(0, t_end, n_pts),
                       rtol=1e-9, atol=1e-11)
    traj   = sol.y.T          # (n_pts, 3)
    skip   = n_pts // 6
    traj   = traj[skip:]      # Einschwingzeit weg
    N      = len(traj)

    eq_pts = equilibria()
    print("Gleichgewichte:")
    for lbl, pt in eq_pts:
        print(f"  {lbl} = ({pt[0]:+.3f}, {pt[1]:+.3f}, {pt[2]:+.3f})")

    # Distanzen vorberechnen: (N, 5)
    dists = np.array([np.linalg.norm(traj - pt, axis=1) for _, pt in eq_pts]).T

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 9))
    fig.patch.set_facecolor('black')
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.90], projection='3d')
    ax.set_facecolor('#050505')

    xlim = (traj[:,0].min()*1.1, traj[:,0].max()*1.1)
    ylim = (traj[:,1].min()*1.1, traj[:,1].max()*1.1)
    zlim = (traj[:,2].min()*1.1, traj[:,2].max()*1.1)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)

    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('#222222')
    ax.tick_params(colors='#444444', labelsize=7)
    ax.set_xlabel('x', color='#555555', labelpad=5)
    ax.set_ylabel('y', color='#555555', labelpad=5)
    ax.set_zlabel('z', color='#555555', labelpad=5)

    title = ax.set_title('Wang 2-Scroll', color='#AAAAAA', fontsize=10)

    # Gleichgewichte: Basis-Scatter (immer sichtbar, klein)
    eq_base    = []
    eq_glow    = []
    eq_labels  = []
    for i, (lbl, pt) in enumerate(eq_pts):
        col = EQ_COLORS[i % len(EQ_COLORS)]
        sc_b = ax.scatter(*pt, color=col, s=30, alpha=0.4,
                          edgecolors='none', zorder=5)
        sc_g = ax.scatter(*pt, color=col, s=0,  alpha=0.0,
                          edgecolors='white', linewidths=0.5, zorder=6)
        txt  = ax.text(pt[0], pt[1], pt[2], f'  {lbl}',
                       color=col, fontsize=8, alpha=0.4)
        eq_base.append(sc_b)
        eq_glow.append(sc_g)
        eq_labels.append(txt)

    # Trajektorie: Line3DCollection (wird pro Frame neu gesetzt)
    line, = ax.plot([], [], [], color='#4488FF', linewidth=0.6, alpha=0.6)
    dot   = ax.scatter([], [], [], color='yellow', s=40, zorder=10)

    # ── Animation ─────────────────────────────────────────────────────────────
    def update(frame):
        i = frame * STEP
        if i >= N:
            i = N - 1

        i0 = max(0, i - TRAIL)
        seg = traj[i0:i+1]

        # Trajektorie
        line.set_data(seg[:, 0], seg[:, 1])
        line.set_3d_properties(seg[:, 2])

        # Aktueller Punkt
        dot._offsets3d = ([traj[i,0]], [traj[i,1]], [traj[i,2]])

        # Gleichgewichte leuchten auf
        for k, (sc_b, sc_g, txt) in enumerate(
                zip(eq_base, eq_glow, eq_labels)):
            d = dists[i, k]
            if d < GLOW_DIST:
                brightness = max(0, 1 - d/GLOW_DIST)
                sc_g._sizes = np.array([300 * brightness**2])
                sc_g.set_alpha(0.8 * brightness)
                txt.set_alpha(1.0)
            else:
                sc_g._sizes = np.array([0])
                sc_g.set_alpha(0.0)
                txt.set_alpha(0.25)

        title.set_text(f'Wang 2-Scroll  —  t = {i/N*t_end*(1-1/6):.1f}')
        return [line, dot] + eq_glow + eq_labels

    n_frames = (N // STEP)
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=1000//FPS, blit=False)

    plt.show()


if __name__ == '__main__':
    main()
