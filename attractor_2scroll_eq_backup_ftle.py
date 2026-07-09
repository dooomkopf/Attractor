#!/usr/bin/env python3
"""
attractor_2scroll_eq.py — Wang 2-Scroll mit Nambu-Flächen + zyklischer Trajektorie.

Nambu-Hamiltonians (arXiv:2511.13332v1):
  H1 = z² - (y+d)²
  H2 = 1/4(x²+y²) + 1/2*a*z - 1/2*a*d*log|z+y+d|

Referenz-Geometrie (H1/H2/Loops) wird einmal aus dem Start-IC gebaut.
Die animierte Trajektorie bekommt pro vollständigem Zyklus einen neuen IC,
gesampelt aus der post-transienten Attraktorbahn selbst. Eine schwache
Ghost-Spur akkumuliert sich über die Zyklen.
"""

import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# ── Parameter ────────────────────────────────────────────────────────────────
A = 3.5
B = 6.4 # 9
C = 5.0
D = 0.6 # 0.06

IC = np.array([4.84, 0.25, -5.44], dtype=float)

TRAIL     = 5000   # Länge des sichtbaren Trajektorie-Schwanzes
STEP      = 150    # Punkte pro Frame (höher = schneller)
GLOW_DIST = 3.5    # Glüh-Schwelle (Einheiten)
FPS       = 30
TRANSIENT_FRAC = 1 / 8

GHOST_STRIDE     = 2      # Nur jede n-te Probe wird ins Ghost geschrieben
GHOST_SIZE       = 1.5
GHOST_ALPHA      = 0.25
GHOST_MAX_POINTS = 200000
N_TRAJS          = 5      # Anzahl Trajektorien
T_EACH           = 40      # Integrationsdauer pro Trajektorie [s]
N_EACH           = 20000   # Punkte pro Trajektorie
C_MIN            = 5.0     # c-Variationsbereich (nur Dissipation, nicht Scroll-Zahl)
C_MAX            = 5.1
B_MIN            = 6.4     # b-Variationsbereich
B_MAX            = 7.0
RNG_SEED         = 7

EQ_COLORS = ['white', 'cyan', 'magenta', 'lime', 'orange']

try:
    plt.style.use('hz.mplstyle')
except Exception:
    plt.rcParams.update({'figure.facecolor': 'black', 'axes.facecolor': '#0A0A0A',
                         'text.color': '#CCCCCC'})


# ── Hamiltonians ─────────────────────────────────────────────────────────────
def h1(x, y, z):
    return z**2 - (y + D)**2


def h2(x, y, z):
    arg = np.abs(z + y + D)
    arg = np.maximum(arg, 1e-12)
    return 0.25 * (x**2 + y**2) + 0.5 * A * z - 0.5 * A * D * np.log(arg)


# ── ODE ──────────────────────────────────────────────────────────────────────
def wang_rhs(t, s):
    x, y, z = s
    return [A*(x - y) - y*z, -B*y + x*z, -C*z + D*x + x*y]


def make_wang_rhs(b, c):
    """Wang-ODE mit variiertem b und c."""
    def rhs(t, s):
        x, y, z = s
        return [A*(x - y) - y*z, -b*y + x*z, -c*z + D*x + x*y]
    return rhs


def circular_distance(i, j, n):
    d = abs(i - j)
    return min(d, n - d)


def choose_cycle_start(n, rng, prev_start=None, min_sep=0):
    if n <= 1:
        return 0
    if prev_start is None or min_sep <= 0 or n <= 2 * min_sep:
        return int(rng.integers(0, n))

    for _ in range(32):
        cand = int(rng.integers(0, n))
        if circular_distance(cand, prev_start, n) >= min_sep:
            return cand
    return int((prev_start + n // 2) % n)


def append_ghost_points(ghost_xyz, traj):
    chunk = np.asarray(traj[::GHOST_STRIDE], dtype=float)
    if chunk.size == 0:
        return ghost_xyz
    if ghost_xyz.size == 0:
        merged = chunk
    else:
        merged = np.vstack((ghost_xyz, chunk))
    if len(merged) > GHOST_MAX_POINTS:
        merged = merged[-GHOST_MAX_POINTS:]
    return merged


# ── Gleichgewichte ───────────────────────────────────────────────────────────
def equilibria():
    pts = [('S1', np.zeros(3))]
    disc_z = A**2 + 4*A*B
    for sz in [+1, -1]:
        z = (-A + sz * np.sqrt(disc_z)) / 2
        a2 = z / B; b2 = D; c2 = -C * z
        disc_x = b2**2 - 4*a2*c2
        if disc_x < 0:
            continue
        for sx in [+1, -1]:
            x = (-b2 + sx * np.sqrt(disc_x)) / (2 * a2)
            y = x * z / B
            pts.append((f'S{len(pts)+1}', np.array([x, y, z])))
    return pts


# ── H1-Fläche ────────────────────────────────────────────────────────────────
def build_h1_sheet(k1, xl, yl, n=80):
    xg = np.linspace(*xl, n); yg = np.linspace(*yl, n)
    XF, YF = np.meshgrid(xg, yg)
    rad = k1 + (YF + D)**2
    zp  = np.where(rad >= 0, np.sqrt(np.maximum(rad, 0)), np.nan)
    return XF, YF, zp, -zp


# ── H2-Fläche (Brentq) ───────────────────────────────────────────────────────
def build_h2_branches(k2, xl, yl, zl, n_xy=60, n_z=300):
    xg = np.linspace(*xl, n_xy); yg = np.linspace(*yl, n_xy)
    z_scan = np.linspace(*zl, n_z)
    XG, YG = np.meshgrid(xg, yg)
    branches = [np.full_like(XG, np.nan) for _ in range(3)]
    for j, yv in enumerate(yg):
        sing = -yv - D
        for i, xv in enumerate(xg):
            vals = h2(xv, yv, z_scan) - k2
            vals = np.asarray(vals, dtype=float)
            vals[np.abs(z_scan - sing) < 1e-6] = np.nan
            roots = []
            for k in range(n_z - 1):
                f0, f1 = vals[k], vals[k+1]
                if not (np.isfinite(f0) and np.isfinite(f1)):
                    continue
                if f0 == 0.0:
                    roots.append(float(z_scan[k])); continue
                if f0 * f1 > 0:
                    continue
                try:
                    r = brentq(lambda z: float(h2(xv, yv, z) - k2),
                               float(z_scan[k]), float(z_scan[k+1]), maxiter=80)
                except ValueError:
                    continue
                if not roots or abs(r - roots[-1]) > 1e-3:
                    roots.append(r)
                if len(roots) == 3:
                    break
            for idx, r in enumerate(sorted(roots)[:3]):
                branches[idx][j, i] = r
    return XG, YG, branches


# ── Schnittkurven H1∩H2 ──────────────────────────────────────────────────────
def build_loops(k1, k2, yl, n=5000):
    yg = np.linspace(*yl, n)
    loops = []
    for sigma in (1.0, -1.0):
        rad = k1 + (yg + D)**2
        zb  = np.full_like(yg, np.nan)
        mask = rad >= 0
        zb[mask] = sigma * np.sqrt(rad[mask])
        xsq = np.full_like(yg, np.nan)
        safe = mask & (np.abs(zb + yg + D) > 1e-10)
        xsq[safe] = (4*k2 - yg[safe]**2 - 2*A*zb[safe]
                     + 2*A*D*np.log(np.abs(zb[safe] + yg[safe] + D)))
        valid = np.isfinite(xsq) & (xsq >= 0)
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        gaps   = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[gaps+1]]
        ends   = np.r_[idx[gaps], idx[-1]]
        for s, e in zip(starts, ends):
            ys = yg[s:e+1]; zs = zb[s:e+1]
            xr = np.sqrt(np.maximum(xsq[s:e+1], 0))
            loops.append(np.column_stack([
                np.concatenate([ xr, -xr[::-1]]),
                np.concatenate([ ys,  ys[::-1]]),
                np.concatenate([ zs,  zs[::-1]]),
            ]))
    return loops


# ── Hauptprogramm ────────────────────────────────────────────────────────────
def main():
    # ── Referenz-Integration (Pool für ICs + Achsengrenzen) ──────────────────
    print("Integriere Referenz-Trajektorie...")
    t_ref = 150; n_ref = 75000
    sol = solve_ivp(wang_rhs, (0, t_ref), IC,
                    method='RK45', t_eval=np.linspace(0, t_ref, n_ref),
                    rtol=1e-9, atol=1e-11)
    traj_pool = sol.y.T
    skip = int(n_ref * TRANSIENT_FRAC)
    traj_pool = traj_pool[skip:]
    N = len(traj_pool)
    print(f"Pool: {N} Punkte")

    eq_pts = equilibria()
    print("Gleichgewichte:")
    for lbl, pt in eq_pts:
        print(f"  {lbl} = ({pt[0]:+.3f}, {pt[1]:+.3f}, {pt[2]:+.3f})")

    # Achsengrenzen aus Pool
    pad = 2.0
    xl = (traj_pool[:,0].min() - pad, traj_pool[:,0].max() + pad)
    yl = (traj_pool[:,1].min() - pad, traj_pool[:,1].max() + pad)
    zl = (traj_pool[:,2].min() - pad, traj_pool[:,2].max() + pad)

    # ── H1/H2-Flächen: FIX aus Referenz-IC (Topologie = Parameter a,d) ──────
    k1_ref = float(h1(*IC))
    k2_ref = float(h2(*IC))
    print(f"Referenz-Geometrie: k1={k1_ref:.4f}  k2={k2_ref:.4f}")
    print("Baue H1-Fläche...")
    xh1, yh1, zh1p, zh1n = build_h1_sheet(k1_ref, xl, yl)
    print("Baue H2-Fläche (~30s)...")
    xh2, yh2, h2_br = build_h2_branches(k2_ref, xl, yl, zl)

    # ── 100 Trajektorien: selber IC, c variiert (b,c ändern Scroll-Zahl nicht) ─
    # Loops: statisch aus k1_ref, k2_ref (H1,H2 hängen nur von a,d ab → fix)
    print("Baue Referenz-Loops (H1∩H2, fix für alle Zyklen)...")
    loops_ref = build_loops(k1_ref, k2_ref, yl)
    print(f"  {len(loops_ref)} Loop(s)")

    rng = np.random.default_rng(RNG_SEED)
    b_vals = rng.uniform(B_MIN, B_MAX, N_TRAJS)
    c_vals = rng.uniform(C_MIN, C_MAX, N_TRAJS)
    print(f"Integriere {N_TRAJS} Trajektorien: b ∈ [{B_MIN},{B_MAX}], c ∈ [{C_MIN},{C_MAX}] (a={A}, d={D})...")
    all_trajs = []
    all_dists = []
    for k, (b_k, c_k) in enumerate(zip(b_vals, c_vals)):
        rhs_k = make_wang_rhs(b_k, c_k)
        sol_k = solve_ivp(rhs_k, (0, T_EACH), IC,
                          method='RK45',
                          t_eval=np.linspace(0, T_EACH, N_EACH),
                          rtol=1e-7, atol=1e-9)
        tr = sol_k.y.T
        all_trajs.append(tr)
        all_dists.append(np.array([np.linalg.norm(tr - pt, axis=1)
                                   for _, pt in eq_pts]).T)
        print(f"  {k+1}/{N_TRAJS}  b={b_k:.3f}  c={c_k:.3f}")
    print("Alle Trajektorien fertig.")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(11, 9), facecolor='black')
    ax  = fig.add_axes([0.02, 0.04, 0.96, 0.92], projection='3d')
    ax.set_facecolor('#050505')

    # H1/H2-Flächen (fix, zeigen Topologie)
    ax.plot_surface(xh1, yh1,  zh1p, color='#4FD9E8', alpha=0.15,
                    linewidth=0, antialiased=True, shade=False)
    ax.plot_surface(xh1, yh1,  zh1n, color='#4FD9E8', alpha=0.15,
                    linewidth=0, antialiased=True, shade=False)
    for br in h2_br:
        if np.isfinite(br).any():
            ax.plot_surface(xh2, yh2, br, color='#C9993A', alpha=0.18,
                            linewidth=0, antialiased=True, shade=False,
                            rcount=60, ccount=60)

    # Referenz-Loops statisch (selbe für alle 100 Zyklen)
    loop_colors = ['#FF8800', '#FF4444', '#FFFF00', '#FF88FF']
    for i, lp in enumerate(loops_ref):
        ax.plot(lp[:,0], lp[:,1], lp[:,2],
                color=loop_colors[i % len(loop_colors)], linewidth=2.5, alpha=0.95)

    # Gleichgewichte
    eq_base  = []
    eq_glow  = []
    eq_texts = []
    for i, (lbl, pt) in enumerate(eq_pts):
        col = EQ_COLORS[i % len(EQ_COLORS)]
        sb = ax.scatter(*pt, color=col, s=25, alpha=0.25,
                        edgecolors='none', zorder=5, depthshade=False)
        sg = ax.scatter(*pt, color=col, s=0,  alpha=0.0,
                        edgecolors='white', linewidths=0.8, zorder=6, depthshade=False)
        tx = ax.text(pt[0], pt[1], pt[2]+0.4, f'  {lbl}',
                     color=col, fontsize=10, alpha=0.85)
        eq_base.append(sb); eq_glow.append(sg); eq_texts.append(tx)

    # Animierte Objekte
    ghost = ax.scatter([], [], [], color='#999999', s=GHOST_SIZE, alpha=GHOST_ALPHA,
                       edgecolors='none', zorder=2, depthshade=False)
    line, = ax.plot([], [], [], color='#4488FF', linewidth=0.7, alpha=0.65)
    dot   = ax.scatter([], [], [], color='yellow', s=50, zorder=10, depthshade=False)

    # Achsen
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_zlim(*zl)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor('#222222')
    ax.tick_params(colors='#444444', labelsize=7)
    ax.set_xlabel('x', color='#666666', labelpad=5)
    ax.set_ylabel('y', color='#666666', labelpad=5)
    ax.set_zlabel('z', color='#666666', labelpad=5)
    title_obj = ax.set_title('', color='#AAAAAA', fontsize=9)

    # ── Animation ────────────────────────────────────────────────────────────
    n_frames = (N_EACH - 1) // STEP + 1
    state = {
        'cycle':          0,
        'traj_idx':       0,
        'frame_in_cycle': 0,
        'traj':           all_trajs[0],
        'dists':          all_dists[0],
        'ghost_xyz':      np.empty((0, 3), dtype=float),
        'ghost_last_i':   0,
    }

    def apply_ghost():
        xyz = state['ghost_xyz']
        if xyz.size == 0:
            ghost._offsets3d = ([], [], [])
        else:
            ghost._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
            ghost.set_sizes(np.full(len(xyz), GHOST_SIZE))

    def set_cycle(idx, archive_previous):
        if archive_previous:
            remaining = state['traj'][state['ghost_last_i']:]
            if len(remaining) > 0:
                state['ghost_xyz'] = append_ghost_points(state['ghost_xyz'], remaining)
            apply_ghost()
            state['cycle'] += 1
        else:
            state['cycle'] = 1
        state['traj_idx']       = idx
        state['frame_in_cycle'] = 0
        state['ghost_last_i']   = 0
        state['traj']           = all_trajs[idx]
        state['dists']          = all_dists[idx]
        print(f"Zyklus {state['cycle']:03d}: c={c_vals[idx]:.2f}")

    set_cycle(0, archive_previous=False)

    def update(_frame):
        local_frame = state['frame_in_cycle']
        i   = min(local_frame * STEP, N_EACH - 1)
        i0  = max(0, i - TRAIL)
        traj  = state['traj']
        dists = state['dists']
        seg   = traj[i0:i+1]

        if i > state['ghost_last_i']:
            state['ghost_xyz'] = append_ghost_points(state['ghost_xyz'], traj[state['ghost_last_i']:i+1])
            apply_ghost()
            state['ghost_last_i'] = i + 1

        line.set_data(seg[:,0], seg[:,1])
        line.set_3d_properties(seg[:,2])
        dot._offsets3d = ([traj[i,0]], [traj[i,1]], [traj[i,2]])

        for k, (sg, tx) in enumerate(zip(eq_glow, eq_texts)):
            d = dists[i, k]
            if d < GLOW_DIST:
                br = max(0.0, 1.0 - d / GLOW_DIST)
                sg._sizes = np.array([400 * br**2])
                sg.set_alpha(0.9 * br)
                tx.set_alpha(1.0)
            else:
                sg._sizes = np.array([0])
                sg.set_alpha(0.0)
                tx.set_alpha(0.25)

        b_now = b_vals[state['traj_idx']]
        c_now = c_vals[state['traj_idx']]
        t_now = (i / max(N_EACH - 1, 1)) * T_EACH
        title_obj.set_text(
            f'Wang 2-Scroll  a={A} b={b_now:.3f} c={c_now:.3f} d={D}  '
            f'k1={k1_ref:.3f} k2={k2_ref:.3f}  '
            f'cycle={state["traj_idx"]+1}/{N_TRAJS}  t={t_now:.1f}')

        if local_frame >= n_frames - 1:
            next_idx = (state['traj_idx'] + 1) % N_TRAJS
            set_cycle(next_idx, archive_previous=True)
        else:
            state['frame_in_cycle'] += 1

        return [ghost, line, dot] + eq_glow + eq_texts

    ani = animation.FuncAnimation(
        fig, update,
        frames=itertools.count(),
        interval=1000 // FPS,
        blit=False,
        cache_frame_data=False,
    )

    plt.show()


if __name__ == '__main__':
    main()
