import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def _midplane_omega2(sim, eps_pc=1.0):
    """Midplane curvature Phi''(0) = d(dphi_dz)/dz at z=0 of the FULL potential.
    This is the harmonic frequency^2 that matches the true potential near z=0.
    Computed by a symmetric finite difference on sim.dphi_dz (odd function)."""
    dphi = sim.dphi_dz(np.array([eps_pc, -eps_pc]), h_pc_dm=sim.h_pc_dm)
    return float((dphi[0] - dphi[1]) / (2.0 * eps_pc))


def _evolve_harmonic(sim, z0, w0, t_myr, n_steps, n_snaps, omega2):
    """Leapfrog integration of the pure harmonic potential a(z) = -omega2 * z,
    using the SAME time units / stepping as sim.evolve_record so the harmonic
    and full-potential animations are directly comparable."""
    z = np.asarray(z0, dtype=float).copy()
    w = np.asarray(w0, dtype=float).copy()
    t_end = t_myr * sim.MYR_TO_TUNIT
    dt = t_end / float(n_steps)

    snap_idx = np.unique(np.linspace(0, n_steps, n_snaps, dtype=int))
    Zs = np.empty((snap_idx.size, z.size), dtype=np.float32)
    Ws = np.empty((snap_idx.size, w.size), dtype=np.float32)
    Ts_myr = np.empty(snap_idx.size, dtype=np.float32)

    a = -omega2 * z
    w += 0.5 * dt * a
    sc = 0
    if snap_idx[0] == 0:
        Zs[sc] = z; Ws[sc] = w; Ts_myr[sc] = 0.0; sc += 1
    for step in range(1, n_steps + 1):
        z += dt * w
        a = -omega2 * z
        w += dt * a
        if sc < snap_idx.size and step == snap_idx[sc]:
            Zs[sc] = z; Ws[sc] = w
            Ts_myr[sc] = step * dt / sim.MYR_TO_TUNIT
            sc += 1
    w -= 0.5 * dt * a
    return Zs, Ws, Ts_myr


def animate_orbits(sim, ics=None, t_myr=1500.0, n_steps=50_000, n_snaps=500,
                   use_satellite=False, show_satellite=None,
                   pulse_cut_sigmas=2.5,
                   harmonic=False,
                   save_path='plots/phase_space_orbits.gif',
                   fps=30):
    """Animate a few stars on iso-energy contours of the vertical potential.

    Background: analytical iso-energy contours  w = +/- sqrt(2*(E - Phi(z)))
    Foreground: dots integrated moving along the orbits.

    harmonic: if True, replace the full sech^2 potential with a pure harmonic
        one, Phi(z) = 0.5 * omega2 * z^2, where omega2 = Phi''(0) is the true
        midplane curvature of the full potential. The two potentials agree near
        z=0 and diverge at large |z| (the full potential is softer / flatter),
        so this shows how anharmonicity changes the orbits. In harmonic mode the
        satellite perturbation is disabled (pure harmonic reference).
    Returns the FuncAnimation object.
    """

    if ics is None:
        ics = np.array([
            [ 50,   4],
            [150,  10],
            [300,  15],
            [500,  20],
            [700,  25],
        ], dtype=float)
    ics = np.asarray(ics, dtype=float)

    # In harmonic mode the satellite is disabled (pure harmonic reference).
    if harmonic:
        use_satellite = False

    # ── Build Phi(z) ──
    z_grid = np.linspace(-sim.z_max_pc, sim.z_max_pc, 4001)
    dz = z_grid[1] - z_grid[0]
    if harmonic:
        omega2 = _midplane_omega2(sim)
        Phi = 0.5 * omega2 * z_grid**2          # already 0 at z=0
    else:
        dphi = sim.dphi_dz(z_grid, h_pc_dm=sim.h_pc_dm)
        Phi  = np.cumsum(dphi) * dz
        Phi -= Phi[len(Phi)//2]

    def phi_at(z):
        return np.interp(z, z_grid, Phi)

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(ics)))

    # ── Integrate trajectories ──
    if harmonic:
        Zs, Ws, Ts = _evolve_harmonic(
            sim, ics[:, 0].copy(), ics[:, 1].copy(),
            t_myr, n_steps, n_snaps, omega2)
    else:
        Zs, Ws, Ts = sim.evolve_record(
            ics[:, 0].copy(), ics[:, 1].copy(),
            t_obs_myr=t_myr, n_steps=n_steps, n_snaps=n_snaps,
            seed=0, use_satellite=use_satellite, h_pc_dm=sim.h_pc_dm,
        )

    # ── Iso-energy contours ──
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("z  [pc]", fontsize=12)
    ax.set_ylabel("w  [km/s]", fontsize=12)
    if harmonic:
        pot_str = "harmonic potential"
    else:
        pot_str = ("sech\u00b2 potential" +
                   (" with satellite" if use_satellite else " (no satellite)"))
    ax.set_title(f"Phase-space orbits — {pot_str}", fontsize=12)
    ax.axhline(0, color='k', lw=0.4)
    ax.axvline(0, color='k', lw=0.4)

    for (z0, w0), color in zip(ics, colors):
        E = 0.5 * w0**2 + phi_at(z0)
        dE = E - Phi
        mask = dE >= 0
        w_upper = np.sqrt(2 * np.where(mask, dE, 0.0))
        zc = z_grid[mask]
        ax.plot(zc,  w_upper[mask], color=color, lw=1.2, alpha=0.4)
        ax.plot(zc, -w_upper[mask], color=color, lw=1.2, alpha=0.4)

    # ── Animated dots ──
    dots = [ax.plot([], [], 'o', color=c, ms=7, zorder=5)[0] for c in colors]
    time_text = ax.text(0.02, 0.96, '', transform=ax.transAxes, fontsize=9,
                         verticalalignment='top')

    ax.set_xlim(-820, 820)
    ax.set_ylim(-35, 35)

    # ── Satellite overlay (only when use_satellite is on) ──
    if show_satellite is None:
        show_satellite = use_satellite
    sigma_t_internal = sim.sigma_t_myr * sim.MYR_TO_TUNIT
    t_cut_myr = pulse_cut_sigmas * sim.sigma_t_myr
    sat_line = ax.axvline(np.nan, color='red', lw=2.0, alpha=0.0, zorder=4)
    sat_label = ax.text(0.98, 0.96, '', transform=ax.transAxes, fontsize=9,
                        ha='right', va='top', color='red')

    def sat_state(t_myr_now):
        z_sat = sim.z0_sat_pc + sim.w_sat_kms * (t_myr_now * sim.MYR_TO_TUNIT)
        env = math.exp(-0.5 * (t_myr_now / sim.sigma_t_myr) ** 2) if t_myr_now <= t_cut_myr else 0.0
        return z_sat, env

    def init():
        for d in dots:
            d.set_data([], [])
        time_text.set_text('')
        sat_line.set_alpha(0.0)
        sat_label.set_text('')
        return dots + [time_text, sat_line, sat_label]

    def update(frame):
        for i, d in enumerate(dots):
            d.set_data([Zs[frame, i]], [Ws[frame, i]])
        t_now = float(Ts[frame])
        time_text.set_text(f"t = {t_now:.0f} Myr")
        if show_satellite:
            z_sat, env = sat_state(t_now)
            sat_line.set_xdata([z_sat, z_sat])
            sat_line.set_alpha(0.85 * env)
            if env > 0.0:
                sat_label.set_text(f"satellite z = {z_sat:.0f} pc  ({env:.2f})")
            else:
                sat_label.set_text('')
        return dots + [time_text, sat_line, sat_label]

    anim = FuncAnimation(fig, update, frames=n_snaps,
                         init_func=init, interval=30, blit=False)
    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(save_dir, exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.mp4':
            anim.save(save_path, writer='ffmpeg', fps=fps, dpi=120)
        else:
            anim.save(save_path, writer='pillow', fps=fps, dpi=120)
        abs_path = os.path.abspath(save_path)
        size_kb = os.path.getsize(abs_path) / 1024
        print(f"Animation saved: {abs_path}  ({size_kb:.1f} KB)")

    plt.close(fig)
    return anim
