"""
Sensitivity test: injection-recovery of a passing-satellite perturbation, scored
with a potential-free steady-state statistic.

Framework role (see detection_framework.md, Part IV):
  * We inject a physically motivated, controlled time-dependent signal (a satellite
    flyby, amplitude knob = Sigma_sat) into an otherwise stationary equilibrium disc.
  * We score each mock snapshot with a statistic that must vanish in steady state,
    WITHOUT fitting the potential, and build:
      - the NULL distribution of the statistic (Sigma_sat = 0, many seeds)
        -> the false-positive background (robustness side);
      - the statistic vs injected amplitude (sensitivity side)
        -> detection fraction and the minimum detectable amplitude.

The statistic (Level-0, potential-free):
  A steady-state DF depends only on the vertical energy E_z = w^2/2 + Phi(z), so it
  is EVEN in w and the density-weighted mean-velocity profile <w>(z) is identically
  zero. Equivalently the vertical mass flux nu(z)<w>(z) vanishes, so by continuity
  d nu / dt = -d[nu <w>]/dz = 0. A nonzero <w>(z) is therefore a direct, potential-
  free detection of df/dt != 0.

    A1 = sqrt( sum_b n_b <w>_b^2 / sum_b n_b )     [km/s]

  With finite N there is an irreducible Poisson floor E[A1^2] ~ (n_bins) sigma_w^2 / N;
  that floor IS the null background this test calibrates.

Run:  /usr/bin/python3 sensitivity_satellite.py
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase_space_simulation import Widmark1DSimulation


# ----------------------------------------------------------------------
# statistics computed on an observed snapshot (z, w)
# ----------------------------------------------------------------------
def bulk_velocity_rms(z, w, zlim=700.0, n_bins=25, n_min=30):
    """Density-weighted RMS of <w>(z): the Level-0 steady-state violation [km/s]."""
    m = np.abs(z) < zlim
    z, w = z[m], w[m]
    edges = np.linspace(-zlim, zlim, n_bins + 1)
    idx = np.clip(np.digitize(z, edges) - 1, 0, n_bins - 1)
    num = den = 0.0
    for b in range(n_bins):
        sel = idx == b
        n = int(sel.sum())
        if n >= n_min:
            num += n * w[sel].mean() ** 2
            den += n
    return float(np.sqrt(num / den)) if den > 0 else np.nan


def mean_w_profile(z, w, zlim=700.0, n_bins=25, n_min=30):
    """Return (z_centers, <w>(z)) for plotting."""
    edges = np.linspace(-zlim, zlim, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(z, edges) - 1, 0, n_bins - 1)
    prof = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = (idx == b) & (np.abs(z) < zlim)
        if sel.sum() >= n_min:
            prof[b] = w[sel].mean()
    return centers, prof


# ----------------------------------------------------------------------
# one mock: equilibrium IC -> evolve with satellite of amplitude Sigma_sat
# ----------------------------------------------------------------------
def run_one(Sigma_sat, seed, N=20000, t_obs=500.0, n_steps=4000, return_snap=False):
    sim = Widmark1DSimulation()
    sim.Sigma_sat = float(Sigma_sat)
    z0, w0, _ = sim.sample_initial_particles(N, seed=seed)
    Zs, Ws, _ = sim.evolve_record(
        z0, w0, t_obs_myr=t_obs, n_steps=n_steps, n_snaps=2, use_satellite=True
    )
    z, w = Zs[-1], Ws[-1]
    out = {"A1": bulk_velocity_rms(z, w)}
    if return_snap:
        out["z"], out["w"] = z, w
    return out


# ----------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------
def main():
    N = 20000
    t_obs = 500.0
    n_steps = 4000
    amplitudes = [0.0, 1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 13.5, 20.0]
    n_seed_null = 18
    n_seed_sig = 9

    t_start = time.time()
    results = {}  # Sigma_sat -> list of A1
    for A in amplitudes:
        n_seed = n_seed_null if A == 0.0 else n_seed_sig
        vals = []
        for s in range(n_seed):
            vals.append(run_one(A, seed=1000 + s, N=N, t_obs=t_obs, n_steps=n_steps)["A1"])
        results[A] = np.array(vals)
        print("Sigma_sat=%5.1f  A1 median=%.3f  (n=%d)"
              % (A, np.nanmedian(results[A]), n_seed))

    # ---- null background and detection threshold ----
    null = results[0.0]
    null_med = float(np.nanmedian(null))
    null_p95 = float(np.nanpercentile(null, 95))
    null_p05 = float(np.nanpercentile(null, 5))

    amps = np.array(amplitudes)
    med = np.array([np.nanmedian(results[A]) for A in amplitudes])
    det_frac = np.array([np.mean(results[A] > null_p95) for A in amplitudes])

    # minimum detectable amplitude: first amplitude with >=90% detection fraction
    A_min = np.nan
    for i in range(1, len(amps)):
        if det_frac[i] >= 0.9:
            # linear interpolation in amplitude on detection fraction
            if det_frac[i - 1] < 0.9 and det_frac[i] != det_frac[i - 1]:
                f = (0.9 - det_frac[i - 1]) / (det_frac[i] - det_frac[i - 1])
                A_min = amps[i - 1] + f * (amps[i] - amps[i - 1])
            else:
                A_min = amps[i]
            break

    print("\nNull A1: median=%.3f  5-95%% = [%.3f, %.3f] km/s"
          % (null_med, null_p05, null_p95))
    print("Minimum detectable Sigma_sat (90%% at null-95th threshold): %.2f Msun/pc^2"
          % A_min)
    print("Total wall time: %.1f s" % (time.time() - t_start))

    # ---- figure 1: the injection-recovery curve ----
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=140)
    for A in amplitudes:
        ax.scatter([A] * len(results[A]), results[A], s=14, color="0.6",
                   alpha=0.6, zorder=2)
    ax.plot(amps, med, "-o", color="C0", lw=2, zorder=3, label="median statistic")
    ax.axhspan(null_p05, null_p95, color="C3", alpha=0.15, zorder=1,
               label="null 5-95% (stationary)")
    ax.axhline(null_p95, color="C3", ls="--", lw=1.2, zorder=1,
               label="null 95th pct = detection threshold")
    if np.isfinite(A_min):
        ax.axvline(A_min, color="k", ls=":", lw=1.2)
        ax.text(A_min, ax.get_ylim()[1] * 0.96,
                "  min. detectable\n  $\\Sigma_{sat}$=%.1f" % A_min,
                va="top", fontsize=9)
    ax.set_xlabel("injected satellite amplitude  $\\Sigma_{sat}$  [$M_\\odot$/pc$^2$]")
    ax.set_ylabel("steady-state statistic  $A_1$  [km/s]")
    ax.set_title("Sensitivity: satellite injection-recovery (potential-free statistic)")
    ax.legend(loc="upper left", fontsize=8)

    axr = ax.twinx()
    axr.plot(amps, det_frac, "-s", color="C2", lw=1.4, ms=5, alpha=0.9)
    axr.axhline(0.9, color="C2", ls=":", lw=1.0, alpha=0.7)
    axr.set_ylabel("detection fraction", color="C2")
    axr.tick_params(axis="y", labelcolor="C2")
    axr.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig("plots/sensitivity_satellite_curve.png")
    print("saved plots/sensitivity_satellite_curve.png")

    # ---- figure 2: what the signal looks like ----
    detA = 6.0 if 6.0 in results else amplitudes[-1]
    snap0 = run_one(0.0, seed=1000, N=60000, t_obs=t_obs, n_steps=n_steps, return_snap=True)
    snapS = run_one(detA, seed=1000, N=60000, t_obs=t_obs, n_steps=n_steps, return_snap=True)

    fig2, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=140)
    zb = np.linspace(-700, 700, 160)
    wb = np.linspace(-55, 55, 160)
    for ax_, snap, ttl in [(axes[0], snap0, "stationary (null)"),
                           (axes[1], snapS, "satellite $\\Sigma_{sat}$=%.0f" % detA)]:
        H, _, _ = np.histogram2d(snap["z"], snap["w"], bins=[zb, wb])
        ax_.imshow(np.log1p(H).T, origin="lower",
                   extent=[-700, 700, -55, 55], aspect="auto", cmap="magma")
        ax_.set_xlabel("z [pc]"); ax_.set_ylabel("w [km/s]"); ax_.set_title(ttl)

    c0, p0 = mean_w_profile(snap0["z"], snap0["w"])
    cS, pS = mean_w_profile(snapS["z"], snapS["w"])
    axes[2].axhline(0, color="0.7", lw=1)
    axes[2].plot(c0, p0, "-o", ms=3, color="C0", label="null")
    axes[2].plot(cS, pS, "-o", ms=3, color="C3",
                 label="satellite $\\Sigma_{sat}$=%.0f" % detA)
    axes[2].set_xlabel("z [pc]"); axes[2].set_ylabel(r"$\langle w \rangle(z)$ [km/s]")
    axes[2].set_title("mean-velocity profile (steady state $\\Rightarrow$ 0)")
    axes[2].legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig("plots/sensitivity_satellite_snapshots.png")
    print("saved plots/sensitivity_satellite_snapshots.png")

    np.savez("plots/sensitivity_satellite_results.npz",
             amplitudes=amps, medians=med, det_frac=det_frac,
             null_p95=null_p95, A_min=A_min,
             **{("A1_%.1f" % A): results[A] for A in amplitudes})


if __name__ == "__main__":
    main()
