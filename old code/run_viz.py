import matplotlib
matplotlib.use("Agg")

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.mixture import GaussianMixture

# ==========================================================
# Widmark 1D: Simulation + Hamiltonian flow + MLE for h_pc_dm
# SPEED OPTIMIZATIONS INCLUDED:
#   (1) Precompute baryon force on z_grid and interpolate (np.interp) each step
#   (2) Fast 2-stage MLE: subsample stars + coarse->refined grid scan
# ALSO:
#   - plot phase space at t=0 and t=t_final
#   - optional animation / spiral map retained
# ==========================================================


# ==========================================================
# Core: physics + sampling + integration
# Units:
#   z in pc
#   w in km/s
#   internal time unit: pc/(km/s)
#   1 km/s ≈ 1.022712 pc/Myr  => t_internal = t_myr * MYR_TO_TUNIT
# ==========================================================
@dataclass
class Widmark1DSimulation:
    # -------- mass model --------
    rho0_thin: float = 0.04
    rho0_thick: float = 0.01
    rho0_gas: float = 0.05

    # DM density amplitude fixed; we infer DM scale height h_pc_dm
    rho_DM: float = 0.01
    h_pc_dm: float = 250.0

    # velocity dispersions (km/s) (kept for completeness / your equilibrium sampler)
    sigma_w_thin: float = 16.0
    sigma_w_thick: float = 24.0
    sigma_w_gas: float = 8.0

    # baryon scale heights (pc)
    h_thin_pc: float = 200.0
    h_thick_pc: float = 400.0
    h_gas_pc: float = 800.0

    # -------- satellite perturbation --------
    Sigma_sat: float = 6.0
    sigma_t_myr: float = 80.0
    H_sat_pc: float = 50.0
    z0_sat_pc: float = 600.0
    w_sat_kms: float = -50.0

    # -------- geometry / technical --------
    z_max_pc: float = 1000.0
    n_z: int = 2001

    # mask params for analysis class
    Z_lim_mask_pc: float = 700.0
    W_lim_mask_kms: float = 40.0

    # gravitational constant in pc (km/s)^2 / Msun
    G: float = 4.3009e-3

    def __post_init__(self):
        self.z_grid = np.linspace(-self.z_max_pc, self.z_max_pc, self.n_z)
        self.MYR_TO_TUNIT = 1.022712

        # -------- SPEEDUP (1): precompute baryon force on grid once --------
        self._dphi_dz_baryons_grid = (
            self.dphi_dz_component_sech2(self.z_grid, self.rho0_thin,  self.h_thin_pc)
            + self.dphi_dz_component_sech2(self.z_grid, self.rho0_thick, self.h_thick_pc)
            + self.dphi_dz_component_sech2(self.z_grid, self.rho0_gas,   self.h_gas_pc)
        )

    # ==========================================================
    # sech^2 sheet potential pieces:
    #   Φ(z) = 4πG ρ0 h^2 log cosh(z/h)
    #   dΦ/dz = 4πG ρ0 h tanh(z/h)
    # ==========================================================
    def phi_component_sech2(self, z: np.ndarray, rho0: float, h_pc: float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return 4.0 * math.pi * self.G * rho0 * h_pc**2 * np.log(np.cosh(z / h_pc))

    def dphi_dz_component_sech2(self, z: np.ndarray, rho0: float, h_pc: float) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return 4.0 * math.pi * self.G * rho0 * h_pc * np.tanh(z / h_pc)

    # -------- baryons (FAST via interpolation) --------
    def dphi_dz_baryons(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return np.interp(z, self.z_grid, self._dphi_dz_baryons_grid)

    # -------- DM term with fitted scale height --------
    def dphi_dz_dm(self, z: np.ndarray, h_pc_dm: Optional[float] = None) -> np.ndarray:
        if h_pc_dm is None:
            h_pc_dm = self.h_pc_dm
        return self.dphi_dz_component_sech2(z, self.rho_DM, float(h_pc_dm))

    def dphi_dz(self, z: np.ndarray, h_pc_dm: Optional[float] = None) -> np.ndarray:
        return self.dphi_dz_baryons(z) + self.dphi_dz_dm(z, h_pc_dm=h_pc_dm)

    # ==========================================================
    # Sampling
    # ==========================================================
    def sample_near_midplane(
        self,
        N: int = 10_000,
        z_mean_pc: float = 0.0,
        z_sigma_pc: float = 20.0,
        w_mean_kms: float = 0.0,
        w_sigma_kms: float = 40.0,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        z0 = rng.normal(z_mean_pc, z_sigma_pc, size=N)
        w0 = rng.normal(w_mean_kms, w_sigma_kms, size=N)
        return z0, w0

    def _sample_component_equilibrium(
        self, N: int, sigma_w: float, rng: np.random.Generator, h_pc_dm: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Isothermal equilibrium in fixed Φ(z):
          ρ(z) ∝ exp[-Φ(z)/σ_w^2],  w|z ~ N(0, σ_w)
        """
        # build Φ(z) on grid (baryons + dm)
        phi_vals = (
            self.phi_component_sech2(self.z_grid, self.rho0_thin,  self.h_thin_pc)
            + self.phi_component_sech2(self.z_grid, self.rho0_thick, self.h_thick_pc)
            + self.phi_component_sech2(self.z_grid, self.rho0_gas,   self.h_gas_pc)
            + self.phi_component_sech2(self.z_grid, self.rho_DM, float(self.h_pc_dm if h_pc_dm is None else h_pc_dm))
        )
        rho_unnorm = np.exp(-phi_vals / sigma_w**2)
        p_z = rho_unnorm / rho_unnorm.sum()

        z_samples = rng.choice(self.z_grid, size=N, p=p_z)
        w_samples = rng.normal(loc=0.0, scale=sigma_w, size=N)
        return z_samples, w_samples

    def sample_initial_particles(
        self, N_total: int, seed: Optional[int] = None, h_pc_dm: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        weights = np.array([self.rho0_thin, self.rho0_thick, self.rho0_gas], float)
        weights /= weights.sum()
        N_comp = rng.multinomial(N_total, weights)

        z_list, w_list, comp_list = [], [], []

        if N_comp[0] > 0:
            z_t, w_t = self._sample_component_equilibrium(N_comp[0], self.sigma_w_thin, rng, h_pc_dm=h_pc_dm)
            z_list.append(z_t); w_list.append(w_t); comp_list.append(np.zeros(N_comp[0], dtype=int))
        if N_comp[1] > 0:
            z_t, w_t = self._sample_component_equilibrium(N_comp[1], self.sigma_w_thick, rng, h_pc_dm=h_pc_dm)
            z_list.append(z_t); w_list.append(w_t); comp_list.append(np.ones(N_comp[1], dtype=int))
        if N_comp[2] > 0:
            z_t, w_t = self._sample_component_equilibrium(N_comp[2], self.sigma_w_gas, rng, h_pc_dm=h_pc_dm)
            z_list.append(z_t); w_list.append(w_t); comp_list.append(np.full(N_comp[2], 2, dtype=int))

        return np.concatenate(z_list), np.concatenate(w_list), np.concatenate(comp_list)

    # ==========================================================
    # Satellite force
    # ==========================================================
    def Kz_sat_common(self, t_internal: float, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        sigma_t_internal = self.sigma_t_myr * self.MYR_TO_TUNIT
        tau = t_internal / sigma_t_internal
        z_sat = self.z0_sat_pc + self.w_sat_kms * t_internal
        return (
            -4.0 * math.pi * self.G * self.Sigma_sat
            * np.exp(-0.5 * tau**2)
            * np.tanh((z - z_sat) / self.H_sat_pc)
        )

    # ==========================================================
    # Symplectic evolution + recording
    # ==========================================================
    def evolve_record(
        self,
        z0: np.ndarray,
        w0: np.ndarray,
        t_obs_myr: float = 400.0,
        n_steps: int = 20_000,
        n_snaps: int = 80,
        seed: Optional[int] = None,
        pulse_cut_sigmas: float = 2.5,
        use_satellite: bool = True,
        h_pc_dm: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        z = np.asarray(z0, dtype=float).copy()
        w = np.asarray(w0, dtype=float).copy()

        # keep deterministic unless you explicitly want randomness
        s_i = np.ones_like(z)

        sigma_t_internal = self.sigma_t_myr * self.MYR_TO_TUNIT
        t_end = t_obs_myr * self.MYR_TO_TUNIT
        dt = t_end / float(n_steps)
        t_cut = pulse_cut_sigmas * sigma_t_internal

        snap_idx = np.unique(np.linspace(0, n_steps, n_snaps, dtype=int))
        Zs = np.empty((snap_idx.size, z.size), dtype=np.float32)
        Ws = np.empty((snap_idx.size, w.size), dtype=np.float32)
        Ts_myr = np.empty(snap_idx.size, dtype=np.float32)

        def accel(t_internal, z_now):
            a_bg = -self.dphi_dz(z_now, h_pc_dm=h_pc_dm)
            if (not use_satellite) or (t_internal > t_cut):
                return a_bg
            return a_bg + s_i * self.Kz_sat_common(t_internal, z_now)

        t = 0.0
        a = accel(t, z)
        w += 0.5 * dt * a  # half kick

        snap_counter = 0
        if snap_idx[0] == 0:
            Zs[snap_counter] = z
            Ws[snap_counter] = w
            Ts_myr[snap_counter] = 0.0
            snap_counter += 1

        for step in range(1, n_steps + 1):
            z += dt * w
            t += dt
            a = accel(t, z)
            w += dt * a

            if snap_counter < snap_idx.size and step == snap_idx[snap_counter]:
                Zs[snap_counter] = z
                Ws[snap_counter] = w
                Ts_myr[snap_counter] = t / self.MYR_TO_TUNIT
                snap_counter += 1

        w -= 0.5 * dt * a
        return Zs, Ws, Ts_myr

    # ==========================================================
    # Backward integrator (Hamiltonian flow inverse) for likelihood
    # ==========================================================
    def back_integrate(
        self,
        z_obs: np.ndarray,
        w_obs: np.ndarray,
        t_obs_myr: float,
        n_steps: int = 10_000,
        use_satellite: bool = False,
        h_pc_dm: Optional[float] = None,
        pulse_cut_sigmas: float = 2.5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        z = np.asarray(z_obs, dtype=float).copy()
        w = np.asarray(w_obs, dtype=float).copy()

        t_end = t_obs_myr * self.MYR_TO_TUNIT
        dt = -t_end / float(n_steps)

        sigma_t_internal = self.sigma_t_myr * self.MYR_TO_TUNIT
        t_cut = pulse_cut_sigmas * sigma_t_internal

        def accel(t_internal, z_now):
            a_bg = -self.dphi_dz(z_now, h_pc_dm=h_pc_dm)
            if (not use_satellite) or (t_internal > t_cut):
                return a_bg
            return a_bg + self.Kz_sat_common(t_internal, z_now)

        t = t_end
        a = accel(t, z)
        w += 0.5 * dt * a

        for _ in range(n_steps):
            z += dt * w
            t += dt
            a = accel(t, z)
            w += dt * a

        w -= 0.5 * dt * a
        return z, w

    # ==========================================================
    # Likelihood pieces: f0 Gaussian at t=0 + Liouville mapping
    # ==========================================================
    def _log_f0_gaussian(self, z0: np.ndarray, w0: np.ndarray, z_sigma: float, w_sigma: float) -> np.ndarray:
        z0 = np.asarray(z0, float)
        w0 = np.asarray(w0, float)
        sz = float(z_sigma)
        sw = float(w_sigma)
        log_norm = -np.log(2.0 * math.pi * sz * sw)
        return log_norm - 0.5 * (z0 / sz) ** 2 - 0.5 * (w0 / sw) ** 2

    def log_likelihood_h_pc_dm(
        self,
        z_obs: np.ndarray,
        w_obs: np.ndarray,
        t_obs_myr: float,
        h_pc_dm: float,
        z0_sigma: float,
        w0_sigma: float,
        n_steps: int = 10_000,
        use_satellite_in_likelihood: bool = False,
    ) -> float:
        z0_hat, w0_hat = self.back_integrate(
            z_obs, w_obs, t_obs_myr,
            n_steps=n_steps,
            use_satellite=use_satellite_in_likelihood,
            h_pc_dm=h_pc_dm,
        )
        return float(np.sum(self._log_f0_gaussian(z0_hat, w0_hat, z0_sigma, w0_sigma)))


# ==========================================================
# Analysis: plotting + spiral map + animation (unchanged style)
# ==========================================================
@dataclass
class Widmark1DAnalysis:
    sim: Widmark1DSimulation

    def mask_M(self, Z: np.ndarray, W: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        W = np.asarray(W, dtype=float)
        r2 = (Z / self.sim.Z_lim_mask_pc) ** 2 + (W / self.sim.W_lim_mask_kms) ** 2
        x = 10.0 * (r2 - 1.0)
        return 1.0 / (1.0 + np.exp(-x))

    def data_bulk_spiral_histograms(
        self,
        z_data: np.ndarray,
        w_data: np.ndarray,
        z_bins: np.ndarray,
        w_bins: np.ndarray,
        n_gmm: int = 6,
        random_state: int = 0,
        residual_mode: str = "fractional",  # "absolute" or "fractional"
    ):
        d, _, _ = np.histogram2d(z_data, w_data, bins=[z_bins, w_bins])

        X = np.vstack([z_data, w_data]).T
        gmm = GaussianMixture(
            n_components=n_gmm,
            covariance_type="full",
            random_state=random_state,
        )
        gmm.fit(X)

        z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])
        w_centers = 0.5 * (w_bins[:-1] + w_bins[1:])
        ZZ, WW = np.meshgrid(z_centers, w_centers, indexing="ij")
        grid_points = np.column_stack([ZZ.ravel(), WW.ravel()])
        bulk_pdf = np.exp(gmm.score_samples(grid_points)).reshape(ZZ.shape)

        M = self.mask_M(ZZ, WW)
        eps = 1e-12
        data_sum = np.sum(M * d)
        pdf_sum = np.sum(M * bulk_pdf)
        B = bulk_pdf * (data_sum / (pdf_sum + eps))  # background in "counts" units

        delta = d - B  # signed residual

        if residual_mode == "absolute":
            R = delta
        elif residual_mode == "fractional":
            R = delta / (B + eps)
        else:
            raise ValueError("residual_mode must be 'absolute' or 'fractional'.")

        R = M * R
        return d, B, R, M, ZZ, WW

    def plot_end_distribution(
        self,
        z_fin: np.ndarray,
        w_fin: np.ndarray,
        *,
        z_lim: float = 700.0,
        w_lim: float = 60.0,
        nbins: int = 240,
        use_log: bool = True,
        title: str = "Phase space histogram",
    ):
        z_bins = np.linspace(-z_lim, z_lim, nbins + 1)
        w_bins = np.linspace(-w_lim, w_lim, nbins + 1)

        H, _, _ = np.histogram2d(z_fin, w_fin, bins=[z_bins, w_bins])
        img = np.log1p(H) if use_log else H

        plt.figure(figsize=(6.2, 5.2), dpi=140)
        plt.imshow(
            img.T,
            origin="lower",
            extent=[-z_lim, z_lim, -w_lim, w_lim],
            aspect="auto",
        )
        plt.xlabel("z [pc]")
        plt.ylabel("w [km/s]")
        plt.title(title)
        plt.colorbar(label="log(1 + counts)" if use_log else "counts")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

    def animate_phase_space_hist(
        self,
        Zs: np.ndarray,
        Ws: np.ndarray,
        Ts_myr: np.ndarray,
        z_lim: float = 700.0,
        w_lim: float = 60.0,
        nbins: int = 240,
        use_log: bool = True,
        interval_ms: int = 40,
    ):
        z_bins = np.linspace(-z_lim, z_lim, nbins + 1)
        w_bins = np.linspace(-w_lim, w_lim, nbins + 1)

        fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=140)
        ax.set_xlabel("z [pc]")
        ax.set_ylabel("w [km/s]")
        ax.set_xlim(-z_lim, z_lim)
        ax.set_ylim(-w_lim, w_lim)
        ax.grid(alpha=0.25)

        H, _, _ = np.histogram2d(Zs[0], Ws[0], bins=[z_bins, w_bins])
        img = np.log1p(H) if use_log else H

        im = ax.imshow(
            img.T,
            origin="lower",
            extent=[-z_lim, z_lim, -w_lim, w_lim],
            aspect="auto",
        )
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label("log(1 + counts)" if use_log else "counts")

        title = ax.set_title(f"t = {Ts_myr[0]:.1f} Myr")

        def update(i):
            H, _, _ = np.histogram2d(Zs[i], Ws[i], bins=[z_bins, w_bins])
            img = np.log1p(H) if use_log else H
            im.set_data(img.T)
            title.set_text(f"t = {Ts_myr[i]:.1f} Myr")
            return (im, title)

        ani = FuncAnimation(fig, update, frames=len(Ts_myr), interval=interval_ms, blit=False)
        plt.tight_layout()
        return fig, ani

    def plot_spiral_map(
        self,
        z_fin: np.ndarray,
        w_fin: np.ndarray,
        z_bins: np.ndarray,
        w_bins: np.ndarray,
        n_gmm: int = 6,
        random_state: int = 0,
        title: str = "Fluctuations around the mean:  M * (d - B) / B",
    ):
        _, _, spiral, _, _, _ = self.data_bulk_spiral_histograms(
            z_fin, w_fin, z_bins=z_bins, w_bins=w_bins, n_gmm=n_gmm, random_state=random_state
        )
        plt.figure(figsize=(6.2, 5.2), dpi=140)
        plt.imshow(
            spiral.T,
            origin="lower",
            extent=[z_bins[0], z_bins[-1], w_bins[0], w_bins[-1]],
            aspect="auto",
        )
        plt.xlabel("z [pc]")
        plt.ylabel("w [km/s]")
        plt.title(title)
        plt.colorbar(label="relative overdensity")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()


# ==========================================================
# FAST 2-STAGE MLE helper (big runtime reduction)
# ==========================================================
def fit_h_pc_dm_mle_fast(
    sim: Widmark1DSimulation,
    z_obs: np.ndarray,
    w_obs: np.ndarray,
    t_obs_myr: float,
    z0_sigma: float,
    w0_sigma: float,
    h_bounds=(50.0, 800.0),
    # stage 1 (fast coarse)
    n_sub_1=3000,
    n_steps_1=2000,
    n_grid_1=31,
    # stage 2 (refine)
    n_sub_2=12000,
    n_steps_2=7000,
    n_grid_2=41,
    use_satellite_in_likelihood=False,
    seed=0,
):
    rng = np.random.default_rng(seed)
    N = z_obs.size
    idx1 = rng.choice(N, size=min(n_sub_1, N), replace=False)
    idx2 = rng.choice(N, size=min(n_sub_2, N), replace=False)

    hs1 = np.linspace(h_bounds[0], h_bounds[1], n_grid_1)
    ll1 = np.array([
        sim.log_likelihood_h_pc_dm(
            z_obs[idx1], w_obs[idx1], t_obs_myr, h,
            z0_sigma=z0_sigma, w0_sigma=w0_sigma,
            n_steps=n_steps_1,
            use_satellite_in_likelihood=use_satellite_in_likelihood
        )
        for h in hs1
    ])
    h1 = float(hs1[np.argmax(ll1)])

    # local refinement window around h1
    span = 0.25 * (h_bounds[1] - h_bounds[0])
    lo = max(h_bounds[0], h1 - 0.15 * span)
    hi = min(h_bounds[1], h1 + 0.15 * span)

    hs2 = np.linspace(lo, hi, n_grid_2)
    ll2 = np.array([
        sim.log_likelihood_h_pc_dm(
            z_obs[idx2], w_obs[idx2], t_obs_myr, h,
            z0_sigma=z0_sigma, w0_sigma=w0_sigma,
            n_steps=n_steps_2,
            use_satellite_in_likelihood=use_satellite_in_likelihood
        )
        for h in hs2
    ])
    h2 = float(hs2[np.argmax(ll2)])
    return (h2, (hs1, ll1), (hs2, ll2))


# ==========================================================
# Minimal plotting: phase space at t=0 and at t_final
# ==========================================================
def plot_phase_space_hist(z, w, title, z_lim=800.0, w_lim=60.0, nbins=240):
    z_bins = np.linspace(-z_lim, z_lim, nbins + 1)
    w_bins = np.linspace(-w_lim, w_lim, nbins + 1)
    H, _, _ = np.histogram2d(z, w, bins=[z_bins, w_bins])

    plt.figure(figsize=(6.2, 5.2), dpi=140)
    plt.imshow(
        np.log1p(H).T,
        origin="lower",
        extent=[-z_lim, z_lim, -w_lim, w_lim],
        aspect="auto",
    )
    plt.xlabel("z [pc]")
    plt.ylabel("w [km/s]")
    plt.title(title)
    plt.colorbar(label="log(1 + counts)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()



# ==========================================================
# Shared constants (sampling / evolution / fitting moved to Cell 3 loop)
# ==========================================================
h_true = 250.0
t_obs = 400.0  # Myr

# initial toy distribution (what likelihood assumes as f0)
z0_sigma = 10.0
w0_sigma = 20.0

sim = Widmark1DSimulation(h_pc_dm=h_true)
ana = Widmark1DAnalysis(sim)

# Run independently — needs Cell 2 (constants) to have been run
# h-only fits only, no NN. Mismatch ratios up to 100x.
import numpy as np
import matplotlib.pyplot as plt
import time as _time
from fitter import fit_h_pc_dm_nn_mle

N_test  = 500
ratios  = [1, 10, 100]                          # sigma_true / sigma_fit
z0_sf_list   = [z0_sigma / r for r in ratios]  # σ_z sweep  (truth=10)
w0_sf_list   = [w0_sigma / r for r in ratios]  # σ_w sweep  (truth=20)

# --- data (generated once with TRUE sigmas) ---
z0_d, w0_d = sim.sample_near_midplane(
    N=N_test, z_sigma_pc=z0_sigma, w_sigma_kms=w0_sigma, seed=0)
Zs, Ws, _ = sim.evolve_record(
    z0_d, w0_d, t_obs_myr=t_obs, n_steps=12000, n_snaps=60,
    seed=0, use_satellite=True, h_pc_dm=h_true)
z_obs = Zs[-1].astype(np.float64)
w_obs = Ws[-1].astype(np.float64)
print(f"Data ready: N={N_test}, h_true={h_true}, z0s={z0_sigma}, w0s={w0_sigma}\n")

def h_only(z0_sf, w0_sf):
    r = fit_h_pc_dm_nn_mle(
        sim, z_obs, w_obs, t_obs,
        z0_sigma=z0_sigma, w0_sigma=w0_sigma,
        z0_sigma_fit=z0_sf, w0_sigma_fit=w0_sf,
        h_init=250.0, use_nn_correction=False, use_satellite=True,
        n_epochs=200, n_sub=500, n_steps=5000, lr_h=5.0, seed=42,
        verbose=False, h_bounds=(5.0, 5000.0),
    )
    return r["h_pc_dm_fitted"]

t0 = _time.time()

print("=== SWEEP 1: σ_z only (σ_w at truth) ===")
h_z = []
for z0_sf, rat in zip(z0_sf_list, ratios):
    h = h_only(z0_sf, w0_sigma)
    h_z.append(h)
    print(f"  ratio={rat:3d}x  σ_z_fit={z0_sf:6.2f}  h_fit={h:.2f}  Δh={h-h_true:+.2f} pc")

print("\n=== SWEEP 2: σ_w only (σ_z at truth) ===")
h_w = []
for w0_sf, rat in zip(w0_sf_list, ratios):
    h = h_only(z0_sigma, w0_sf)
    h_w.append(h)
    print(f"  ratio={rat:3d}x  σ_w_fit={w0_sf:6.2f}  h_fit={h:.2f}  Δh={h-h_true:+.2f} pc")

print("\n=== SWEEP 3: both σ_z and σ_w (same ratio) ===")
h_both = []
for z0_sf, w0_sf, rat in zip(z0_sf_list, w0_sf_list, ratios):
    h = h_only(z0_sf, w0_sf)
    h_both.append(h)
    print(f"  ratio={rat:3d}x  σ_z_fit={z0_sf:6.2f}  σ_w_fit={w0_sf:6.2f}  h_fit={h:.2f}  Δh={h-h_true:+.2f} pc")

print(f"\nDone in {(_time.time()-t0)/60:.1f} min")

# --- summary table ---
h_z    = np.array(h_z)
h_w    = np.array(h_w)
h_both = np.array(h_both)
ratios_arr = np.array(ratios, dtype=float)

print(f"\n{'ratio':>6s}  {'Δh σ_z only':>14s}  {'Δh σ_w only':>14s}  {'Δh both':>12s}")
print("-" * 52)
for i, rat in enumerate(ratios):
    print(f"{rat:6d}x  {h_z[i]-h_true:+12.2f} pc  {h_w[i]-h_true:+12.2f} pc  {h_both[i]-h_true:+10.2f} pc")

# --- plot ---
fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
ax.axhline(h_true, color='k', ls='--', lw=1.5, label=f'True h = {h_true:.0f} pc')
ax.plot(ratios_arr, h_z,    'C0o-', ms=9, lw=2, label='σ_z mismatched only')
ax.plot(ratios_arr, h_w,    'C1s-', ms=9, lw=2, label='σ_w mismatched only')
ax.plot(ratios_arr, h_both, 'C2^-', ms=9, lw=2, label='both mismatched')
ax.set_xscale('log')
ax.set_xticks(ratios_arr)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Mismatch ratio  σ_true / σ_fit', fontsize=12)
ax.set_ylabel('Fitted h_dm  [pc]', fontsize=12)
ax.set_title('h_dm bias vs phase-space sigma mismatch\n(h-only, N=500, no NN)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('plots/mismatch_ratio_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# Phase-space visualisation
# Requires Cell 7 to have been run (h_z, h_w, h_both, z_obs, w_obs, z0_d, w0_d defined)
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse
from fitter import torch_back_integrate

def back_integrate_coords(h_fit, n_steps=3000):
    z0, w0 = torch_back_integrate(
        torch.tensor(z_obs, dtype=torch.float64),
        torch.tensor(w_obs, dtype=torch.float64),
        t_obs,
        torch.tensor(float(h_fit), dtype=torch.float64),
        sim,
        n_steps=n_steps, use_satellite=True,
    )
    return z0.detach().numpy(), w0.detach().numpy()

def gauss_ellipse(ax, sigma_z, sigma_w, nsigma=2, **kw):
    ax.add_patch(Ellipse((0, 0),
                         width=2*nsigma*sigma_z, height=2*nsigma*sigma_w,
                         facecolor='none', **kw))

# ── Plot 1: reference ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=130)

ax = axes[0]
ax.scatter(z0_d, w0_d, s=5, alpha=0.4, color='C0', label='Stars at t = 0')
gauss_ellipse(ax, z0_sigma, w0_sigma, nsigma=1, edgecolor='C0', lw=2,  label='1σ true')
gauss_ellipse(ax, z0_sigma, w0_sigma, nsigma=2, edgecolor='C0', lw=1,  label='2σ true', linestyle='--')
ax.set_xlabel('z  [pc]'); ax.set_ylabel('w  [km/s]')
ax.set_title('True initial conditions  (t = 0)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(z_obs, w_obs, s=5, alpha=0.4, color='C3', label='Stars at t = {:.0f} Myr'.format(t_obs))
ax.set_xlabel('z  [pc]'); ax.set_ylabel('w  [km/s]')
ax.set_title('Observed data  (t = {:.0f} Myr)'.format(t_obs))
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle('Reference phase space', fontsize=12)
plt.tight_layout()
plt.savefig('plots/mismatch_reference.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Plot 2: back-integrated t=0 for every fit ──────────────────────────────
sweep_labels = [r'$\sigma_z$ only', r'$\sigma_w$ only', 'Both']
h_fits_all   = [h_z, h_w, h_both]
z_sf_all     = [z0_sf_list,    [z0_sigma]*3,  z0_sf_list]
w_sf_all     = [[w0_sigma]*3,  w0_sf_list,    w0_sf_list]

# shared legend handles
leg_handles = [
    mlines.Line2D([], [], color='grey', marker='o', ms=5, ls='none', alpha=0.5,
                  label='True data  (t = 0)'),
    mlines.Line2D([], [], color='C1',   marker='o', ms=5, ls='none', alpha=0.7,
                  label='Back-integrated to t = 0  (fitted h_dm)'),
    mlines.Line2D([], [], color='C1',   lw=2.0, ls='-',
                  label='2σ ellipse assumed by fit'),
    mlines.Line2D([], [], color='grey', lw=1.5, ls='--',
                  label='2σ true distribution'),
]

fig, axes = plt.subplots(3, 3, figsize=(14, 12), dpi=120)
fig.suptitle('Back-integrated t = 0 phase space per fit', fontsize=13, y=1.01)

row_labels = ['ratio = 1x', 'ratio = 10x', 'ratio = 100x']

for col, (slabel, h_fits, z_sfs, w_sfs) in enumerate(
        zip(sweep_labels, h_fits_all, z_sf_all, w_sf_all)):
    axes[0, col].set_title(slabel, fontsize=10, fontweight='bold', pad=8)
    for row, (rat, h_fit, z_sf, w_sf) in enumerate(
            zip(ratios, h_fits, z_sfs, w_sfs)):
        ax = axes[row, col]

        # true t=0 data (grey)
        ax.scatter(z0_d, w0_d, s=3, alpha=0.25, color='grey', zorder=1)
        gauss_ellipse(ax, z0_sigma, w0_sigma, nsigma=2,
                      edgecolor='grey', lw=1.2, linestyle='--', zorder=3)

        # back-integrated (orange)
        z0_bi, w0_bi = back_integrate_coords(h_fit)
        ax.scatter(z0_bi, w0_bi, s=4, alpha=0.45, color='C1', zorder=2)
        gauss_ellipse(ax, z_sf, w_sf, nsigma=2,
                      edgecolor='C1', lw=1.8, zorder=4)

        dh = h_fit - h_true
        ax.set_title('h_fit = {:.0f} pc   Δh = {:+.0f} pc'.format(h_fit, dh), fontsize=7.5)
        ax.set_xlabel('z  [pc]', fontsize=7.5)
        ax.set_ylabel('w  [km/s]', fontsize=7.5)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2)

        if col == 0:
            ax.annotate(row_labels[row], xy=(-0.32, 0.5), xycoords='axes fraction',
                        fontsize=8, fontweight='bold', rotation=90, va='center')

# shared legend below the grid
fig.legend(handles=leg_handles, loc='lower center', ncol=4,
           fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

plt.tight_layout()
plt.savefig('plots/mismatch_backintegrated.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: plots/mismatch_reference.png  |  plots/mismatch_backintegrated.png")
