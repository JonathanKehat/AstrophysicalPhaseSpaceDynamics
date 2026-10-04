import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.mixture import GaussianMixture


@dataclass
class Widmark1DSimulation:
    rho0_thin: float = 0.04
    rho0_thick: float = 0.01
    rho0_gas: float = 0.05

    rho_DM: float = 0.01
    h_pc_dm: float = 250.0

    sigma_w_thin: float = 16.0
    sigma_w_thick: float = 24.0
    sigma_w_gas: float = 8.0

    h_thin_pc: float = 200.0
    h_thick_pc: float = 400.0
    h_gas_pc: float = 800.0

    Sigma_sat: float = 6.0
    sigma_t_myr: float = 80.0
    H_sat_pc: float = 50.0
    z0_sat_pc: float = 600.0
    w_sat_kms: float = -50.0

    z_max_pc: float = 1000.0
    n_z: int = 2001

    Z_lim_mask_pc: float = 700.0
    W_lim_mask_kms: float = 40.0

    G: float = 4.3009e-3

    def __post_init__(self):
        self.z_grid = np.linspace(-self.z_max_pc, self.z_max_pc, self.n_z)
        self.MYR_TO_TUNIT = 1.022712
        self._dphi_dz_baryons_grid = (
            self.dphi_dz_component_sech2(self.z_grid, self.rho0_thin,  self.h_thin_pc)
            + self.dphi_dz_component_sech2(self.z_grid, self.rho0_thick, self.h_thick_pc)
            + self.dphi_dz_component_sech2(self.z_grid, self.rho0_gas,   self.h_gas_pc)
        )

    def phi_component_sech2(self, z, rho0, h_pc):
        z = np.asarray(z, dtype=float)
        return 4.0 * math.pi * self.G * rho0 * h_pc**2 * np.log(np.cosh(z / h_pc))

    def dphi_dz_component_sech2(self, z, rho0, h_pc):
        z = np.asarray(z, dtype=float)
        return 4.0 * math.pi * self.G * rho0 * h_pc * np.tanh(z / h_pc)

    def dphi_dz_baryons(self, z):
        # Analytic sum of the three tanh components.  The old np.interp over
        # z_grid clamped the force beyond |z| = z_max_pc, which disagreed with
        # the analytic force used by fitter.torch_back_integrate and broke the
        # forward/backward round trip for stars reaching |z| > 1000 pc.
        z = np.asarray(z, dtype=float)
        return (self.dphi_dz_component_sech2(z, self.rho0_thin,  self.h_thin_pc)
              + self.dphi_dz_component_sech2(z, self.rho0_thick, self.h_thick_pc)
              + self.dphi_dz_component_sech2(z, self.rho0_gas,   self.h_gas_pc))

    def dphi_dz_dm(self, z, h_pc_dm=None):
        if h_pc_dm is None:
            h_pc_dm = self.h_pc_dm
        return self.dphi_dz_component_sech2(z, self.rho_DM, float(h_pc_dm))

    def dphi_dz(self, z, h_pc_dm=None):
        return self.dphi_dz_baryons(z) + self.dphi_dz_dm(z, h_pc_dm=h_pc_dm)

    def sample_near_midplane(self, N=10_000, z_mean_pc=0.0, z_sigma_pc=20.0,
                             w_mean_kms=0.0, w_sigma_kms=40.0, seed=None):
        rng = np.random.default_rng(seed)
        z0 = rng.normal(z_mean_pc, z_sigma_pc, size=N)
        w0 = rng.normal(w_mean_kms, w_sigma_kms, size=N)
        return z0, w0

    def _sample_component_equilibrium(self, N, sigma_w, rng, h_pc_dm=None):
        phi_vals = (
            self.phi_component_sech2(self.z_grid, self.rho0_thin,  self.h_thin_pc)
            + self.phi_component_sech2(self.z_grid, self.rho0_thick, self.h_thick_pc)
            + self.phi_component_sech2(self.z_grid, self.rho0_gas,   self.h_gas_pc)
            + self.phi_component_sech2(self.z_grid, self.rho_DM,
                                       float(self.h_pc_dm if h_pc_dm is None else h_pc_dm))
        )
        rho_unnorm = np.exp(-phi_vals / sigma_w**2)
        p_z = rho_unnorm / rho_unnorm.sum()
        z_samples = rng.choice(self.z_grid, size=N, p=p_z)
        w_samples = rng.normal(loc=0.0, scale=sigma_w, size=N)
        return z_samples, w_samples

    def sample_initial_particles(self, N_total, seed=None, h_pc_dm=None):
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

    def Kz_sat_common(self, t_internal, z):
        z = np.asarray(z, dtype=float)
        sigma_t_internal = self.sigma_t_myr * self.MYR_TO_TUNIT
        tau = t_internal / sigma_t_internal
        z_sat = self.z0_sat_pc + self.w_sat_kms * t_internal
        return (-4.0 * math.pi * self.G * self.Sigma_sat
                * np.exp(-0.5 * tau**2)
                * np.tanh((z - z_sat) / self.H_sat_pc))

    def evolve_record(self, z0, w0, t_obs_myr=400.0, n_steps=20_000, n_snaps=80,
                      seed=None, pulse_cut_sigmas=2.5, use_satellite=True, h_pc_dm=None,
                      h_fourier_a=None, h_fourier_b=None):
        rng = np.random.default_rng(seed)
        z = np.asarray(z0, dtype=float).copy()
        w = np.asarray(w0, dtype=float).copy()
        s_i = np.ones_like(z)

        sigma_t_internal = self.sigma_t_myr * self.MYR_TO_TUNIT
        t_end = t_obs_myr * self.MYR_TO_TUNIT
        dt = t_end / float(n_steps)
        t_cut = pulse_cut_sigmas * sigma_t_internal

        # Optional time-dependent DM scale height:
        #   h(t) = h0 + sum_k [a_k cos(2*pi*k*t/t_end) + b_k sin(2*pi*k*t/t_end)]
        # with t/t_end in [0,1] (0 = initial, 1 = observation), matching the
        # convention in fitter.torch_back_integrate / eval_h_of_t_fourier.
        base_h0 = self.h_pc_dm if h_pc_dm is None else h_pc_dm
        time_dep_h = h_fourier_a is not None

        def h_of_t(t_internal):
            if not time_dep_h:
                return base_h0
            tn = t_internal / t_end
            h = base_h0
            for k in range(len(h_fourier_a)):
                ang = 2.0 * math.pi * (k + 1) * tn
                h = h + h_fourier_a[k] * math.cos(ang) + h_fourier_b[k] * math.sin(ang)
            return max(h, 20.0)

        snap_idx = np.unique(np.linspace(0, n_steps, n_snaps, dtype=int))
        Zs = np.empty((snap_idx.size, z.size), dtype=np.float32)
        Ws = np.empty((snap_idx.size, w.size), dtype=np.float32)
        Ts_myr = np.empty(snap_idx.size, dtype=np.float32)

        def accel(t_internal, z_now):
            if time_dep_h:
                a_bg = -(self.dphi_dz_baryons(z_now)
                         + self.dphi_dz_dm(z_now, h_pc_dm=h_of_t(t_internal)))
            else:
                a_bg = -self.dphi_dz(z_now, h_pc_dm=h_pc_dm)
            if (not use_satellite) or (t_internal > t_cut):
                return a_bg
            return a_bg + s_i * self.Kz_sat_common(t_internal, z_now)

        t = 0.0
        a = accel(t, z)
        w += 0.5 * dt * a

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

    def back_integrate(self, z_obs, w_obs, t_obs_myr, n_steps=10_000,
                       use_satellite=False, h_pc_dm=None, pulse_cut_sigmas=2.5):
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

    def _log_f0_gaussian(self, z0, w0, z_sigma, w_sigma):
        z0 = np.asarray(z0, float)
        w0 = np.asarray(w0, float)
        sz = float(z_sigma); sw = float(w_sigma)
        log_norm = -np.log(2.0 * math.pi * sz * sw)
        return log_norm - 0.5 * (z0 / sz) ** 2 - 0.5 * (w0 / sw) ** 2

    def log_likelihood_h_pc_dm(self, z_obs, w_obs, t_obs_myr, h_pc_dm,
                               z0_sigma, w0_sigma, n_steps=10_000,
                               use_satellite_in_likelihood=False):
        z0_hat, w0_hat = self.back_integrate(
            z_obs, w_obs, t_obs_myr, n_steps=n_steps,
            use_satellite=use_satellite_in_likelihood, h_pc_dm=h_pc_dm)
        return float(np.sum(self._log_f0_gaussian(z0_hat, w0_hat, z0_sigma, w0_sigma)))


@dataclass
class Widmark1DAnalysis:
    sim: Widmark1DSimulation

    def mask_M(self, Z, W):
        Z = np.asarray(Z, dtype=float); W = np.asarray(W, dtype=float)
        r2 = (Z / self.sim.Z_lim_mask_pc) ** 2 + (W / self.sim.W_lim_mask_kms) ** 2
        x = 10.0 * (r2 - 1.0)
        return 1.0 / (1.0 + np.exp(-x))

    def data_bulk_spiral_histograms(self, z_data, w_data, z_bins, w_bins,
                                    n_gmm=6, random_state=0, residual_mode="fractional"):
        d, _, _ = np.histogram2d(z_data, w_data, bins=[z_bins, w_bins])
        X = np.vstack([z_data, w_data]).T
        gmm = GaussianMixture(n_components=n_gmm, covariance_type="full", random_state=random_state)
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
        B = bulk_pdf * (data_sum / (pdf_sum + eps))
        delta = d - B

        if residual_mode == "absolute":
            R = delta
        elif residual_mode == "fractional":
            R = delta / (B + eps)
        else:
            raise ValueError("residual_mode must be 'absolute' or 'fractional'.")
        R = M * R
        return d, B, R, M, ZZ, WW


def fit_h_pc_dm_mle_fast(sim, z_obs, w_obs, t_obs_myr, z0_sigma, w0_sigma,
                         h_bounds=(50.0, 800.0),
                         n_sub_1=3000, n_steps_1=2000, n_grid_1=31,
                         n_sub_2=12000, n_steps_2=7000, n_grid_2=41,
                         use_satellite_in_likelihood=False, seed=0):
    rng = np.random.default_rng(seed)
    N = z_obs.size
    idx1 = rng.choice(N, size=min(n_sub_1, N), replace=False)
    idx2 = rng.choice(N, size=min(n_sub_2, N), replace=False)

    hs1 = np.linspace(h_bounds[0], h_bounds[1], n_grid_1)
    ll1 = np.array([
        sim.log_likelihood_h_pc_dm(z_obs[idx1], w_obs[idx1], t_obs_myr, h,
                                    z0_sigma=z0_sigma, w0_sigma=w0_sigma,
                                    n_steps=n_steps_1,
                                    use_satellite_in_likelihood=use_satellite_in_likelihood)
        for h in hs1
    ])
    h1 = float(hs1[np.argmax(ll1)])

    span = 0.25 * (h_bounds[1] - h_bounds[0])
    lo = max(h_bounds[0], h1 - 0.15 * span)
    hi = min(h_bounds[1], h1 + 0.15 * span)

    hs2 = np.linspace(lo, hi, n_grid_2)
    ll2 = np.array([
        sim.log_likelihood_h_pc_dm(z_obs[idx2], w_obs[idx2], t_obs_myr, h,
                                    z0_sigma=z0_sigma, w0_sigma=w0_sigma,
                                    n_steps=n_steps_2,
                                    use_satellite_in_likelihood=use_satellite_in_likelihood)
        for h in hs2
    ])
    h2 = float(hs2[np.argmax(ll2)])
    return (h2, (hs1, ll1), (hs2, ll2))


def plot_phase_space_hist(z, w, title, z_lim=800.0, w_lim=60.0, nbins=240):
    z_bins = np.linspace(-z_lim, z_lim, nbins + 1)
    w_bins = np.linspace(-w_lim, w_lim, nbins + 1)
    H, _, _ = np.histogram2d(z, w, bins=[z_bins, w_bins])

    plt.figure(figsize=(6.2, 5.2), dpi=140)
    plt.imshow(np.log1p(H).T, origin="lower",
               extent=[-z_lim, z_lim, -w_lim, w_lim], aspect="auto")
    plt.xlabel("z [pc]"); plt.ylabel("w [km/s]"); plt.title(title)
    plt.colorbar(label="log(1 + counts)")
    plt.grid(alpha=0.2); plt.tight_layout(); plt.show()
