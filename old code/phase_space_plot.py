import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def plot_phase_space_before_after(sim, z0_sigma, w0_sigma, t_obs, h_true,
                                   N_spiral=10000, use_satellite=True, seed=0,
                                   equilibrium=True, N_resid=200_000, n_steps=8000,
                                   Sigma_sat=2.5, boost_kms=0.0, smooth_sigma=2.5,
                                   save_path='plots/phase_space_before_after.png'):
    """Phase-space before/after PLUS a mean-subtracted residual panel.

    Panels: (0) t=0 initial, (1) t=t_obs snapshot, (2) t=t_obs relative-density
    residual (n - n_smooth)/n_smooth -- the *mean-subtracted* view that isolates
    the spiral. As in Widmark et al. (2021) Fig. A.1 the spiral is only clearly
    visible after subtracting the smooth background.

    equilibrium=True (default) seeds the dense, stationary EQUILIBRIUM DF (the
    Widmark background), so the residual shows the spiral on a full sea -- the
    faithful comparison. equilibrium=False uses the old thin-Gaussian blob (the
    inference IC), which is non-equilibrium and winds up the whole population.

    PERTURBATION STRENGTH sets how distorted the CENTER looks; evolution TIME
    sets how many times the spiral winds. To match Widmark's smooth, dense core
    the perturbation must be gentle (~10-20% residual, not ~50%):
      * Sigma_sat  -- satellite kick amplitude (overrides sim.Sigma_sat via a
                      local copy; default 2.5 keeps the centre smooth; the sim
                      default of 6 visibly churns it). Set None to use sim's own.
      * boost_kms  -- if >0, add a uniform vertical velocity kick dw (the classic
                      clean phase-spiral generator); pair with use_satellite=False
                      for the most symmetric, undistorted centre.

    The scatter panels use an N_spiral subset; the residual histogram uses the
    full N_resid sample (needs many particles to be clean).
    """
    rng = np.random.default_rng(seed)
    N_all = max(N_spiral, N_resid)

    sim_use = sim
    if Sigma_sat is not None:
        sim_use = copy.deepcopy(sim)
        sim_use.Sigma_sat = Sigma_sat

    if equilibrium:
        z0_all, w0_all, _ = sim_use.sample_initial_particles(
            N_total=N_all, seed=seed, h_pc_dm=h_true)
        init_xlim, init_label = 800.0, 'equilibrium DF'
    else:
        z0_all, w0_all = sim_use.sample_near_midplane(
            N=N_all, z_sigma_pc=z0_sigma, w_sigma_kms=w0_sigma, seed=seed)
        init_xlim, init_label = 250.0, 'Gaussian blob'

    if boost_kms:
        w0_all = w0_all + boost_kms   # uniform vertical kick -> clean spiral seed

    Zs_sp, Ws_sp, Ts_sp = sim_use.evolve_record(
        z0_all, w0_all, t_obs_myr=t_obs, n_steps=n_steps, n_snaps=60,
        seed=seed, use_satellite=use_satellite, h_pc_dm=h_true)
    z_fin_all, w_fin_all = Zs_sp[-1].astype(float), Ws_sp[-1].astype(float)

    # ---- number-density maps (as Widmark/Gaia show them) + mean-subtracted residual ----
    # All three panels are STAR-COUNT maps, not scatter coloured by phase angle
    # (angle-colouring paints a spiral onto even a smooth disk). In the raw count
    # maps the spiral is faint -- it only stands out after mean subtraction.
    z_lim, w_lim, nb = 900.0, 70.0, 130
    z_bins = np.linspace(-z_lim, z_lim, nb + 1)
    w_bins = np.linspace(-w_lim, w_lim, nb + 1)
    ext = [-z_lim, z_lim, -w_lim, w_lim]

    d0, _, _ = np.histogram2d(z0_all,    w0_all,    bins=[z_bins, w_bins])
    d1, _, _ = np.histogram2d(z_fin_all, w_fin_all, bins=[z_bins, w_bins])

    # residual on a COARSER grid: more stars/bin -> the gentle (~10%) spiral beats
    # Poisson noise without having to strengthen the perturbation.
    nb_r = 80
    zr = np.linspace(-z_lim, z_lim, nb_r + 1)
    wr = np.linspace(-w_lim, w_lim, nb_r + 1)
    dr, _, _ = np.histogram2d(z_fin_all, w_fin_all, bins=[zr, wr])
    bg = gaussian_filter(dr, sigma=smooth_sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(bg > 0.12 * bg.max(), (dr - bg) / bg, np.nan)
    vmax = np.nanpercentile(np.abs(R), 95)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), dpi=150, facecolor='#0a0a0a')
    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors='#cccccc', labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    im0 = axes[0].imshow(d0.T, origin='lower', extent=ext, aspect='auto', cmap='magma',
                         vmax=np.percentile(d0[d0 > 0], 99.0))
    axes[0].set_title(f't = 0 Myr   number density  ({init_label})', color='white', fontsize=12, pad=8)
    axes[0].set_xlim(-init_xlim, init_xlim); axes[0].set_ylim(-55, 55)
    cb0 = plt.colorbar(im0, ax=axes[0], pad=0.02); cb0.ax.tick_params(colors='#dddddd')
    cb0.set_label('star counts', color='#dddddd')

    im1 = axes[1].imshow(d1.T, origin='lower', extent=ext, aspect='auto', cmap='magma',
                         vmax=np.percentile(d1[d1 > 0], 99.0))
    axes[1].set_title(f't = {t_obs:.0f} Myr   number density  (raw)', color='white', fontsize=12, pad=8)
    axes[1].set_xlim(-700, 700); axes[1].set_ylim(-55, 55)
    cb1 = plt.colorbar(im1, ax=axes[1], pad=0.02); cb1.ax.tick_params(colors='#dddddd')
    cb1.set_label('star counts', color='#dddddd')

    im2 = axes[2].imshow(R.T, origin='lower', extent=ext, aspect='auto',
                         cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title(f't = {t_obs:.0f} Myr   mean-subtracted  (n-$\\bar{{n}}$)/$\\bar{{n}}$',
                      color='white', fontsize=12, pad=8)
    axes[2].set_xlim(-700, 700); axes[2].set_ylim(-55, 55)
    cb2 = plt.colorbar(im2, ax=axes[2], pad=0.02); cb2.ax.tick_params(colors='#dddddd')
    cb2.set_label('relative overdensity', color='#dddddd')

    for ax in axes:
        ax.set_xlabel('z  [pc]',   color='#dddddd', fontsize=11)
        ax.set_ylabel('w  [km/s]', color='#dddddd', fontsize=11)
        ax.grid(alpha=0.12, color='white', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved: {os.path.abspath(save_path)}")

    plt.show()
    return fig, (Zs_sp, Ws_sp, Ts_sp)


def generate_widmark_spiral(sim, t_obs=400.0, h_true=250.0, N=300_000,
                            Sigma_sat=None, boost_kms=0.0, use_satellite=True,
                            seed=0, n_steps=8000):
    """Generate a Widmark-style vertical phase-space spiral.

    Faithful to Widmark et al. (2021), Fig. A.1: seed the *equilibrium* DF (a
    dense, stationary, centrally-peaked background — all three disk components
    sampled from exp(-Phi/sigma_w^2)), then apply a perturbation (satellite kick
    and/or a uniform boost). The anharmonicity of Phi phase-mixes that coherent
    perturbation into a spiral (a harmonic Phi would only rotate it rigidly).

    Returns (z0, w0, z_fin, w_fin): the *equilibrium* IC (t=0, pre-kick) and the
    evolved snapshot (t=t_obs).
    """
    sim_use = sim
    if Sigma_sat is not None:
        sim_use = copy.deepcopy(sim)
        sim_use.Sigma_sat = Sigma_sat

    z0, w0, _ = sim_use.sample_initial_particles(N_total=N, seed=seed, h_pc_dm=h_true)
    w_start = w0 + boost_kms if boost_kms else w0
    Zs, Ws, _ = sim_use.evolve_record(
        z0, w_start, t_obs_myr=t_obs, n_steps=n_steps, n_snaps=2,
        seed=seed, use_satellite=use_satellite, h_pc_dm=h_true)
    return z0, w0, Zs[-1].astype(float), Ws[-1].astype(float)


def _total_phi(sim, z, h_dm):
    """Total vertical potential Phi(z) = baryons (3 sech^2 discs) + DM halo."""
    z = np.asarray(z, dtype=float)
    return (sim.phi_component_sech2(z, sim.rho0_thin,  sim.h_thin_pc)
            + sim.phi_component_sech2(z, sim.rho0_thick, sim.h_thick_pc)
            + sim.phi_component_sech2(z, sim.rho0_gas,   sim.h_gas_pc)
            + sim.phi_component_sech2(z, sim.rho_DM,     float(h_dm)))


def relative_density_residual(z, w, sim, h_dm, z_bins, w_bins,
                              n_E=45, thresh=0.05, display_smooth=1.0):
    """Residual after subtracting the STEADY-STATE background f = f(E).

    A steady DF depends only on integrals of motion, so it is constant along
    orbits: the steady background is the phase-average of the counts at fixed
    energy E = 1/2 w^2 + Phi(z). Subtracting it (per equal-population energy
    shell) leaves ONLY the phase-dependent, non-stationary part -- the spiral.
    For an equilibrium snapshot this residual is ~0 (noise); a spiral gives
    coherent arms. Unlike a Gaussian-blur background this has NO core/ring
    unsharp-mask artifact on the peaked equilibrium, so t=0 genuinely looks flat.
    """
    d, _, _ = np.histogram2d(z, w, bins=[z_bins, w_bins])
    zc = 0.5 * (z_bins[:-1] + z_bins[1:])
    wc = 0.5 * (w_bins[:-1] + w_bins[1:])
    ZZ, WW = np.meshgrid(zc, wc, indexing='ij')
    E = 0.5 * WW**2 + _total_phi(sim, ZZ, h_dm)

    # background B(E) = mean counts within equal-population energy shells
    Eflat, dflat = E.ravel(), d.ravel()
    edges = np.quantile(Eflat, np.linspace(0.0, 1.0, n_E + 1))
    edges[-1] += 1e-6
    idx = np.clip(np.digitize(Eflat, edges) - 1, 0, n_E - 1)
    shell_mean = np.array([dflat[idx == k].mean() if np.any(idx == k) else 0.0
                           for k in range(n_E)])
    B = shell_mean[idx].reshape(d.shape)

    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(B > thresh * B.max(), (d - B) / B, np.nan)

    if display_smooth:  # NaN-aware smoothing: suppresses per-bin noise, keeps arms
        filled = np.where(np.isnan(R), 0.0, R)
        wt = (~np.isnan(R)).astype(float)
        num = gaussian_filter(filled, display_smooth)
        den = gaussian_filter(wt, display_smooth)
        Rs = np.where(den > 1e-6, num / den, np.nan)
        R = np.where(np.isnan(R), np.nan, Rs)
    return R


def plot_mean_subtraction_two_times(sim, t_obs=400.0, h_true=250.0, N=300_000,
                                    Sigma_sat=2.5, boost_kms=0.0, use_satellite=True,
                                    seed=0, n_steps=8000, z_lim=700.0, w_lim=55.0,
                                    nbins=80, smooth_sigma=2.5,
                                    save_path='plots/mean_subtraction_t0_t400.png'):
    """Mean-subtracted phase space at t=0 vs t=t_obs -> visual proof that df/dt != 0.

    At t=0 the DF is the stationary equilibrium: the relative-density residual is
    featureless (noise about zero). At t=t_obs a spiral has wound up: the residual
    is a clear coherent pattern. BOTH panels share one colour scale (set by the
    t_obs residual) so the t=0 panel genuinely looks flat -- the change between
    the two panels is the non-stationarity, i.e. df/dt != 0.
    """
    z0, w0, z_fin, w_fin = generate_widmark_spiral(
        sim, t_obs=t_obs, h_true=h_true, N=N, Sigma_sat=Sigma_sat,
        boost_kms=boost_kms, use_satellite=use_satellite, seed=seed, n_steps=n_steps)

    z_bins = np.linspace(-z_lim, z_lim, nbins + 1)
    w_bins = np.linspace(-w_lim, w_lim, nbins + 1)
    extent = [-z_lim, z_lim, -w_lim, w_lim]

    R0 = relative_density_residual(z0,    w0,    sim, h_true, z_bins, w_bins)
    R1 = relative_density_residual(z_fin, w_fin, sim, h_true, z_bins, w_bins)
    vmax = np.nanpercentile(np.abs(R1), 95)   # shared scale, set by the t_obs spiral

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=150, facecolor='#0a0a0a')
    for ax in axes:
        ax.set_facecolor('black')
        ax.tick_params(colors='#cccccc', labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    titles = [f't = 0 Myr   (equilibrium — steady)',
              f't = {t_obs:.0f} Myr   (spiral — $\\partial f/\\partial t \\neq 0$)']
    for ax, R, title in zip(axes, (R0, R1), titles):
        im = ax.imshow(R.T, origin='lower', extent=extent, aspect='auto',
                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(title, color='white', fontsize=12, pad=8)
        ax.set_xlabel('z  [pc]',   color='#dddddd', fontsize=11)
        ax.set_ylabel('w  [km/s]', color='#dddddd', fontsize=11)
        ax.grid(alpha=0.12, color='white', linewidth=0.5)
        cb = plt.colorbar(im, ax=ax, pad=0.02); cb.ax.tick_params(colors='#dddddd')
        cb.set_label(r'relative overdensity  $(n-\bar{n})/\bar{n}$', color='#dddddd')

    fig.suptitle('Mean-subtracted phase space: flat at t=0, wound spiral at '
                 f'{t_obs:.0f} Myr', color='white', fontsize=13)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Saved: {os.path.abspath(save_path)}")

    plt.show()
    return fig, (R0, R1)


def plot_widmark_spiral(sim, t_obs=400.0, h_true=250.0, N=300_000,
                        Sigma_sat=None, seed=0, n_steps=8000,
                        z_lim=900.0, w_lim=70.0, nbins=160, smooth_sigma=3.0,
                        save_path='plots/spiral_widmark.png'):
    """Density + mean-subtracted view of the Widmark phase-space spiral.

    Three panels: (1) t=0 equilibrium number density, (2) t=t_obs number
    density (dense center + wound spiral), (3) relative-density residual
    (n - n_smooth)/n_smooth, which isolates the spiral -- as in Widmark the
    spiral is only clearly visible after subtracting the smooth background.
    """
    z0, w0, z_fin, w_fin = generate_widmark_spiral(
        sim, t_obs=t_obs, h_true=h_true, N=N, Sigma_sat=Sigma_sat,
        seed=seed, n_steps=n_steps)

    z_bins = np.linspace(-z_lim, z_lim, nbins + 1)
    w_bins = np.linspace(-w_lim, w_lim, nbins + 1)
    extent = [-z_lim, z_lim, -w_lim, w_lim]

    def density(z, w):
        d, _, _ = np.histogram2d(z, w, bins=[z_bins, w_bins])
        return d

    def residual(d):
        bg = gaussian_filter(d, sigma=smooth_sigma)
        with np.errstate(invalid="ignore", divide="ignore"):
            R = np.where(bg > 0.05 * bg.max(), (d - bg) / bg, np.nan)
        return R

    d0 = density(z0, w0)
    d1 = density(z_fin, w_fin)
    R1 = residual(d1)
    vmax = np.nanpercentile(np.abs(R1), 98)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=150)
    axes[0].imshow(np.log1p(d0).T, origin="lower", extent=extent, aspect="auto", cmap="magma")
    axes[0].set_title(f"t = 0   equilibrium  (N = {N:,})")
    axes[1].imshow(np.log1p(d1).T, origin="lower", extent=extent, aspect="auto", cmap="magma")
    axes[1].set_title(f"t = {t_obs:.0f} Myr   number density")
    im = axes[2].imshow(R1.T, origin="lower", extent=extent, aspect="auto",
                        cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[2].set_title("relative density  (n - n_smooth)/n_smooth")
    plt.colorbar(im, ax=axes[2], pad=0.02, label="relative overdensity")
    for ax in axes:
        ax.set_xlabel("z [pc]"); ax.set_ylabel("w [km/s]")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {os.path.abspath(save_path)}")

    plt.show()
    return fig, (z0, w0, z_fin, w_fin)
