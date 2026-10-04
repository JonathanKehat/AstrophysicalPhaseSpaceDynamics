import numpy as np
# ==========================================================
# Neural-Network Potential Correction + Differentiable MLE
# Joint optimisation of h_pc_dm and a small NN that learns a
# bounded additive potential correction V_nn(z).
# The force correction is derived via autograd: dV_nn/dz.
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as _math
import time as _time

# ---------- 1a. Potential Correction MLP (time-independent) ----------
class PotentialCorrectionMLP(nn.Module):
    """2-hidden-layer MLP (8 neurons each, Tanh activations).
    Last layer zero-initialised so the network starts as V_nn = 0.
    Output is a normalised potential shape (bounded by tanh ~ [-1,1]);
    scaled externally by the learnable amplitude A_nn."""
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_norm):
        """z_norm: (..., 1) normalised to ~[-1, 1]."""
        return self.net(z_norm)


# ---------- 1b. Time-Dependent Potential Correction MLP ----------
class PotentialCorrectionMLPTimeDep(nn.Module):
    """2-input MLP: (z_norm, t_norm) -> scalar potential correction.
    Same hidden architecture as PotentialCorrectionMLP (8 neurons, Tanh).
    Last layer zero-initialised so the network starts as V_nn = 0."""
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, zt_norm):
        """zt_norm: (..., 2) where col0 = z/z_max, col1 = t/t_end."""
        return self.net(zt_norm)


# ---------- 1d. Physics-informed density MLP (Test 2d) ----------
class PhysInformedDensityMLP(nn.Module):
    """Outputs a non-negative density rho_nn(z) via softplus.
    Symmetric in z (input is |z|/z_max -> rho_nn even in z).
    Last layer zero-initialised so initial raw output is 0; the resulting
    rho_nn(z) = softplus(0) = ln(2) is a uniform constant. We then SCALE by
    a learnable amplitude A_nn (initialised so V_nn ~ 0.1 * cap)."""
    def __init__(self, hidden=32, n_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_norm):
        """z_norm: tensor of shape (...,). Returns rho_nn >= 0 of same shape."""
        abs_z = z_norm.abs().unsqueeze(-1)
        raw = self.net(abs_z).squeeze(-1)
        return F.softplus(raw)


def compute_physinformed_potential(nn_model, A_nn, z_grid, sim):
    """Given rho_nn(z) = A_nn * nn_model(z/z_max), integrate Poisson in 1D:
        d^2 V_nn / dz^2 = 4 pi G rho_nn(z)
    with BCs V_nn(0) = 0, V_nn'(0) = 0 (symmetric rho).

    z_grid must be a symmetric grid with ODD length spanning [-z_max, z_max].
    Returns (V_grid, Vp_grid), both of shape (N,) on the same grid.
    V_grid is even in z (symmetric); Vp_grid is odd (antisymmetric).
    Differentiable w.r.t. NN weights and A_nn.
    """
    assert z_grid.shape[0] % 2 == 1, "z_grid must have odd length"
    four_pi_G = 4.0 * _math.pi * sim.G
    z_norm = z_grid / sim.z_max_pc
    rho = A_nn * nn_model(z_norm)
    # Force exact symmetry (rho should already be symmetric via |z| input)
    rho = 0.5 * (rho + rho.flip(0))

    N    = z_grid.shape[0]
    mid  = N // 2
    dz   = (z_grid[1] - z_grid[0])
    zero = torch.zeros(1, dtype=rho.dtype, device=rho.device)

    # Trapezoidal cumulative integral on the positive-z half (including z=0)
    rho_pos    = rho[mid:]                                # (mid+1,)
    trap_rho   = 0.5 * (rho_pos[1:] + rho_pos[:-1]) * dz  # (mid,)
    Vp_pos     = four_pi_G * torch.cat([zero, torch.cumsum(trap_rho, 0)])  # (mid+1,)

    trap_Vp    = 0.5 * (Vp_pos[1:] + Vp_pos[:-1]) * dz    # (mid,)
    V_pos      = torch.cat([zero, torch.cumsum(trap_Vp, 0)])  # (mid+1,)

    # Mirror to negative-z half. V is even, V' is odd.
    V_grid  = torch.cat([V_pos[1:].flip(0),  V_pos])      # (N,)
    Vp_grid = torch.cat([-Vp_pos[1:].flip(0), Vp_pos])    # (N,)
    return V_grid, Vp_grid


def _interp_linear_1d(query, x_grid, y_grid):
    """Differentiable linear interpolation y(query) given y_grid on x_grid.
    Clamps query to the grid range; x_grid must be sorted ascending."""
    q     = query.clamp(x_grid[0], x_grid[-1])
    i_r   = torch.searchsorted(x_grid, q).clamp(1, x_grid.shape[0] - 1)
    i_l   = i_r - 1
    x_l   = x_grid[i_l]
    x_r   = x_grid[i_r]
    y_l   = y_grid[i_l]
    y_r   = y_grid[i_r]
    w     = (q - x_l) / (x_r - x_l + 1e-30)
    return y_l + w * (y_r - y_l)


# ---------- 1e. Physics-informed time-dependent density MLP (Test 2f) -------
class PhysInformedDensityMLPTimeDep(nn.Module):
    """Outputs a non-negative density rho_nn(z, t) via softplus.
    Symmetric in z (input is |z|/z_max -> rho_nn even in z), free in t.
    Last layer zero-initialised -> rho_nn = ln(2) (constant) at init."""
    def __init__(self, hidden=32, n_layers=3):
        super().__init__()
        layers = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_norm, t_norm):
        """z_norm, t_norm: 1D tensors of same length. Returns rho >= 0 (same length)."""
        abs_z = z_norm.abs()
        inp   = torch.stack([abs_z, t_norm], dim=-1)
        raw   = self.net(inp).squeeze(-1)
        return F.softplus(raw)


# ---------- 1c. Higher-capacity MLP (3 hidden layers) ----------
class PotentialCorrectionMLPBig(nn.Module):
    """3-hidden-layer MLP (default 32 neurons each, Tanh activations).
    Last layer zero-initialised so V_nn = 0 at start. Designed for stronger
    flexibility -- can absorb biases that the 8-neuron version cannot."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_norm):
        return self.net(z_norm)


# ---------- 2. Differentiable leapfrog back-integrator ----------
def torch_back_integrate(
    z_obs, w_obs, t_obs_myr, h_pc_dm,
    sim, n_steps=5000,
    use_satellite=False, pulse_cut_sigmas=2.5,
    nn_model=None, A_nn=None, use_nn_correction=False,
    time_dependent_nn=False,
    h_fourier_a=None, h_fourier_b=None,
    sat_amp=None,
    use_box_satellite=False,
    box_t_on=None, box_t_off=None, box_tau=None, box_z_sat=None,
    use_physinformed_nn=False,
    physinformed_Vp_grid=None, physinformed_z_grid=None,
    dt_step_myr=None,
):
    """Pure-PyTorch KDK leapfrog that maps (z_obs, w_obs) at t_obs
    backwards to t=0.  Fully differentiable w.r.t. h_pc_dm, A_nn, and NN params.

    Two ways to discretise the total time t_obs_myr:

      * `dt_step_myr=None` (default): FIXED step COUNT.  n_steps leapfrog steps
        of size dt = t_obs/n_steps, so the timestep moves with the total time.
      * `dt_step_myr=<Myr>`: FIXED TIMESTEP.  The step length is pinned in
        advance (chosen small enough for leapfrog to be accurate) and the
        *number* of steps carries the total time:
            t_obs = N * dt_step,   N = n_full + frac.
        The integration is n_full full KDK steps plus one closing KDK step of
        length frac*dt_step, which makes the map (and the likelihood) a
        CONTINUOUS, differentiable function of the total time even though the
        step count is an integer.  d/dt_obs then flows through that last partial
        step alone -- which is exactly the continuous-time derivative of the
        flow map (the Hamiltonian vector field at the endpoint), so the gradient
        is cheaper AND free of the discretisation-error derivative that the
        fixed-step-count scheme mixes in.

    When use_nn_correction=True, adds a potential correction:
        V_nn(z) = A_nn * NN(z / z_max)          (time_dependent_nn=False)
        V_nn(z,t) = A_nn * NN(z/z_max, t/t_end) (time_dependent_nn=True)
    and derives the force via autograd: dV_nn/dz.

    When h_fourier_a / h_fourier_b (each a 1-D tensor of K coefficients) are
    given, the DM scale height becomes time-dependent:
        h(t) = h_pc_dm + sum_k [ a_k cos(2*pi*k*t/t_end)
                               + b_k sin(2*pi*k*t/t_end) ],   t in [0, t_end].
    The DM force is fully differentiable w.r.t. h_pc_dm and the coefficients.
    """
    G   = sim.G
    pi  = _math.pi
    MYR = sim.MYR_TO_TUNIT

    # baryon constants
    rho0_thin,  h_thin  = sim.rho0_thin,  sim.h_thin_pc
    rho0_thick, h_thick = sim.rho0_thick, sim.h_thick_pc
    rho0_gas,   h_gas   = sim.rho0_gas,   sim.h_gas_pc
    rho_DM   = sim.rho_DM
    z_max_pc = sim.z_max_pc

    # satellite constants
    sigma_t_int = sim.sigma_t_myr * MYR
    t_cut       = pulse_cut_sigmas * sigma_t_int
    four_pi_G   = 4.0 * pi * G

    t_end = t_obs_myr * MYR
    if dt_step_myr is None:
        n_full = int(n_steps)
        dt     = -t_end / float(n_steps)
        ds     = None                      # no partial closing step
    else:
        dt     = -float(dt_step_myr) * MYR         # FIXED timestep (a constant)
        _t_det = float(t_obs_myr.detach()) if torch.is_tensor(t_obs_myr) else float(t_obs_myr)
        n_full = int(_math.floor(_t_det / float(dt_step_myr) + 1e-9))
        # length of the closing partial step; differentiable in t_obs_myr
        ds     = -(t_end - n_full * float(dt_step_myr) * MYR)

    z = z_obs.clone()
    w = w_obs.clone()

    def _accel(t_int, z_now):
        # Baryon force (analytical tanh, 3 components)
        dphi = four_pi_G * (
              rho0_thin  * h_thin  * torch.tanh(z_now / h_thin)
            + rho0_thick * h_thick * torch.tanh(z_now / h_thick)
            + rho0_gas   * h_gas   * torch.tanh(z_now / h_gas)
        )
        # DM scale height: constant, or time-dependent (Fourier series in t)
        if h_fourier_a is not None:
            t_norm = t_int / t_end
            h_eff = h_pc_dm
            for _k in range(h_fourier_a.shape[0]):
                _ang = 2.0 * pi * (_k + 1) * t_norm
                h_eff = (h_eff
                         + h_fourier_a[_k] * _math.cos(_ang)
                         + h_fourier_b[_k] * _math.sin(_ang))
            h_eff = torch.clamp(h_eff, min=20.0, max=2000.0)
        else:
            h_eff = h_pc_dm
        # DM force (differentiable w.r.t. h_pc_dm and any Fourier coeffs)
        dphi = dphi + four_pi_G * rho_DM * h_eff * torch.tanh(z_now / h_eff)

        # NN potential correction: V_nn = A_nn * NN(...)
        # Force correction: dV_nn/dz via autograd
        if use_nn_correction and nn_model is not None and A_nn is not None:
            with torch.enable_grad():
                z_g = z_now.detach().requires_grad_(True)
                z_norm = z_g / z_max_pc

                if time_dependent_nn:
                    # Time-dependent: input is (N, 2) = [z_norm, t_norm]
                    t_norm_val = t_int / t_end  # scalar in [0, 1]
                    t_norm_vec = torch.full_like(z_norm, t_norm_val)
                    zt_input = torch.stack([z_norm, t_norm_vec], dim=-1)  # (N, 2)
                    V_nn = (A_nn * nn_model(zt_input)).squeeze(-1)
                else:
                    # Time-independent: input is (N, 1) = z_norm
                    V_nn = (A_nn * nn_model(z_norm.unsqueeze(-1))).squeeze(-1)

                dV_dz = torch.autograd.grad(V_nn.sum(), z_g, create_graph=True)[0]
            dphi = dphi + dV_dz

        a = -dphi

        # Satellite perturbation (amplitude = sim.Sigma_sat, or a learnable sat_amp)
        if use_satellite and t_int <= t_cut:
            tau   = t_int / sigma_t_int
            z_sat = sim.z0_sat_pc + sim.w_sat_kms * t_int
            _amp  = sim.Sigma_sat if sat_amp is None else sat_amp
            a = a + (
                -four_pi_G * _amp
                * _math.exp(-0.5 * tau ** 2)
                * torch.tanh((z_now - z_sat) / sim.H_sat_pc)
            )

        # Physics-informed NN: linear interpolation of precomputed V'(z) from
        # rho_nn(z) >= 0 via Poisson. Acceleration a += -V_nn'(z).
        if use_physinformed_nn and physinformed_Vp_grid is not None and physinformed_z_grid is not None:
            Vp_at_z = _interp_linear_1d(z_now, physinformed_z_grid, physinformed_Vp_grid)
            a = a - Vp_at_z

        # BOX-envelope satellite (theta-function approx; parked at fixed z_sat).
        # Differentiable w.r.t. sat_amp, box_t_on, box_t_off via sigmoid edges.
        if use_box_satellite and box_t_on is not None and box_t_off is not None and sat_amp is not None:
            z_sat_box = box_z_sat if box_z_sat is not None else sim.z0_sat_pc
            env_box = (torch.sigmoid((t_int - box_t_on) / box_tau)
                       * torch.sigmoid((box_t_off - t_int) / box_tau))
            a = a + (
                -four_pi_G * sat_amp * env_box
                * torch.tanh((z_now - z_sat_box) / sim.H_sat_pc)
            )
        return a

    t = t_end
    if n_full > 0:
        # half kick
        a = _accel(t, z)
        w = w + 0.5 * dt * a

        # drift-kick loop
        for _ in range(n_full):
            z = z + dt * w
            t = t + dt
            a = _accel(t, z)
            w = w + dt * a

        # undo trailing half kick
        w = w - 0.5 * dt * a

    # closing KDK step of the leftover length (fixed-timestep mode only)
    if ds is not None:
        a = _accel(t, z)
        w = w + 0.5 * ds * a
        z = z + ds * w
        t = t + ds
        a = _accel(t, z)
        w = w + 0.5 * ds * a
    return z, w


# ---------- 3. Joint optimiser ----------
def fit_h_pc_dm_nn_mle(
    sim, z_obs_np, w_obs_np, t_obs_myr,
    z0_sigma, w0_sigma,
    h_init=250.0,
    use_nn_correction=True,
    use_satellite=True,
    n_epochs=200,
    n_sub=1000,
    n_steps=5000,
    lr_nn=1e-3,
    lr_h=5.0,
    lr_A=1e-2,
    h_bounds=(50.0, 800.0),
    seed=42,
    verbose=True,
    # --- new parameters for sigma-mismatch / time-dep study ---
    sigma_mismatch_frac=0.0,
    z0_sigma_fit=None,   # explicit fit sigma (overrides sigma_mismatch_frac)
    w0_sigma_fit=None,   # explicit fit sigma (overrides sigma_mismatch_frac)
    time_dependent_nn=False,
    nn_frac_bound=0.1,
    nn_hidden=8,
    nn_depth=2,
    lambda_l2_A=0.0,
):
    """Joint MLE for h_pc_dm and optional NN potential correction.

    New parameters:
        sigma_mismatch_frac: fitting PDF uses z0_sigma*(1+frac), w0_sigma*(1+frac)
        z0_sigma_fit:        explicit fitting sigma_z (overrides sigma_mismatch_frac)
        w0_sigma_fit:        explicit fitting sigma_w (overrides sigma_mismatch_frac)
        time_dependent_nn:   if True, uses PotentialCorrectionMLPTimeDep (2-input)
        nn_frac_bound:       A_nn_max = nn_frac_bound * Phi_DM_max
        nn_hidden:           number of neurons per hidden layer (default 8;
                             bump to 32-64 for higher capacity against PDF-mismatch bias)
    All defaults reproduce the original behaviour.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N   = z_obs_np.size

    # Fitting sigmas: explicit values override sigma_mismatch_frac
    if z0_sigma_fit is None:
        z0_sigma_fit = z0_sigma * (1.0 + sigma_mismatch_frac)
    if w0_sigma_fit is None:
        w0_sigma_fit = w0_sigma * (1.0 + sigma_mismatch_frac)

    h_pc_dm = torch.nn.Parameter(torch.tensor(h_init, dtype=torch.float64))

    # ---- Reference for the NN cap: peak of the FULL static potential ----
    # (baryons + DM, evaluated at h_init, centered so Phi_static(0) = 0).
    # nn_frac_bound enforces  max_z |V_nn(z)| <= nn_frac_bound * Phi_static_peak,
    # i.e. the NN correction contributes at most nn_frac_bound (e.g. 10%) of the
    # static potential's peak. Enforced by projection on A_nn after each step,
    # NOT just an amplitude clamp -- so the NN can't sneak around the cap by
    # growing its weights while A_nn looks small.
    four_pi_G = 4.0 * _math.pi * sim.G
    _z_ref = np.linspace(-sim.z_max_pc, sim.z_max_pc, 401)
    _phi_static_ref = (
        sim.phi_component_sech2(_z_ref, sim.rho0_thin,  sim.h_thin_pc)
      + sim.phi_component_sech2(_z_ref, sim.rho0_thick, sim.h_thick_pc)
      + sim.phi_component_sech2(_z_ref, sim.rho0_gas,   sim.h_gas_pc)
      + four_pi_G * sim.rho_DM * h_init**2 * np.log(np.cosh(_z_ref / h_init))
    )
    _mid = len(_z_ref) // 2
    _phi_static_ref = _phi_static_ref - _phi_static_ref[_mid]      # shift Phi(0) = 0
    Phi_static_peak = float(np.max(np.abs(_phi_static_ref)))
    # Conservative amplitude clamp (NN output bounded by 1 in magnitude under tanh)
    A_nn_max = nn_frac_bound * Phi_static_peak
    # Grids used for the per-step projection of |V_nn| (torch, no_grad)
    _z_grid_norm = torch.tensor(_z_ref / sim.z_max_pc, dtype=torch.float64).unsqueeze(-1)
    # For time-dep NN we also need a t-grid (normalised to [0, 1])
    _t_grid_norm = torch.linspace(0.0, 1.0, 24, dtype=torch.float64)
    _proj_cap_value = nn_frac_bound * Phi_static_peak

    if use_nn_correction:
        if time_dependent_nn:
            nn_model = PotentialCorrectionMLPTimeDep(hidden=nn_hidden).double()
        else:
            if nn_depth >= 3:
                nn_model = PotentialCorrectionMLPBig(hidden=nn_hidden).double()
            else:
                nn_model = PotentialCorrectionMLP(hidden=nn_hidden).double()
        # A_nn must start > 0: with a zero-initialised NN last layer the
        # correction V = A_nn * NN(z) is still 0 at init (so the fit starts as
        # "no correction"), but a non-zero A_nn lets the NN weights receive a
        # gradient and switch on. Initialising A_nn = 0 freezes BOTH A_nn and
        # the NN weights at zero (dead correction) -- the original bug.
        A_nn = torch.nn.Parameter(torch.tensor(0.1 * A_nn_max, dtype=torch.float64))
        param_groups = [
            {"params": nn_model.parameters(), "lr": lr_nn},
            {"params": [h_pc_dm],             "lr": lr_h},
            {"params": [A_nn],                "lr": lr_A},
        ]
    else:
        nn_model = None
        A_nn = None
        param_groups = [{"params": [h_pc_dm], "lr": lr_h}]

    optimizer = torch.optim.Adam(param_groups)

    loss_history = []
    h_history    = []
    A_nn_history = []
    t0 = _time.time()

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        idx = rng.choice(N, size=min(n_sub, N), replace=False)
        z_t = torch.tensor(z_obs_np[idx], dtype=torch.float64)
        w_t = torch.tensor(w_obs_np[idx], dtype=torch.float64)

        z0, w0 = torch_back_integrate(
            z_t, w_t, t_obs_myr, h_pc_dm,
            sim, n_steps=n_steps,
            use_satellite=use_satellite,
            nn_model=nn_model,
            A_nn=A_nn,
            use_nn_correction=use_nn_correction,
            time_dependent_nn=time_dependent_nn,
        )

        # Negative log-likelihood under Gaussian f0 (with mismatched sigmas)
        nll = (0.5 * (z0 / z0_sigma_fit) ** 2
             + 0.5 * (w0 / w0_sigma_fit) ** 2
             + _math.log(2.0 * _math.pi * z0_sigma_fit * w0_sigma_fit))
        loss = nll.mean()
        # Optional L2 penalty on A_nn (Test 2e: does regularizing the NN help?)
        if lambda_l2_A > 0.0 and A_nn is not None:
            loss = loss + lambda_l2_A * (A_nn ** 2)

        loss.backward()

        if use_nn_correction and nn_model is not None:
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([h_pc_dm], max_norm=50.0)

        optimizer.step()

        with torch.no_grad():
            h_pc_dm.clamp_(*h_bounds)
            if A_nn is not None:
                # Step 1: conservative amplitude cap (assumes |NN| <= 1)
                A_nn.clamp_(0.0, A_nn_max)
                # Step 2: TIGHT projection cap.
                # Enforce max_{z[,t]} |V_nn(z[,t]) - V_nn(0[,t])| <= nn_frac_bound * Phi_static_peak
                # so the time-dep correction never exceeds nn_frac_bound of the full
                # static potential's peak, regardless of NN weight growth.
                if time_dependent_nn:
                    # Build (nz, nt) grid; subtract V_nn at z=0 per t-slice
                    _nz = _z_grid_norm.shape[0]
                    _nt = _t_grid_norm.shape[0]
                    _z_col = _z_grid_norm.expand(_nz, _nt)                      # (nz, nt)
                    _t_col = _t_grid_norm.unsqueeze(0).expand(_nz, _nt)         # (nz, nt)
                    _zt    = torch.stack([_z_col, _t_col], dim=-1).reshape(-1, 2)
                    V_nn_vals = (A_nn * nn_model(_zt).squeeze(-1)).reshape(_nz, _nt)
                    V_nn_vals = V_nn_vals - V_nn_vals[_mid:_mid+1, :]
                else:
                    V_nn_vals = (A_nn * nn_model(_z_grid_norm).squeeze(-1))
                    V_nn_vals = V_nn_vals - V_nn_vals[_mid]
                v_peak = V_nn_vals.abs().max().item()
                if v_peak > _proj_cap_value > 0.0:
                    A_nn.mul_(_proj_cap_value / v_peak)

        loss_history.append(loss.item())
        h_history.append(h_pc_dm.item())
        if A_nn is not None:
            A_nn_history.append(A_nn.item())

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            tag = "NN+h" if use_nn_correction else "h-only"
            A_str = f"  A_nn={A_nn.item():.2e}" if A_nn is not None else ""
            mm_str = (f"  σz_fit={z0_sigma_fit:.1f}  σw_fit={w0_sigma_fit:.1f}"
                      if (z0_sigma_fit != z0_sigma or w0_sigma_fit != w0_sigma) else "")
            td_str = "  [time-dep]" if time_dependent_nn else ""
            elapsed = _time.time() - t0
            print(f"[{tag}{td_str}] epoch {epoch+1:4d}/{n_epochs}  "
                  f"loss={loss.item():.4f}  h_pc_dm={h_pc_dm.item():.2f}{A_str}{mm_str}"
                  f"  ({elapsed:.0f}s)", flush=True)

    # Final achieved fraction: peak |V_nn(z[,t]) - V_nn(0[,t])| / Phi_static_peak
    nn_frac_achieved = None
    if use_nn_correction and nn_model is not None and A_nn is not None:
        with torch.no_grad():
            if time_dependent_nn:
                _nz = _z_grid_norm.shape[0]; _nt = _t_grid_norm.shape[0]
                _z_col = _z_grid_norm.expand(_nz, _nt)
                _t_col = _t_grid_norm.unsqueeze(0).expand(_nz, _nt)
                _zt = torch.stack([_z_col, _t_col], dim=-1).reshape(-1, 2)
                V_nn_final = (A_nn * nn_model(_zt).squeeze(-1)).reshape(_nz, _nt)
                V_nn_final = V_nn_final - V_nn_final[_mid:_mid+1, :]
            else:
                V_nn_final = (A_nn * nn_model(_z_grid_norm).squeeze(-1))
                V_nn_final = V_nn_final - V_nn_final[_mid]
            v_peak_final = V_nn_final.abs().max().item()
        nn_frac_achieved = v_peak_final / Phi_static_peak if Phi_static_peak > 0 else 0.0

    return {
        "h_pc_dm_fitted": h_pc_dm.item(),
        "nn_model":       nn_model,
        "A_nn":           A_nn.item() if A_nn is not None else None,
        "A_nn_max":       A_nn_max if use_nn_correction else None,
        "Phi_static_peak":   Phi_static_peak if use_nn_correction else None,
        "nn_frac_cap":       nn_frac_bound if use_nn_correction else None,
        "nn_frac_achieved":  nn_frac_achieved,
        "loss_history":   loss_history,
        "h_history":      h_history,
        "A_nn_history":   A_nn_history,
        "time_dependent_nn": time_dependent_nn,
    }


# ---------- 4. Time-dependent halo height h(t) (Fourier series) ----------
def eval_h_of_t_fourier(t_norm, h0, a_coeffs, b_coeffs):
    """Evaluate h(t) = h0 + sum_k [a_k cos(2*pi*k*t_norm) + b_k sin(2*pi*k*t_norm)].

    t_norm in [0, 1] (0 = initial time, 1 = observation time). NumPy helper
    used for plotting / reporting; mirrors the torch path in torch_back_integrate.
    """
    t_norm = np.asarray(t_norm, dtype=float)
    h = np.full_like(t_norm, float(h0))
    for k in range(len(a_coeffs)):
        ang = 2.0 * np.pi * (k + 1) * t_norm
        h = h + a_coeffs[k] * np.cos(ang) + b_coeffs[k] * np.sin(ang)
    return h


def fit_h_of_t_fourier_mle(
    sim, z_obs_np, w_obs_np, t_obs_myr,
    z0_sigma, w0_sigma,
    h_init=250.0,
    n_fourier=3,
    use_satellite=True,
    n_epochs=200, n_sub=500, n_steps=800,
    lr_h=5.0, lr_fourier=2.0,
    h_bounds=(50.0, 800.0),
    time_dep_frac=0.10,
    z0_sigma_fit=None, w0_sigma_fit=None,
    seed=42, verbose=True,
):
    """Joint MLE for a TIME-DEPENDENT DM scale height h(t):

        h(t) = h0 + sum_{k=1}^{K} [ a_k cos(2*pi*k*t/t_end)
                                  + b_k sin(2*pi*k*t/t_end) ]

    with t in [0, t_end] (t_end = t_obs). t/t_end = 0 -> initial time,
    t/t_end = 1 -> observation time. The base height h0 and the K (a_k, b_k)
    pairs are learnable; a_k = b_k = 0 at init, so the model starts as a static
    halo at h0 and only develops time dependence if the data prefer it.

    time_dep_frac caps how much of the gravitational potential may be carried by
    the time-dependent piece: after each step the Fourier coefficients are
    projected so the peak time variation of the DM potential (evaluated at
    z = z_max) stays within time_dep_frac of the static DM potential depth.
    Set to None to disable the cap.

    z0_sigma_fit / w0_sigma_fit let the fitting PDF be deliberately mismatched
    (same convention as fit_h_pc_dm_nn_mle) to drive a static-fit bias that the
    flexible h(t) may then absorb.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = z_obs_np.size

    if z0_sigma_fit is None:
        z0_sigma_fit = z0_sigma
    if w0_sigma_fit is None:
        w0_sigma_fit = w0_sigma

    h0       = torch.nn.Parameter(torch.tensor(h_init, dtype=torch.float64))
    a_coeffs = torch.nn.Parameter(torch.zeros(n_fourier, dtype=torch.float64))
    b_coeffs = torch.nn.Parameter(torch.zeros(n_fourier, dtype=torch.float64))

    optimizer = torch.optim.Adam([
        {"params": [h0],       "lr": lr_h},
        {"params": [a_coeffs], "lr": lr_fourier},
        {"params": [b_coeffs], "lr": lr_fourier},
    ])

    # Constants for the time-dependent-potential-fraction projection
    four_pi_G   = 4.0 * _math.pi * sim.G
    rho_DM      = sim.rho_DM
    z_max       = sim.z_max_pc
    t_grid_proj = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)

    def _phi_dm_at_zmax(h):
        return four_pi_G * rho_DM * h ** 2 * torch.log(torch.cosh(z_max / h))

    def _timedep_fraction():
        """Peak |Phi_DM(z_max,t) - Phi_DM(z_max,static)| / |Phi_DM(z_max,static)|."""
        h_t = h0 + torch.zeros_like(t_grid_proj)
        for _k in range(n_fourier):
            _ang = 2.0 * _math.pi * (_k + 1) * t_grid_proj
            h_t = h_t + a_coeffs[_k] * torch.cos(_ang) + b_coeffs[_k] * torch.sin(_ang)
        h_t  = torch.clamp(h_t, 20.0, 2000.0)
        phi0 = _phi_dm_at_zmax(h0)
        exc  = (_phi_dm_at_zmax(h_t) - phi0).abs().max()
        return exc, phi0.abs()

    loss_history, h0_history = [], []
    t0 = _time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        idx = rng.choice(N, size=min(n_sub, N), replace=False)
        z_t = torch.tensor(z_obs_np[idx], dtype=torch.float64)
        w_t = torch.tensor(w_obs_np[idx], dtype=torch.float64)

        z0, w0 = torch_back_integrate(
            z_t, w_t, t_obs_myr, h0, sim, n_steps=n_steps,
            use_satellite=use_satellite,
            h_fourier_a=a_coeffs, h_fourier_b=b_coeffs,
        )

        nll = (0.5 * (z0 / z0_sigma_fit) ** 2
             + 0.5 * (w0 / w0_sigma_fit) ** 2
             + _math.log(2.0 * _math.pi * z0_sigma_fit * w0_sigma_fit))
        loss = nll.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([h0], max_norm=50.0)
        torch.nn.utils.clip_grad_norm_([a_coeffs, b_coeffs], max_norm=50.0)
        optimizer.step()

        with torch.no_grad():
            h0.clamp_(*h_bounds)
            if time_dep_frac is not None:
                exc, phi0_abs = _timedep_fraction()
                allowed = time_dep_frac * phi0_abs
                if exc.item() > allowed.item() > 0.0:
                    scale = (allowed / exc)
                    a_coeffs.mul_(scale)
                    b_coeffs.mul_(scale)

        loss_history.append(loss.item())
        h0_history.append(h0.item())

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"[h(t)-fourier] epoch {epoch+1:4d}/{n_epochs}  loss={loss.item():.4f}  "
                  f"h0={h0.item():.2f}  |a|max={a_coeffs.abs().max().item():.2f}  "
                  f"|b|max={b_coeffs.abs().max().item():.2f}  ({_time.time()-t0:.0f}s)",
                  flush=True)

    a_np = a_coeffs.detach().numpy().copy()
    b_np = b_coeffs.detach().numpy().copy()
    h0_val = h0.item()
    t_norm_grid = np.linspace(0.0, 1.0, 200)
    h_of_t = eval_h_of_t_fourier(t_norm_grid, h0_val, a_np, b_np)

    with torch.no_grad():
        _exc, _phi0 = _timedep_fraction()
        timedep_frac_achieved = float(_exc / _phi0) if _phi0 > 0 else 0.0

    return {
        "h0_fitted":   h0_val,
        "time_dep_frac_cap":      time_dep_frac,
        "time_dep_frac_achieved": timedep_frac_achieved,
        "a_coeffs":    a_np,
        "b_coeffs":    b_np,
        "n_fourier":   n_fourier,
        "t_norm_grid": t_norm_grid,
        "h_of_t":      h_of_t,
        "h_mean":      float(np.mean(h_of_t)),
        "h_at_tobs":   float(eval_h_of_t_fourier(1.0, h0_val, a_np, b_np)),
        "h_at_t0":     float(eval_h_of_t_fourier(0.0, h0_val, a_np, b_np)),
        "loss_history": loss_history,
        "h0_history":   h0_history,
    }


# ---------- 5. Time-dependent SATELLITE amplitude (fitted, alpha-capped) ----------
def fit_h_sat_amp_mle(
    sim, z_obs_np, w_obs_np, t_obs_myr,
    z0_sigma, w0_sigma,
    h_init=250.0,
    n_epochs=200, n_sub=500, n_steps=2000,
    lr_h=5.0, lr_sat=2.0,
    h_bounds=(50.0, 800.0),
    time_dep_frac=0.10,
    z0_sigma_fit=None, w0_sigma_fit=None,
    seed=42, verbose=True,
):
    """Joint MLE for the DM scale height h0 AND a fitted SATELLITE amplitude
    (the time-dependent component, modelled with the satellite's true functional
    form sim.Kz_sat_common but a free amplitude).

    The amplitude is capped so the satellite's peak potential excursion stays
    within time_dep_frac of the static DM potential depth at z_max. Since the
    satellite potential is LINEAR in its amplitude, this is a simple clamp:
        sat_amp <= time_dep_frac * Phi_DM_TI(z_max) / (peak satellite potential per unit amp).
    alpha (time_dep_frac) is a LIMIT; the fitted amplitude is free to grow up to it.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = z_obs_np.size
    if z0_sigma_fit is None:
        z0_sigma_fit = z0_sigma
    if w0_sigma_fit is None:
        w0_sigma_fit = w0_sigma

    h0      = torch.nn.Parameter(torch.tensor(h_init, dtype=torch.float64))
    sat_amp = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))  # OK: multiplies a fixed shape

    # --- satellite potential per unit amplitude, for the alpha cap ---
    four_pi_G   = 4.0 * _math.pi * sim.G
    sigma_t_int = sim.sigma_t_myr * sim.MYR_TO_TUNIT
    t_cut       = 2.5 * sigma_t_int
    H           = sim.H_sat_pc
    zg = np.linspace(-sim.z_max_pc, sim.z_max_pc, 201)
    tg = np.linspace(0.0, t_cut, 60)
    peak = 0.0
    for t in tg:
        z_sat = sim.z0_sat_pc + sim.w_sat_kms * t
        env   = _math.exp(-0.5 * (t / sigma_t_int) ** 2)
        V  = four_pi_G * env * H * np.log(np.cosh((zg  - z_sat) / H))
        V0 = four_pi_G * env * H * _math.log(_math.cosh((0.0 - z_sat) / H))
        peak = max(peak, float(np.max(np.abs(V - V0))))
    Phi_DM_TI   = four_pi_G * sim.rho_DM * h_init ** 2 * np.log(np.cosh(sim.z_max_pc / h_init))
    sat_amp_max = (time_dep_frac * Phi_DM_TI / peak) if (time_dep_frac is not None and peak > 0) else None

    optimizer = torch.optim.Adam([
        {"params": [h0],      "lr": lr_h},
        {"params": [sat_amp], "lr": lr_sat},
    ])

    loss_history, h0_history = [], []
    t0 = _time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        idx = rng.choice(N, size=min(n_sub, N), replace=False)
        z_t = torch.tensor(z_obs_np[idx], dtype=torch.float64)
        w_t = torch.tensor(w_obs_np[idx], dtype=torch.float64)

        z0, w0 = torch_back_integrate(
            z_t, w_t, t_obs_myr, h0, sim, n_steps=n_steps,
            use_satellite=True, sat_amp=sat_amp)

        nll = (0.5 * (z0 / z0_sigma_fit) ** 2
             + 0.5 * (w0 / w0_sigma_fit) ** 2
             + _math.log(2.0 * _math.pi * z0_sigma_fit * w0_sigma_fit))
        loss = nll.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([h0], max_norm=50.0)
        torch.nn.utils.clip_grad_norm_([sat_amp], max_norm=50.0)
        optimizer.step()

        with torch.no_grad():
            h0.clamp_(*h_bounds)
            if sat_amp_max is not None:
                sat_amp.clamp_(0.0, sat_amp_max)
            else:
                sat_amp.clamp_(min=0.0)

        loss_history.append(loss.item())
        h0_history.append(h0.item())
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            cap = sat_amp_max if sat_amp_max is not None else float("inf")
            print(f"[h+sat] epoch {epoch+1:4d}/{n_epochs}  loss={loss.item():.4f}  "
                  f"h0={h0.item():.2f}  sat_amp={sat_amp.item():.2f}/{cap:.2f}  "
                  f"({_time.time()-t0:.0f}s)", flush=True)

    achieved = float(sat_amp.item() * peak / Phi_DM_TI) if Phi_DM_TI > 0 else 0.0
    return {
        "h0_fitted":              h0.item(),
        "sat_amp_fitted":         sat_amp.item(),
        "sat_amp_max":            sat_amp_max,
        "time_dep_frac_cap":      time_dep_frac,
        "time_dep_frac_achieved": achieved,
        "loss_history":           loss_history,
        "h0_history":             h0_history,
    }


# ---------- 6. Box-envelope satellite (theta-function in time) ----------
def fit_h_sat_box_mle(
    sim, z_obs_np, w_obs_np, t_obs_myr,
    z0_sigma, w0_sigma,
    h_init=250.0,
    t_on_init_myr=None, t_off_init_myr=None,
    sat_amp_init_frac=0.5,
    box_smoothing_myr=5.0,
    n_epochs=200, n_sub=500, n_steps=2000,
    lr_h=5.0, lr_sat=2.0, lr_window_myr=5.0,
    h_bounds=(50.0, 800.0),
    time_dep_frac=0.10,
    z0_sigma_fit=None, w0_sigma_fit=None,
    seed=42, verbose=True,
):
    """Joint MLE for h0 + (sat_amp, t_on, t_off) of a BOX-envelope satellite.

    Free parameters (4 total):
        h0       : DM scale height of the time-independent halo
        sat_amp  : satellite strength (the "alpha" in the user's notation)
        t_on     : start of the ON window  (in Myr)
        t_off    : end   of the ON window  (in Myr)

    Spatial shape: tanh((z - z0_sat)/H_sat) with z_sat parked at sim.z0_sat_pc
    (off-midplane -> the perturbation is asymmetric in z about the midplane).

    Time envelope: a SMOOTHED box (theta-function approximation):
        env(t) = sigmoid((t - t_on)/tau) * sigmoid((t_off - t)/tau)
    with tau = box_smoothing_myr (Myr). As tau -> 0 the envelope -> a strict
    theta-function; finite tau is needed for Adam gradients on t_on, t_off.

    sat_amp is hard-clamped each step so that the satellite's peak potential
    (over z, with env=1) stays within time_dep_frac * Phi_DM_TI(z_max).

    sat_amp is initialised at sat_amp_init_frac * sat_amp_max (non-zero) to
    bootstrap t_on/t_off gradients (they vanish when sat_amp = 0).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = z_obs_np.size
    if z0_sigma_fit is None:
        z0_sigma_fit = z0_sigma
    if w0_sigma_fit is None:
        w0_sigma_fit = w0_sigma

    MYR = sim.MYR_TO_TUNIT
    t_obs_int = t_obs_myr * MYR

    if t_on_init_myr is None:
        t_on_init_myr = 0.0
    if t_off_init_myr is None:
        t_off_init_myr = t_obs_myr

    # --- Cap on sat_amp: peak potential per unit amp (env=1, z_sat fixed) ---
    four_pi_G   = 4.0 * _math.pi * sim.G
    H           = sim.H_sat_pc
    z_sat_fixed = sim.z0_sat_pc
    zg = np.linspace(-sim.z_max_pc, sim.z_max_pc, 401)
    V  = four_pi_G * H * np.log(np.cosh((zg  - z_sat_fixed) / H))
    V0 = four_pi_G * H * _math.log(_math.cosh((0.0 - z_sat_fixed) / H))
    peak = float(np.max(np.abs(V - V0)))
    Phi_DM_TI   = four_pi_G * sim.rho_DM * h_init ** 2 * np.log(np.cosh(sim.z_max_pc / h_init))
    sat_amp_max = (time_dep_frac * Phi_DM_TI / peak) if (time_dep_frac is not None and peak > 0) else None

    init_amp = (sat_amp_init_frac * sat_amp_max) if sat_amp_max is not None else 0.0

    h0      = torch.nn.Parameter(torch.tensor(h_init, dtype=torch.float64))
    sat_amp = torch.nn.Parameter(torch.tensor(init_amp, dtype=torch.float64))
    t_on    = torch.nn.Parameter(torch.tensor(t_on_init_myr  * MYR, dtype=torch.float64))
    t_off   = torch.nn.Parameter(torch.tensor(t_off_init_myr * MYR, dtype=torch.float64))
    box_tau = box_smoothing_myr * MYR
    # lr_window is given in Myr/step; convert to internal time units
    lr_window_int = lr_window_myr * MYR

    optimizer = torch.optim.Adam([
        {"params": [h0],          "lr": lr_h},
        {"params": [sat_amp],     "lr": lr_sat},
        {"params": [t_on, t_off], "lr": lr_window_int},
    ])

    loss_history, h0_history = [], []
    t_on_hist, t_off_hist, sat_amp_hist = [], [], []
    t0 = _time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        idx = rng.choice(N, size=min(n_sub, N), replace=False)
        z_t = torch.tensor(z_obs_np[idx], dtype=torch.float64)
        w_t = torch.tensor(w_obs_np[idx], dtype=torch.float64)

        z0_, w0_ = torch_back_integrate(
            z_t, w_t, t_obs_myr, h0, sim, n_steps=n_steps,
            use_satellite=False,
            use_box_satellite=True, sat_amp=sat_amp,
            box_t_on=t_on, box_t_off=t_off, box_tau=box_tau,
            box_z_sat=z_sat_fixed,
        )

        nll = (0.5 * (z0_ / z0_sigma_fit) ** 2
             + 0.5 * (w0_ / w0_sigma_fit) ** 2
             + _math.log(2.0 * _math.pi * z0_sigma_fit * w0_sigma_fit))
        loss = nll.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([h0],          max_norm=50.0)
        torch.nn.utils.clip_grad_norm_([sat_amp],     max_norm=50.0)
        torch.nn.utils.clip_grad_norm_([t_on, t_off], max_norm=50.0 * MYR)
        optimizer.step()

        with torch.no_grad():
            h0.clamp_(*h_bounds)
            if sat_amp_max is not None:
                sat_amp.clamp_(0.0, sat_amp_max)
            else:
                sat_amp.clamp_(min=0.0)
            t_on.clamp_(0.0, t_obs_int)
            t_off.clamp_(0.0, t_obs_int)
            # If the window inverts, swap so t_on <= t_off
            if t_on.item() > t_off.item():
                _a = t_on.item(); _b = t_off.item()
                t_on.fill_(_b); t_off.fill_(_a)

        loss_history.append(loss.item())
        h0_history.append(h0.item())
        sat_amp_hist.append(sat_amp.item())
        t_on_hist.append(t_on.item() / MYR)
        t_off_hist.append(t_off.item() / MYR)
        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            cap = sat_amp_max if sat_amp_max is not None else float("inf")
            print(f"[h+sat-box] epoch {epoch+1:4d}/{n_epochs}  loss={loss.item():.4f}  "
                  f"h0={h0.item():.2f}  sat_amp={sat_amp.item():.3f}/{cap:.3f}  "
                  f"t_on={t_on.item()/MYR:.1f}  t_off={t_off.item()/MYR:.1f}  "
                  f"({_time.time()-t0:.0f}s)", flush=True)

    achieved = float(sat_amp.item() * peak / Phi_DM_TI) if Phi_DM_TI > 0 else 0.0
    return {
        "h0_fitted":              h0.item(),
        "sat_amp_fitted":         sat_amp.item(),
        "sat_amp_max":            sat_amp_max,
        "t_on_myr":               t_on.item() / MYR,
        "t_off_myr":              t_off.item() / MYR,
        "box_smoothing_myr":      box_smoothing_myr,
        "z_sat_pc":               z_sat_fixed,
        "time_dep_frac_cap":      time_dep_frac,
        "time_dep_frac_achieved": achieved,
        "loss_history":           loss_history,
        "h0_history":             h0_history,
        "sat_amp_history":        sat_amp_hist,
        "t_on_history":           t_on_hist,
        "t_off_history":          t_off_hist,
    }


# ---------- 6b. Free-sigma joint MLE (Test 2h: positive control) -----------
def fit_h_freesigma_mle(
    sim, z_obs_np, w_obs_np, t_obs_myr,
    z0_sigma, w0_sigma,
    z0_sigma_fit_init=None, w0_sigma_fit_init=None,
    h_init=250.0,
    n_epochs=200, n_sub=500, n_steps=2000,
    lr_h=5.0, lr_sigma=0.05,
    h_bounds=(50.0, 800.0),
    sigma_bounds=(0.1, 100.0),
    fit_sigma_z=True, fit_sigma_w=True,
    seed=42, verbose=True,
):
    """Joint MLE for h_pc_dm AND the assumed-initial-PDF sigmas as FREE parameters.

    The fitting PDF is f_0(z, w) = N(0, sigma_z_fit) x N(0, sigma_w_fit) with
    sigma_z_fit and/or sigma_w_fit LEARNABLE (not frozen at a mismatched value).

    NLL includes the log(sigma_z * sigma_w) normalisation so the sigmas are
    identifiable (they can't run to infinity).

    This is the structural fix advocated in note_why_potential_correction_fails:
    by letting the frozen wrong constant of the loss float, the loss-minimum
    moves to (h_true, sigma_z_true, sigma_w_true).  No NN involved.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = z_obs_np.size

    if z0_sigma_fit_init is None: z0_sigma_fit_init = z0_sigma
    if w0_sigma_fit_init is None: w0_sigma_fit_init = w0_sigma

    h_pc_dm = torch.nn.Parameter(torch.tensor(h_init, dtype=torch.float64))
    # Log-parameterise sigmas for positivity
    log_sz = torch.nn.Parameter(torch.tensor(_math.log(z0_sigma_fit_init), dtype=torch.float64),
                                requires_grad=fit_sigma_z)
    log_sw = torch.nn.Parameter(torch.tensor(_math.log(w0_sigma_fit_init), dtype=torch.float64),
                                requires_grad=fit_sigma_w)

    param_groups = [{"params": [h_pc_dm], "lr": lr_h}]
    if fit_sigma_z: param_groups.append({"params": [log_sz], "lr": lr_sigma})
    if fit_sigma_w: param_groups.append({"params": [log_sw], "lr": lr_sigma})
    optimizer = torch.optim.Adam(param_groups)

    loss_history, h_history, sz_history, sw_history = [], [], [], []
    log_sb_lo = _math.log(sigma_bounds[0])
    log_sb_hi = _math.log(sigma_bounds[1])
    t0 = _time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        idx = rng.choice(N, size=min(n_sub, N), replace=False)
        z_t = torch.tensor(z_obs_np[idx], dtype=torch.float64)
        w_t = torch.tensor(w_obs_np[idx], dtype=torch.float64)

        z0, w0 = torch_back_integrate(
            z_t, w_t, t_obs_myr, h_pc_dm, sim, n_steps=n_steps,
            use_satellite=False,
        )

        sz = torch.exp(log_sz)
        sw = torch.exp(log_sw)
        nll = (0.5 * (z0 / sz) ** 2
             + 0.5 * (w0 / sw) ** 2
             + torch.log(sz) + torch.log(sw)
             + _math.log(2.0 * _math.pi))
        loss = nll.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_([h_pc_dm], max_norm=50.0)
        optimizer.step()

        with torch.no_grad():
            h_pc_dm.clamp_(*h_bounds)
            log_sz.clamp_(log_sb_lo, log_sb_hi)
            log_sw.clamp_(log_sb_lo, log_sb_hi)

        loss_history.append(loss.item())
        h_history.append(h_pc_dm.item())
        sz_history.append(float(torch.exp(log_sz).item()))
        sw_history.append(float(torch.exp(log_sw).item()))

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"[h+freeSigma] epoch {epoch+1:4d}/{n_epochs}  loss={loss.item():.4f}  "
                  f"h={h_pc_dm.item():.2f}  sz={sz_history[-1]:.3f}  sw={sw_history[-1]:.3f}  "
                  f"({_time.time()-t0:.0f}s)", flush=True)

    return {
        "h_pc_dm_fitted":  h_pc_dm.item(),
        "sigma_z_fitted":  float(torch.exp(log_sz).item()),
        "sigma_w_fitted":  float(torch.exp(log_sw).item()),
        "fit_sigma_z":     fit_sigma_z,
        "fit_sigma_w":     fit_sigma_w,
        "loss_history":    loss_history,
        "h_history":       h_history,
        "sigma_z_history": sz_history,
        "sigma_w_history": sw_history,
    }


def fit_h_freegaussian_nn_mle(
    sim, z_obs_np, w_obs_np, t_obs_myr,
    z0_sigma, w0_sigma,
    mu_z_init=0.0, mu_w_init=0.0,
    z0_sigma_fit_init=None, w0_sigma_fit_init=None,
    h_init=250.0,
    use_nn_correction=True,
    use_satellite=False,
    n_epochs=200, n_sub=500, n_steps=2000,
    lr_h=5.0, lr_sigma=0.05, lr_mu=0.5,
    lr_nn=1e-3, lr_A=1e-2,
    h_bounds=(50.0, 800.0),
    sigma_bounds=(0.1, 5000.0),   # upper raised from 100: vertical-equilibrium
                                  # sigma_z ~ 259 pc (hot component up to ~1000 pc)
                                  # must not be clamped (applies to frozen sigmas too)
    mu_bounds=(-100.0, 100.0),
    fit_mu_z=True, fit_mu_w=True,
    fit_sigma_z=True, fit_sigma_w=True,
    nn_frac_bound=0.1,
    nn_hidden=8,
    time_dependent_nn=False,
    seed=42, verbose=True,
):
    """Joint MLE for h_pc_dm, the FULL Gaussian initial DF (mean AND sigma in
    both z and w), and an optional NN potential correction capped at
    nn_frac_bound (default 10%) of the static potential's peak.

    If time_dependent_nn=True the correction is V_nn(z, t) = A_nn * MLP(z/z_max,
    t/t_end) (PotentialCorrectionMLPTimeDep); the projection cap then holds over
    the whole (z, t) grid: max_{z,t} |V_nn(z,t) - V_nn(0,t)| <= cap.

    Fitting PDF:  f0(z, w) = N(mu_z, sigma_z) x N(mu_w, sigma_w),
    with mu_z, mu_w, sigma_z, sigma_w all LEARNABLE.  The NLL includes the
    log(sigma_z * sigma_w) normalisation so the sigmas are identifiable.

    The NN correction V_nn(z) = A_nn * MLP(z/z_max) is hard-capped by the same
    projection used in fit_h_pc_dm_nn_mle:
        max_z |V_nn(z) - V_nn(0)| <= nn_frac_bound * Phi_static_peak,
    enforced after every step (not just an amplitude clamp), so the NN cannot
    sneak past the cap by growing its weights.

    Purpose: check whether giving the fit this extra freedom (free DF mean+sigma
    AND a 10% NN potential bump) still recovers the truth -- i.e. returns the
    correct simulation PDF (mu~truth, sigma~truth) and the correct h_dm, with the
    NN staying near zero.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = z_obs_np.size

    if z0_sigma_fit_init is None: z0_sigma_fit_init = z0_sigma
    if w0_sigma_fit_init is None: w0_sigma_fit_init = w0_sigma

    h_pc_dm = torch.nn.Parameter(torch.tensor(h_init, dtype=torch.float64))
    mu_z = torch.nn.Parameter(torch.tensor(float(mu_z_init), dtype=torch.float64),
                              requires_grad=fit_mu_z)
    mu_w = torch.nn.Parameter(torch.tensor(float(mu_w_init), dtype=torch.float64),
                              requires_grad=fit_mu_w)
    # Log-parameterise sigmas for positivity
    log_sz = torch.nn.Parameter(torch.tensor(_math.log(z0_sigma_fit_init), dtype=torch.float64),
                                requires_grad=fit_sigma_z)
    log_sw = torch.nn.Parameter(torch.tensor(_math.log(w0_sigma_fit_init), dtype=torch.float64),
                                requires_grad=fit_sigma_w)

    # ---- NN-correction cap reference: peak of the FULL static potential ----
    four_pi_G = 4.0 * _math.pi * sim.G
    _z_ref = np.linspace(-sim.z_max_pc, sim.z_max_pc, 401)
    _phi_static_ref = (
        sim.phi_component_sech2(_z_ref, sim.rho0_thin,  sim.h_thin_pc)
      + sim.phi_component_sech2(_z_ref, sim.rho0_thick, sim.h_thick_pc)
      + sim.phi_component_sech2(_z_ref, sim.rho0_gas,   sim.h_gas_pc)
      + four_pi_G * sim.rho_DM * h_init**2 * np.log(np.cosh(_z_ref / h_init))
    )
    _mid = len(_z_ref) // 2
    _phi_static_ref = _phi_static_ref - _phi_static_ref[_mid]
    Phi_static_peak = float(np.max(np.abs(_phi_static_ref)))
    A_nn_max = nn_frac_bound * Phi_static_peak
    _z_grid_norm = torch.tensor(_z_ref / sim.z_max_pc, dtype=torch.float64).unsqueeze(-1)
    _proj_cap_value = nn_frac_bound * Phi_static_peak

    # For a time-dependent NN the cap must hold over (z, t): precompute a (z, t)
    # grid, flattened z-major so it reshapes back to (n_z, n_t).
    _n_z_ref = len(_z_ref)
    _n_t_ref = 21
    if time_dependent_nn:
        _t_ref = np.linspace(0.0, 1.0, _n_t_ref)
        _ZZ, _TT = np.meshgrid(_z_ref / sim.z_max_pc, _t_ref, indexing='ij')
        _zt_grid_norm = torch.tensor(
            np.stack([_ZZ.ravel(), _TT.ravel()], axis=-1), dtype=torch.float64)

    param_groups = [{"params": [h_pc_dm], "lr": lr_h}]
    if fit_mu_z:    param_groups.append({"params": [mu_z],   "lr": lr_mu})
    if fit_mu_w:    param_groups.append({"params": [mu_w],   "lr": lr_mu})
    if fit_sigma_z: param_groups.append({"params": [log_sz], "lr": lr_sigma})
    if fit_sigma_w: param_groups.append({"params": [log_sw], "lr": lr_sigma})

    if use_nn_correction:
        nn_model = (PotentialCorrectionMLPTimeDep(hidden=nn_hidden) if time_dependent_nn
                    else PotentialCorrectionMLP(hidden=nn_hidden)).double()
        # A_nn must start > 0 so the (zero-initialised) NN weights get a gradient.
        A_nn = torch.nn.Parameter(torch.tensor(0.1 * A_nn_max, dtype=torch.float64))
        param_groups.append({"params": nn_model.parameters(), "lr": lr_nn})
        param_groups.append({"params": [A_nn], "lr": lr_A})
    else:
        nn_model = None
        A_nn = None

    optimizer = torch.optim.Adam(param_groups)

    # peak |V_nn - midplane| over z (static) or (z,t) (time-dep); used for the
    # projection cap and final reporting. Returns (V_grid, peak_value).
    def _vnn_peak_grid():
        if not use_nn_correction:
            return None, 0.0
        if time_dependent_nn:
            V = (A_nn * nn_model(_zt_grid_norm).squeeze(-1)).reshape(_n_z_ref, _n_t_ref)
            V = V - V[_mid:_mid + 1, :]
        else:
            V = (A_nn * nn_model(_z_grid_norm).squeeze(-1))
            V = V - V[_mid]
        return V, V.abs().max().item()

    loss_history, h_history = [], []
    muz_history, muw_history, sz_history, sw_history, A_nn_history = [], [], [], [], []
    log_sb_lo = _math.log(sigma_bounds[0])
    log_sb_hi = _math.log(sigma_bounds[1])
    t0 = _time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        idx = rng.choice(N, size=min(n_sub, N), replace=False)
        z_t = torch.tensor(z_obs_np[idx], dtype=torch.float64)
        w_t = torch.tensor(w_obs_np[idx], dtype=torch.float64)

        z0, w0 = torch_back_integrate(
            z_t, w_t, t_obs_myr, h_pc_dm, sim, n_steps=n_steps,
            use_satellite=use_satellite,
            nn_model=nn_model, A_nn=A_nn,
            use_nn_correction=use_nn_correction,
            time_dependent_nn=time_dependent_nn,
        )

        sz = torch.exp(log_sz)
        sw = torch.exp(log_sw)
        nll = (0.5 * ((z0 - mu_z) / sz) ** 2
             + 0.5 * ((w0 - mu_w) / sw) ** 2
             + torch.log(sz) + torch.log(sw)
             + _math.log(2.0 * _math.pi))
        loss = nll.mean()

        loss.backward()
        if use_nn_correction and nn_model is not None:
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([h_pc_dm], max_norm=50.0)
        optimizer.step()

        with torch.no_grad():
            h_pc_dm.clamp_(*h_bounds)
            mu_z.clamp_(*mu_bounds)
            mu_w.clamp_(*mu_bounds)
            log_sz.clamp_(log_sb_lo, log_sb_hi)
            log_sw.clamp_(log_sb_lo, log_sb_hi)
            if A_nn is not None:
                # Step 1: conservative amplitude cap (assumes |NN| <= 1)
                A_nn.clamp_(0.0, A_nn_max)
                # Step 2: tight projection cap on peak |V_nn - midplane| over (z[,t])
                _, v_peak = _vnn_peak_grid()
                if v_peak > _proj_cap_value > 0.0:
                    A_nn.mul_(_proj_cap_value / v_peak)

        loss_history.append(loss.item())
        h_history.append(h_pc_dm.item())
        muz_history.append(mu_z.item())
        muw_history.append(mu_w.item())
        sz_history.append(float(torch.exp(log_sz).item()))
        sw_history.append(float(torch.exp(log_sw).item()))
        if A_nn is not None:
            A_nn_history.append(A_nn.item())

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            A_str = f"  A_nn={A_nn.item():.2e}" if A_nn is not None else ""
            print(f"[h+freeGauss{'+NN' if use_nn_correction else ''}] "
                  f"epoch {epoch+1:4d}/{n_epochs}  loss={loss.item():.4f}  "
                  f"h={h_pc_dm.item():.2f}  muz={muz_history[-1]:.2f}  muw={muw_history[-1]:.2f}  "
                  f"sz={sz_history[-1]:.3f}  sw={sw_history[-1]:.3f}{A_str}  "
                  f"({_time.time()-t0:.0f}s)", flush=True)

    # Final achieved NN fraction: peak |V_nn - midplane| / Phi_static_peak
    nn_frac_achieved = None
    V_nn_grid_out = None
    if use_nn_correction and nn_model is not None and A_nn is not None:
        with torch.no_grad():
            V_nn_final, v_peak_final = _vnn_peak_grid()
            V_nn_grid_out = V_nn_final.cpu().numpy()
        nn_frac_achieved = v_peak_final / Phi_static_peak if Phi_static_peak > 0 else 0.0

    return {
        "h_pc_dm_fitted":  h_pc_dm.item(),
        "mu_z_fitted":     mu_z.item(),
        "mu_w_fitted":     mu_w.item(),
        "sigma_z_fitted":  float(torch.exp(log_sz).item()),
        "sigma_w_fitted":  float(torch.exp(log_sw).item()),
        "nn_model":        nn_model,
        "A_nn":            A_nn.item() if A_nn is not None else None,
        "A_nn_max":        A_nn_max if use_nn_correction else None,
        "Phi_static_peak": Phi_static_peak if use_nn_correction else None,
        "nn_frac_cap":     nn_frac_bound if use_nn_correction else None,
        "nn_frac_achieved": nn_frac_achieved,
        "time_dependent_nn": time_dependent_nn,
        "z_grid":          _z_ref,
        "t_grid_norm":     (_t_ref if time_dependent_nn else None),
        "V_nn_grid":       V_nn_grid_out,
        "loss_history":    loss_history,
        "h_history":       h_history,
        "mu_z_history":    muz_history,
        "mu_w_history":    muw_history,
        "sigma_z_history": sz_history,
        "sigma_w_history": sw_history,
        "A_nn_history":    A_nn_history,
    }


# ---------- 6b. Double-Gaussian DF + time-dependent potential dPhi(x,t) ------
def fit_h_doublegaussian_timedep_mle(
    sim, z_obs_np, w_obs_np, t_obs_myr,
    # component-1 init (the "cold" population)
    mu_z1_init=0.0, mu_w1_init=0.0, sz1_init=8.0,  sw1_init=15.0,
    # component-2 init (the "hot" population)
    mu_z2_init=0.0, mu_w2_init=0.0, sz2_init=20.0, sw2_init=35.0,
    weight1_init=0.5,
    h_init=250.0,
    use_timedep_potential=True,
    use_satellite=False,
    n_epochs=200, n_sub=500, n_steps=2000,
    lr_h=5.0, lr_sigma=0.05, lr_mu=0.5, lr_w=0.05,
    lr_nn=1e-3, lr_A=1e-2,
    h_bounds=(50.0, 800.0),
    sigma_bounds=(0.1, 100.0),
    mu_bounds=(-200.0, 200.0),
    dphi_frac_bound=0.10,
    nn_hidden=8, n_t_grid=21,
    seed=42, verbose=True,
):
    """Task 1.  Joint MLE for:
        * h_pc_dm,
        * a DOUBLE-Gaussian initial DF -- a 2-component mixture
              f0(z,w) = pi   * N(mu_z1,sz1) N(mu_w1,sw1)
                      + (1-pi)* N(mu_z2,sz2) N(mu_w2,sw2),
          with EVERY property free (both means, both sigmas of each component,
          and the mixing weight pi), and
        * an optional TIME-DEPENDENT potential correction dPhi(x,t) =
              A_nn * MLP(z/z_max, t/t_end)
          i.e. the total potential is Phi(x) + dPhi(x,t).  dPhi is hard-capped
          by per-step projection so that
              max_{z,t} |dPhi(z,t) - dPhi(0,t)| <= dphi_frac_bound * Phi_static_peak.

    This is the natural extension of fit_h_freegaussian_nn_mle (Test 3): the DF
    is given far more freedom (two full Gaussian components instead of one) AND
    the potential bump is now time dependent.  The scientific question is whether
    this extra freedom still returns the correct h_dm and the true two-population
    DF, with dPhi(x,t) staying small when there is no genuine time-dependent
    feature in the data.

    The mixing weight is parameterised through a logit (pi = sigmoid(logit)) and
    the sigmas through logs, so all constraints are handled by the parameterisation
    plus light clamping.  NLL uses logaddexp over the two components and includes
    the log(sz*sw) normalisation of each, so the sigmas and weight are identifiable.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = z_obs_np.size

    h_pc_dm = torch.nn.Parameter(torch.tensor(float(h_init), dtype=torch.float64))
    # component means
    mu_z1 = torch.nn.Parameter(torch.tensor(float(mu_z1_init), dtype=torch.float64))
    mu_w1 = torch.nn.Parameter(torch.tensor(float(mu_w1_init), dtype=torch.float64))
    mu_z2 = torch.nn.Parameter(torch.tensor(float(mu_z2_init), dtype=torch.float64))
    mu_w2 = torch.nn.Parameter(torch.tensor(float(mu_w2_init), dtype=torch.float64))
    # component sigmas (log-parameterised for positivity)
    log_sz1 = torch.nn.Parameter(torch.tensor(_math.log(sz1_init), dtype=torch.float64))
    log_sw1 = torch.nn.Parameter(torch.tensor(_math.log(sw1_init), dtype=torch.float64))
    log_sz2 = torch.nn.Parameter(torch.tensor(_math.log(sz2_init), dtype=torch.float64))
    log_sw2 = torch.nn.Parameter(torch.tensor(_math.log(sw2_init), dtype=torch.float64))
    # mixing weight (logit -> pi = sigmoid(logit))
    _w0 = min(max(float(weight1_init), 1e-3), 1.0 - 1e-3)
    logit_w = torch.nn.Parameter(torch.tensor(_math.log(_w0 / (1.0 - _w0)), dtype=torch.float64))

    # ---- dPhi(x,t) cap reference: peak of the FULL static potential ----
    four_pi_G = 4.0 * _math.pi * sim.G
    _z_ref = np.linspace(-sim.z_max_pc, sim.z_max_pc, 401)
    _phi_static_ref = (
        sim.phi_component_sech2(_z_ref, sim.rho0_thin,  sim.h_thin_pc)
      + sim.phi_component_sech2(_z_ref, sim.rho0_thick, sim.h_thick_pc)
      + sim.phi_component_sech2(_z_ref, sim.rho0_gas,   sim.h_gas_pc)
      + four_pi_G * sim.rho_DM * h_init**2 * np.log(np.cosh(_z_ref / h_init))
    )
    _mid = len(_z_ref) // 2
    _phi_static_ref = _phi_static_ref - _phi_static_ref[_mid]
    Phi_static_peak = float(np.max(np.abs(_phi_static_ref)))
    A_nn_max = dphi_frac_bound * Phi_static_peak
    _proj_cap_value = dphi_frac_bound * Phi_static_peak

    # (z, t) grid for the time-dependent projection cap
    _z_norm_col = _z_ref / sim.z_max_pc
    _t_norm_row = np.linspace(0.0, 1.0, n_t_grid)
    _ZZ, _TT = np.meshgrid(_z_norm_col, _t_norm_row)           # (n_t, n_z)
    _zt_grid = torch.tensor(np.stack([_ZZ.ravel(), _TT.ravel()], axis=-1),
                            dtype=torch.float64)                # (n_t*n_z, 2)
    _nz = len(_z_norm_col)

    def _dphi_peak(nn_model, A_nn):
        """max_{z,t} |dPhi(z,t) - dPhi(0,t)| on the grid."""
        V = (A_nn * nn_model(_zt_grid).squeeze(-1)).reshape(n_t_grid, _nz)
        V = V - V[:, _mid:_mid + 1]                            # subtract z=0 per row
        return V.abs().max()

    param_groups = [
        {"params": [h_pc_dm], "lr": lr_h},
        {"params": [mu_z1, mu_w1, mu_z2, mu_w2], "lr": lr_mu},
        {"params": [log_sz1, log_sw1, log_sz2, log_sw2], "lr": lr_sigma},
        {"params": [logit_w], "lr": lr_w},
    ]
    if use_timedep_potential:
        nn_model = PotentialCorrectionMLPTimeDep(hidden=nn_hidden).double()
        A_nn = torch.nn.Parameter(torch.tensor(0.1 * A_nn_max, dtype=torch.float64))
        param_groups.append({"params": nn_model.parameters(), "lr": lr_nn})
        param_groups.append({"params": [A_nn], "lr": lr_A})
    else:
        nn_model, A_nn = None, None

    optimizer = torch.optim.Adam(param_groups)

    log_sb_lo, log_sb_hi = _math.log(sigma_bounds[0]), _math.log(sigma_bounds[1])
    LOG2PI = _math.log(2.0 * _math.pi)
    hist = {k: [] for k in ("loss", "h", "pi", "A_nn",
                            "mu_z1", "mu_w1", "sz1", "sw1",
                            "mu_z2", "mu_w2", "sz2", "sw2")}
    t0 = _time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        idx = rng.choice(N, size=min(n_sub, N), replace=False)
        z_t = torch.tensor(z_obs_np[idx], dtype=torch.float64)
        w_t = torch.tensor(w_obs_np[idx], dtype=torch.float64)

        z0, w0 = torch_back_integrate(
            z_t, w_t, t_obs_myr, h_pc_dm, sim, n_steps=n_steps,
            use_satellite=use_satellite,
            nn_model=nn_model, A_nn=A_nn,
            use_nn_correction=use_timedep_potential,
            time_dependent_nn=use_timedep_potential,
        )

        sz1, sw1 = torch.exp(log_sz1), torch.exp(log_sw1)
        sz2, sw2 = torch.exp(log_sz2), torch.exp(log_sw2)
        log_pi1 = torch.nn.functional.logsigmoid(logit_w)
        log_pi2 = torch.nn.functional.logsigmoid(-logit_w)
        comp1 = (log_pi1
                 - 0.5 * ((z0 - mu_z1) / sz1) ** 2
                 - 0.5 * ((w0 - mu_w1) / sw1) ** 2
                 - torch.log(sz1) - torch.log(sw1) - LOG2PI)
        comp2 = (log_pi2
                 - 0.5 * ((z0 - mu_z2) / sz2) ** 2
                 - 0.5 * ((w0 - mu_w2) / sw2) ** 2
                 - torch.log(sz2) - torch.log(sw2) - LOG2PI)
        log_f0 = torch.logaddexp(comp1, comp2)
        loss = (-log_f0).mean()

        loss.backward()
        if use_timedep_potential and nn_model is not None:
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([h_pc_dm], max_norm=50.0)
        optimizer.step()

        with torch.no_grad():
            h_pc_dm.clamp_(*h_bounds)
            for m in (mu_z1, mu_w1, mu_z2, mu_w2):
                m.clamp_(*mu_bounds)
            for ls in (log_sz1, log_sw1, log_sz2, log_sw2):
                ls.clamp_(log_sb_lo, log_sb_hi)
            if A_nn is not None:
                A_nn.clamp_(0.0, A_nn_max)
                v_peak = _dphi_peak(nn_model, A_nn).item()
                if v_peak > _proj_cap_value > 0.0:
                    A_nn.mul_(_proj_cap_value / v_peak)

        pi1 = torch.sigmoid(logit_w).item()
        hist["loss"].append(loss.item());  hist["h"].append(h_pc_dm.item())
        hist["pi"].append(pi1);            hist["A_nn"].append(A_nn.item() if A_nn is not None else 0.0)
        hist["mu_z1"].append(mu_z1.item()); hist["mu_w1"].append(mu_w1.item())
        hist["sz1"].append(sz1.item());     hist["sw1"].append(sw1.item())
        hist["mu_z2"].append(mu_z2.item()); hist["mu_w2"].append(mu_w2.item())
        hist["sz2"].append(sz2.item());     hist["sw2"].append(sw2.item())

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            A_str = f"  A_nn={A_nn.item():.2e}" if A_nn is not None else ""
            print(f"[h+2Gauss{'+dPhi(t)' if use_timedep_potential else ''}] "
                  f"ep {epoch+1:4d}/{n_epochs}  loss={loss.item():.4f}  h={h_pc_dm.item():.2f}  "
                  f"pi={pi1:.2f}  c1(sz,sw)=({sz1.item():.1f},{sw1.item():.1f})  "
                  f"c2=({sz2.item():.1f},{sw2.item():.1f}){A_str}  ({_time.time()-t0:.0f}s)",
                  flush=True)

    # final achieved dPhi fraction
    dphi_frac_achieved = None
    if use_timedep_potential and nn_model is not None and A_nn is not None:
        with torch.no_grad():
            dphi_frac_achieved = (_dphi_peak(nn_model, A_nn).item() / Phi_static_peak
                                  if Phi_static_peak > 0 else 0.0)

    return {
        "h_pc_dm_fitted": h_pc_dm.item(),
        "pi_fitted":      torch.sigmoid(logit_w).item(),
        "comp1": dict(mu_z=mu_z1.item(), mu_w=mu_w1.item(),
                      sigma_z=torch.exp(log_sz1).item(), sigma_w=torch.exp(log_sw1).item()),
        "comp2": dict(mu_z=mu_z2.item(), mu_w=mu_w2.item(),
                      sigma_z=torch.exp(log_sz2).item(), sigma_w=torch.exp(log_sw2).item()),
        "nn_model": nn_model,
        "A_nn": A_nn.item() if A_nn is not None else None,
        "A_nn_max": A_nn_max if use_timedep_potential else None,
        "Phi_static_peak": Phi_static_peak if use_timedep_potential else None,
        "dphi_frac_cap": dphi_frac_bound if use_timedep_potential else None,
        "dphi_frac_achieved": dphi_frac_achieved,
        "history": hist,
    }


# ---------- 7. Physics-informed NN potential (rho_nn >= 0 via Poisson) ------
def fit_h_physinformed_nn_mle(
    sim, z_obs_np, w_obs_np, t_obs_myr,
    z0_sigma, w0_sigma,
    h_init=250.0,
    n_epochs=200, n_sub=500, n_steps=2000,
    lr_h=5.0, lr_nn=1e-3, lr_A=1e-2,
    h_bounds=(50.0, 800.0),
    nn_frac_bound=0.10,
    nn_hidden=32, nn_layers=3,
    n_z_grid=401,
    z0_sigma_fit=None, w0_sigma_fit=None,
    seed=42, verbose=True,
):
    """Joint MLE for h_pc_dm with a PHYSICS-INFORMED NN correction.

    The NN parameterises a non-negative density rho_nn(z) >= 0 via softplus
    (symmetric in z). V_nn(z) is then DERIVED from Poisson:
        d^2 V_nn / dz^2 = 4 pi G rho_nn(z),  V_nn(0)=0, V_nn'(0)=0.
    Because rho_nn >= 0, V_nn is everywhere CONVEX -- it cannot mimic the
    concave anharmonic h-correction. So the (h, NN) degeneracy that lets a
    free NN absorb h-shifts is broken: the NN can still absorb a symmetric
    PDF mismatch (via a midplane mass concentration) but not h-changes.

    Free parameters: h_pc_dm, NN weights, A_nn (positive amplitude on rho_nn).
    Cap (projected each step): max|V_nn(z)| <= nn_frac_bound * Phi_static_peak.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = z_obs_np.size
    if z0_sigma_fit is None: z0_sigma_fit = z0_sigma
    if w0_sigma_fit is None: w0_sigma_fit = w0_sigma

    # --- Reference: full static potential peak (baryons + DM, h_init) ---
    four_pi_G = 4.0 * _math.pi * sim.G
    _z_ref = np.linspace(-sim.z_max_pc, sim.z_max_pc, 401)
    _phi_static_ref = (
        sim.phi_component_sech2(_z_ref, sim.rho0_thin,  sim.h_thin_pc)
      + sim.phi_component_sech2(_z_ref, sim.rho0_thick, sim.h_thick_pc)
      + sim.phi_component_sech2(_z_ref, sim.rho0_gas,   sim.h_gas_pc)
      + four_pi_G * sim.rho_DM * h_init**2 * np.log(np.cosh(_z_ref / h_init))
    )
    _mid_ref = len(_z_ref) // 2
    _phi_static_ref = _phi_static_ref - _phi_static_ref[_mid_ref]
    Phi_static_peak = float(np.max(np.abs(_phi_static_ref)))
    cap_value = nn_frac_bound * Phi_static_peak

    # --- z grid for Poisson integration (must be odd-length, symmetric) ---
    if n_z_grid % 2 == 0:
        n_z_grid += 1
    z_grid_pi = torch.tensor(
        np.linspace(-sim.z_max_pc, sim.z_max_pc, n_z_grid),
        dtype=torch.float64,
    )

    # --- NN and amplitude ---
    nn_model = PhysInformedDensityMLP(hidden=nn_hidden, n_layers=nn_layers).double()
    # At init, NN raw output = 0 -> softplus(0) = ln(2) (constant).
    # V_nn(z) from constant rho = (2 pi G rho)*z^2; peak at z_max:
    V_unit_peak = 2.0 * _math.pi * sim.G * _math.log(2.0) * sim.z_max_pc ** 2
    # Init A_nn so V_nn at init is 10% of the cap (well away from zero)
    A_nn_init = (0.1 * cap_value) / max(V_unit_peak, 1e-30)
    A_nn = torch.nn.Parameter(torch.tensor(A_nn_init, dtype=torch.float64))
    h_pc_dm = torch.nn.Parameter(torch.tensor(h_init, dtype=torch.float64))

    optimizer = torch.optim.Adam([
        {"params": nn_model.parameters(), "lr": lr_nn},
        {"params": [A_nn],                "lr": lr_A},
        {"params": [h_pc_dm],             "lr": lr_h},
    ])

    loss_history, h_history, A_nn_history = [], [], []
    t0 = _time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        idx = rng.choice(N, size=min(n_sub, N), replace=False)
        z_t = torch.tensor(z_obs_np[idx], dtype=torch.float64)
        w_t = torch.tensor(w_obs_np[idx], dtype=torch.float64)

        # Compute V_nn and V'_nn on the grid (differentiable through NN, A_nn)
        V_grid, Vp_grid = compute_physinformed_potential(nn_model, A_nn, z_grid_pi, sim)

        # Back-integrate using interpolated V' from the grid
        z0, w0 = torch_back_integrate(
            z_t, w_t, t_obs_myr, h_pc_dm, sim, n_steps=n_steps,
            use_satellite=False,
            use_physinformed_nn=True,
            physinformed_Vp_grid=Vp_grid,
            physinformed_z_grid=z_grid_pi,
        )

        nll = (0.5 * (z0 / z0_sigma_fit) ** 2
             + 0.5 * (w0 / w0_sigma_fit) ** 2
             + _math.log(2.0 * _math.pi * z0_sigma_fit * w0_sigma_fit))
        loss = nll.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([h_pc_dm], max_norm=50.0)
        torch.nn.utils.clip_grad_norm_([A_nn],    max_norm=10.0)
        optimizer.step()

        with torch.no_grad():
            h_pc_dm.clamp_(*h_bounds)
            A_nn.clamp_(min=0.0)
            # Projection: re-evaluate |V_nn| peak after step; scale A_nn if it exceeds cap
            V_check, _ = compute_physinformed_potential(nn_model, A_nn, z_grid_pi, sim)
            v_peak = V_check.abs().max().item()
            if v_peak > cap_value > 0.0:
                A_nn.mul_(cap_value / v_peak)

        loss_history.append(loss.item())
        h_history.append(h_pc_dm.item())
        A_nn_history.append(A_nn.item())

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(f"[phys-NN] epoch {epoch+1:4d}/{n_epochs}  loss={loss.item():.4f}  "
                  f"h={h_pc_dm.item():.2f}  A_nn={A_nn.item():.3e}  "
                  f"V_peak/cap={v_peak/cap_value:.2f}  "
                  f"({_time.time()-t0:.0f}s)", flush=True)

    # Final V_nn, achieved fraction
    with torch.no_grad():
        V_final, _Vp_final = compute_physinformed_potential(nn_model, A_nn, z_grid_pi, sim)
        v_peak_final = V_final.abs().max().item()
    nn_frac_achieved = v_peak_final / Phi_static_peak if Phi_static_peak > 0 else 0.0

    return {
        "h_pc_dm_fitted":       h_pc_dm.item(),
        "nn_model":             nn_model,
        "A_nn":                 A_nn.item(),
        "z_grid_pi":            z_grid_pi.detach().cpu().numpy(),
        "V_nn_grid":            V_final.detach().cpu().numpy(),
        "Phi_static_peak":      Phi_static_peak,
        "nn_frac_cap":          nn_frac_bound,
        "nn_frac_achieved":     nn_frac_achieved,
        "loss_history":         loss_history,
        "h_history":            h_history,
        "A_nn_history":         A_nn_history,
    }


# ==========================================================

# ==========================================================
# 8. Test 4 framework: data -> normalizing flow -> Hamiltonian flow
#    Stage 1: a RealNVP normalizing flow learns f_obs(z, w) from the
#             observed snapshot alone (no physics involved).
#    Stage 2: torch_back_integrate (h_dm free + NN potential correction
#             hard-capped at nn_frac_bound of the static peak) maps the
#             observed points back to t=0.  By Liouville
#                 f_obs(z, w) = f0(z0(z, w), w0(z, w)),
#    so the fit matches the flow's log-density to the KNOWN
#    double-Gaussian log f0 at the back-integrated points.
# ==========================================================

class _AffineCoupling(nn.Module):
    """RealNVP affine coupling for 2-D data.  One coordinate passes through;
    the other gets an affine map whose (log-scale, shift) are conditioned on
    the pass-through coordinate.  `flip` swaps the roles so a stack of layers
    transforms both coordinates.  Last layer zero-initialised -> the layer
    starts as the identity."""
    def __init__(self, flip, hidden=64):
        super().__init__()
        self.flip = bool(flip)
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def _split(self, xy):
        return (xy[:, 1:2], xy[:, 0:1]) if self.flip else (xy[:, 0:1], xy[:, 1:2])

    def _join(self, a, b):
        return torch.cat([b, a], dim=1) if self.flip else torch.cat([a, b], dim=1)

    def _scale_shift(self, a):
        st = self.net(a)
        s = 3.0 * torch.tanh(st[:, 0:1] / 3.0)      # soft-bounded log-scale
        return s, st[:, 1:2]

    def forward(self, xy):
        """Data -> base.  Returns (uv, log|det d(uv)/d(xy)|) per point."""
        a, b = self._split(xy)
        s, t = self._scale_shift(a)
        return self._join(a, (b - t) * torch.exp(-s)), -s.squeeze(-1)

    def inverse(self, uv):
        """Base -> data (used for sampling)."""
        a, b = self._split(uv)
        s, t = self._scale_shift(a)
        return self._join(a, b * torch.exp(s) + t)


class ZWNormalizingFlow(nn.Module):
    """RealNVP density model of the observed phase-space snapshot (z, w).
    Internally works on standardised coordinates; log_prob returns the density
    in physical units, i.e. per (pc * km/s) -- directly comparable with f0."""
    def __init__(self, n_couplings=8, hidden=64):
        super().__init__()
        self.layers = nn.ModuleList(
            _AffineCoupling(flip=(i % 2 == 1), hidden=hidden)
            for i in range(n_couplings))
        self.register_buffer('mu',  torch.zeros(2, dtype=torch.float64))
        self.register_buffer('sig', torch.ones(2, dtype=torch.float64))

    def set_standardization(self, z_obs_np, w_obs_np):
        self.mu  = torch.tensor([float(np.mean(z_obs_np)), float(np.mean(w_obs_np))],
                                dtype=torch.float64)
        self.sig = torch.tensor([float(np.std(z_obs_np)), float(np.std(w_obs_np))],
                                dtype=torch.float64)

    def log_prob(self, z, w):
        """z, w: 1-D tensors (pc, km/s).  Returns log f_obs(z, w) per point."""
        xy = (torch.stack([z, w], dim=1) - self.mu) / self.sig
        log_det = xy.new_zeros(xy.shape[0])
        for layer in self.layers:
            xy, ld = layer(xy)
            log_det = log_det + ld
        log_base = -0.5 * (xy ** 2).sum(dim=1) - _math.log(2.0 * _math.pi)
        return log_base + log_det - torch.log(self.sig).sum()

    @torch.no_grad()
    def sample(self, n, seed=None):
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        uv = torch.randn(n, 2, generator=gen, dtype=torch.float64)
        for layer in reversed(self.layers):
            uv = layer.inverse(uv)
        xy = uv * self.sig + self.mu
        return xy[:, 0], xy[:, 1]


def fit_zw_normalizing_flow(
    z_obs_np, w_obs_np,
    n_couplings=8, hidden=64,
    n_epochs=1500, batch_size=None, lr=1e-3,
    seed=0, verbose=True,
):
    """Stage 1: maximum-likelihood training of the RealNVP on the observed
    (z, w) snapshot.  batch_size=None -> full batch.
    Returns {'flow': ZWNormalizingFlow, 'loss_history': [mean NLL, nats]}."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = z_obs_np.size
    z_all = torch.tensor(z_obs_np, dtype=torch.float64)
    w_all = torch.tensor(w_obs_np, dtype=torch.float64)

    flow = ZWNormalizingFlow(n_couplings=n_couplings, hidden=hidden).double()
    flow.set_standardization(z_obs_np, w_obs_np)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)

    loss_history = []
    t0 = _time.time()
    for epoch in range(n_epochs):
        if batch_size is None or batch_size >= N:
            zb, wb = z_all, w_all
        else:
            idx = rng.choice(N, size=batch_size, replace=False)
            zb, wb = z_all[idx], w_all[idx]
        opt.zero_grad()
        loss = -flow.log_prob(zb, wb).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=10.0)
        opt.step()
        loss_history.append(loss.item())
        if verbose and ((epoch + 1) % 200 == 0 or epoch == 0):
            print(f"[NF] epoch {epoch+1:4d}/{n_epochs}  NLL={loss.item():.4f}  "
                  f"({_time.time()-t0:.0f}s)", flush=True)
    flow.eval()
    return {"flow": flow, "loss_history": loss_history}


def log_f0_double_gaussian_torch(z0, w0, pi1, comp1, comp2):
    """log of the 2-component Gaussian-mixture initial DF at (z0, w0), per
    (pc * km/s).  comp = dict(mu_z, mu_w, sz, sw) -- Task 1 convention."""
    def _log_comp(c):
        return (-0.5 * ((z0 - c['mu_z']) / c['sz']) ** 2
                - 0.5 * ((w0 - c['mu_w']) / c['sw']) ** 2
                - _math.log(2.0 * _math.pi * c['sz'] * c['sw']))
    return torch.logaddexp(_log_comp(comp1) + _math.log(f0_pi1_safe(pi1)),
                           _log_comp(comp2) + _math.log(f0_pi1_safe(1.0 - pi1)))


def f0_pi1_safe(p):
    """Clamp a mixture weight away from 0 so log() is finite."""
    return max(float(p), 1e-300)


def fit_h_nn_flow_to_f0_mle(
    sim, flow, z_obs_np, w_obs_np, t_obs_myr,
    f0_pi1, f0_comp1, f0_comp2,
    h_init=200.0,
    use_nn_correction=True,
    n_epochs=200, n_sub=500, n_steps=2000,
    lr_h=5.0, lr_nn=1e-3, lr_A=1e-2,
    h_bounds=(50.0, 800.0),
    nn_frac_bound=0.1, nn_hidden=8,
    loss_mode='match',
    seed=42, verbose=True,
):
    """Stage 2 of the Test 4 framework: Hamiltonian-flow fit to the true f0.

    torch_back_integrate maps each observed point (z, w) back to t=0 under
    Phi_static(h_dm) + V_nn(z), with V_nn = A_nn * MLP(z/z_max) hard-capped at
    nn_frac_bound of the static-potential peak (same projection cap as
    fit_h_freegaussian_nn_mle, and same A_nn = 0.1*cap init so the
    zero-initialised NN weights receive gradient).

    By Liouville the model observed density is f_model(z,w) = f0(T(z,w)), so
    with the trained normalizing flow providing log f_obs the residual
        r(z,w) = log f_NF(z,w) - log f0(T(z,w))
    vanishes for the correct potential (up to NF estimation noise).

    loss_mode:
      'match' -> mean r^2 over a data subsample (uses the NF density directly)
      'nll'   -> plain MLE, -mean log f0(T(z,w)) (NF only used as diagnostic)

    Free parameters: h_pc_dm, A_nn, NN weights.  f0 is FIXED at truth.
    Returns fitted values, per-epoch histories of both metrics, the final
    V_nn(z) on a grid, and the full-data back-integrated cloud (z0, w0).
    """
    assert loss_mode in ('match', 'nll')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = z_obs_np.size

    h_pc_dm = torch.nn.Parameter(torch.tensor(float(h_init), dtype=torch.float64))

    # ---- NN-correction cap reference: peak of the FULL static potential ----
    four_pi_G = 4.0 * _math.pi * sim.G
    _z_ref = np.linspace(-sim.z_max_pc, sim.z_max_pc, 401)
    _phi_static_ref = (
        sim.phi_component_sech2(_z_ref, sim.rho0_thin,  sim.h_thin_pc)
      + sim.phi_component_sech2(_z_ref, sim.rho0_thick, sim.h_thick_pc)
      + sim.phi_component_sech2(_z_ref, sim.rho0_gas,   sim.h_gas_pc)
      + four_pi_G * sim.rho_DM * h_init**2 * np.log(np.cosh(_z_ref / h_init))
    )
    _mid = len(_z_ref) // 2
    _phi_static_ref = _phi_static_ref - _phi_static_ref[_mid]
    Phi_static_peak = float(np.max(np.abs(_phi_static_ref)))
    A_nn_max = nn_frac_bound * Phi_static_peak
    _z_grid_norm = torch.tensor(_z_ref / sim.z_max_pc, dtype=torch.float64).unsqueeze(-1)
    _proj_cap_value = nn_frac_bound * Phi_static_peak

    param_groups = [{"params": [h_pc_dm], "lr": lr_h}]
    if use_nn_correction:
        nn_model = PotentialCorrectionMLP(hidden=nn_hidden).double()
        # A_nn must start > 0 so the (zero-initialised) NN weights get a gradient.
        A_nn = torch.nn.Parameter(torch.tensor(0.1 * A_nn_max, dtype=torch.float64))
        param_groups.append({"params": nn_model.parameters(), "lr": lr_nn})
        param_groups.append({"params": [A_nn], "lr": lr_A})
    else:
        nn_model, A_nn = None, None

    optimizer = torch.optim.Adam(param_groups)

    # log f_NF at the data points is independent of (h, NN): precompute once.
    # flow=None is allowed for pure-MLE ('nll') use: logf_nf is then only a
    # (zeroed) diagnostic and the 'match' loss must not be selected.
    z_all = torch.tensor(z_obs_np, dtype=torch.float64)
    w_all = torch.tensor(w_obs_np, dtype=torch.float64)
    if flow is None:
        assert loss_mode == 'nll', "loss_mode='match' requires a trained flow"
        logf_nf_all = torch.zeros(N, dtype=torch.float64)
    else:
        with torch.no_grad():
            logf_nf_all = flow.log_prob(z_all, w_all)

    loss_history, match_history, nll_history = [], [], []
    h_history, A_nn_history = [], []
    t0 = _time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        idx = rng.choice(N, size=min(n_sub, N), replace=False)
        z_t = z_all[idx]
        w_t = w_all[idx]
        logf_nf = logf_nf_all[idx]

        z0, w0 = torch_back_integrate(
            z_t, w_t, t_obs_myr, h_pc_dm, sim, n_steps=n_steps,
            use_satellite=False,
            nn_model=nn_model, A_nn=A_nn,
            use_nn_correction=use_nn_correction,
        )

        logf0 = log_f0_double_gaussian_torch(z0, w0, f0_pi1, f0_comp1, f0_comp2)
        match = ((logf_nf - logf0) ** 2).mean()
        nll   = -logf0.mean()
        loss  = match if loss_mode == 'match' else nll

        loss.backward()
        if nn_model is not None:
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([h_pc_dm], max_norm=50.0)
        optimizer.step()

        with torch.no_grad():
            h_pc_dm.clamp_(*h_bounds)
            if A_nn is not None:
                # Step 1: conservative amplitude cap (assumes |NN| <= 1)
                A_nn.clamp_(0.0, A_nn_max)
                # Step 2: tight projection cap on peak |V_nn(z) - V_nn(0)|
                V_nn_vals = (A_nn * nn_model(_z_grid_norm).squeeze(-1))
                V_nn_vals = V_nn_vals - V_nn_vals[_mid]
                v_peak = V_nn_vals.abs().max().item()
                if v_peak > _proj_cap_value > 0.0:
                    A_nn.mul_(_proj_cap_value / v_peak)

        loss_history.append(loss.item())
        match_history.append(match.item())
        nll_history.append(nll.item())
        h_history.append(h_pc_dm.item())
        if A_nn is not None:
            A_nn_history.append(A_nn.item())

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            A_str = f"  A_nn={A_nn.item():.2e}" if A_nn is not None else ""
            print(f"[NF->Hflow:{loss_mode}] epoch {epoch+1:4d}/{n_epochs}  "
                  f"match={match.item():.4f}  nll={nll.item():.4f}  "
                  f"h={h_pc_dm.item():.2f}{A_str}  "
                  f"({_time.time()-t0:.0f}s)", flush=True)

    # ---- final: full-data back-integration + V_nn shape + achieved cap ----
    with torch.no_grad():
        z0_fin, w0_fin = torch_back_integrate(
            z_all, w_all, t_obs_myr, h_pc_dm, sim, n_steps=n_steps,
            use_satellite=False,
            nn_model=nn_model, A_nn=A_nn,
            use_nn_correction=use_nn_correction,
        )
        logf0_fin = log_f0_double_gaussian_torch(
            z0_fin, w0_fin, f0_pi1, f0_comp1, f0_comp2)

    nn_frac_achieved = None
    V_nn_grid_np = None
    if use_nn_correction and nn_model is not None and A_nn is not None:
        with torch.no_grad():
            V_nn_final = (A_nn * nn_model(_z_grid_norm).squeeze(-1))
            V_nn_final = V_nn_final - V_nn_final[_mid]
            v_peak_final = V_nn_final.abs().max().item()
        nn_frac_achieved = v_peak_final / Phi_static_peak if Phi_static_peak > 0 else 0.0
        V_nn_grid_np = V_nn_final.numpy()

    return {
        "h_pc_dm_fitted":   h_pc_dm.item(),
        "nn_model":         nn_model,
        "A_nn":             A_nn.item() if A_nn is not None else None,
        "A_nn_max":         A_nn_max if use_nn_correction else None,
        "Phi_static_peak":  Phi_static_peak,
        "nn_frac_cap":      nn_frac_bound if use_nn_correction else None,
        "nn_frac_achieved": nn_frac_achieved,
        "z_grid":           _z_ref,
        "V_nn_grid":        V_nn_grid_np,
        "loss_history":     loss_history,
        "match_history":    match_history,
        "nll_history":      nll_history,
        "h_history":        h_history,
        "A_nn_history":     A_nn_history,
        "z0_final":         z0_fin.numpy(),
        "w0_final":         w0_fin.numpy(),
        "logf_nf_data":     logf_nf_all.numpy(),
        "logf0_final":      logf0_fin.numpy(),
    }


# ==========================================================
# 9. Joint (flow-as-f0) + h_dm + optional time-dependent NN potential.
#    Generalises Test 9's inline joint scheme: the normalizing flow IS the
#    initial DF f0 (flexible), fitted TOGETHER with h_dm and an optional
#    dPhi(z,t) NN potential correction (capped at nn_frac_bound of the static
#    peak).  Maximises  sum_i log f0_flow( T_{h,dPhi}(x_i) )  (Liouville).
# ==========================================================
def fit_h_flow_f0_joint_mle(
    sim, z_obs_np, w_obs_np, t_obs_myr,
    h_init=200.0,
    use_nn_correction=False, time_dependent_nn=True,
    nn_frac_bound=0.10, nn_hidden=8,
    n_couplings=8, flow_hidden=64,
    n_epochs=250, n_sub=500, n_steps=1000,
    lr_h=2.0, lr_flow=1e-3, lr_nn=1e-2, lr_A=5e-2,
    h_bounds=(50.0, 800.0), seed=0, verbose=False,
):
    """Flow AS f0, jointly with h (and optional time-dep NN potential)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    N = z_obs_np.size
    z_all = torch.tensor(z_obs_np, dtype=torch.float64)
    w_all = torch.tensor(w_obs_np, dtype=torch.float64)

    flow = ZWNormalizingFlow(n_couplings=n_couplings, hidden=flow_hidden).double()
    # standardise the flow from the h_init back-integrated cloud
    with torch.no_grad():
        z0i, w0i = torch_back_integrate(z_all, w_all, t_obs_myr,
                                        torch.tensor(float(h_init), dtype=torch.float64),
                                        sim, n_steps=n_steps)
    flow.set_standardization(z0i.numpy(), w0i.numpy())

    h_pc_dm = torch.nn.Parameter(torch.tensor(float(h_init), dtype=torch.float64))
    param_groups = [{"params": flow.parameters(), "lr": lr_flow},
                    {"params": [h_pc_dm], "lr": lr_h}]

    # ---- NN potential correction setup (projection cap on the static peak) ----
    nn_model = A_nn = A_nn_max = None
    if use_nn_correction:
        four_pi_G = 4.0 * _math.pi * sim.G
        _z_ref = np.linspace(-sim.z_max_pc, sim.z_max_pc, 401)
        _phi = (sim.phi_component_sech2(_z_ref, sim.rho0_thin,  sim.h_thin_pc)
              + sim.phi_component_sech2(_z_ref, sim.rho0_thick, sim.h_thick_pc)
              + sim.phi_component_sech2(_z_ref, sim.rho0_gas,   sim.h_gas_pc)
              + four_pi_G * sim.rho_DM * h_init**2 * np.log(np.cosh(_z_ref / h_init)))
        _mid = len(_z_ref) // 2
        _phi = _phi - _phi[_mid]
        Phi_static_peak = float(np.max(np.abs(_phi)))
        A_nn_max = nn_frac_bound * Phi_static_peak
        _z_grid_norm = torch.tensor(_z_ref / sim.z_max_pc, dtype=torch.float64).unsqueeze(-1)
        _nz, _nt = len(_z_ref), 21
        if time_dependent_nn:
            _t_ref = np.linspace(0.0, 1.0, _nt)
            _ZZ, _TT = np.meshgrid(_z_ref / sim.z_max_pc, _t_ref, indexing='ij')
            _zt_grid = torch.tensor(np.stack([_ZZ.ravel(), _TT.ravel()], axis=-1), dtype=torch.float64)
        nn_model = (PotentialCorrectionMLPTimeDep(hidden=nn_hidden) if time_dependent_nn
                    else PotentialCorrectionMLP(hidden=nn_hidden)).double()
        A_nn = torch.nn.Parameter(torch.tensor(0.1 * A_nn_max, dtype=torch.float64))
        param_groups.append({"params": nn_model.parameters(), "lr": lr_nn})
        param_groups.append({"params": [A_nn], "lr": lr_A})

        def _vpeak():
            if time_dependent_nn:
                V = (A_nn * nn_model(_zt_grid).squeeze(-1)).reshape(_nz, _nt)
                V = V - V[_mid:_mid + 1, :]
            else:
                V = (A_nn * nn_model(_z_grid_norm).squeeze(-1)); V = V - V[_mid]
            return V.abs().max().item()

    optimizer = torch.optim.Adam(param_groups)
    h_history, A_hist = [], []
    t0 = _time.time()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        idx = rng.choice(N, size=min(n_sub, N), replace=False)
        z0, w0 = torch_back_integrate(z_all[idx], w_all[idx], t_obs_myr, h_pc_dm, sim,
                                      n_steps=n_steps, nn_model=nn_model, A_nn=A_nn,
                                      use_nn_correction=use_nn_correction,
                                      time_dependent_nn=time_dependent_nn)
        loss = -flow.log_prob(z0, w0).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([h_pc_dm], max_norm=50.0)
        if use_nn_correction:
            torch.nn.utils.clip_grad_norm_(nn_model.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            h_pc_dm.clamp_(*h_bounds)
            if A_nn is not None:
                A_nn.clamp_(0.0, A_nn_max)
                vp = _vpeak()
                if vp > A_nn_max > 0.0:
                    A_nn.mul_(A_nn_max / vp)
        h_history.append(h_pc_dm.item())
        if A_nn is not None: A_hist.append(A_nn.item())
        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"[flow-f0{'+NN' if use_nn_correction else ''}] ep {epoch+1}/{n_epochs} "
                  f"loss={loss.item():.4f} h={h_pc_dm.item():.1f} ({_time.time()-t0:.0f}s)", flush=True)

    nn_frac = None
    if use_nn_correction:
        with torch.no_grad():
            nn_frac = _vpeak() / Phi_static_peak if Phi_static_peak > 0 else 0.0
    return {"h_pc_dm_fitted": h_pc_dm.item(), "h_history": h_history,
            "A_nn_history": A_hist, "nn_frac_achieved": nn_frac, "flow": flow,
            "nn_model": nn_model,
            "A_nn": (A_nn.item() if A_nn is not None else None),
            "A_nn_max": A_nn_max,
            "time_dependent_nn": time_dependent_nn}
