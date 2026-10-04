"""Shared setup + helpers for every experiment.

One source of truth for the simulation, the true parameters, the clean-data
generators and the small numerical helpers (joint MLE, back-integration,
profile-likelihood crossings, marginals).  Import this in every exp_*.py so no
test re-derives `sim`, sigmas, etc. -- this replaces the per-cell setup guards
that used to be copy-pasted into every notebook cell.
"""
import math
import os
import pickle
import time
from functools import partial as _partial
import numpy as np
import torch
import torch.nn as nn

from phase_space_simulation import Widmark1DSimulation, Widmark1DAnalysis
from fitter import torch_back_integrate

# ---- true parameters (the values the data are generated with) --------------
H_TRUE = 250.0        # dark-matter scale height [pc]
T_OBS = 400.0         # observation time [Myr]
W0_SIGMA = 20.0       # initial vertical-velocity dispersion [km/s]

sim = Widmark1DSimulation(h_pc_dm=H_TRUE)
ana = Widmark1DAnalysis(sim)

# midplane vertical frequency -> equilibrium sigma_z = sigma_w / nu
NU = math.sqrt(4.0 * math.pi * sim.G *
               (sim.rho0_thin + sim.rho0_thick + sim.rho0_gas + sim.rho_DM))
Z0_SIGMA = W0_SIGMA / NU


def sz_eq(sw):
    """Equilibrium sigma_z for a given sigma_w."""
    return sw / NU


# ---- data generators -------------------------------------------------------
def make_gaussian_data(N=2000, z_sigma=None, w_sigma=None, mu_z=0.0, mu_w=0.0,
                       seed_sample=7, seed_evolve=0):
    """Clean snapshot from a single centred 2-D Gaussian f0, evolved to t_obs
    in the true potential, NO satellite.  (Baseline used by Tests 9, 10, A, B.)"""
    z_sigma = Z0_SIGMA if z_sigma is None else z_sigma
    w_sigma = W0_SIGMA if w_sigma is None else w_sigma
    z0, w0 = sim.sample_near_midplane(N=N, z_mean_pc=mu_z, z_sigma_pc=z_sigma,
                                      w_mean_kms=mu_w, w_sigma_kms=w_sigma, seed=seed_sample)
    Zs, Ws, _ = sim.evolve_record(z0, w0, t_obs_myr=T_OBS, n_steps=12000, n_snaps=10,
                                  seed=seed_evolve, use_satellite=False, h_pc_dm=H_TRUE)
    return Zs[-1].astype(np.float64), Ws[-1].astype(np.float64)


# ---- Hamiltonian-flow helpers ----------------------------------------------
def backint(z_obs, w_obs, h, ns=2000, t_obs=None, dt_step=None):
    """Back-integrate the observed cloud to t=0 at scale height h (numpy in/out).
    `t_obs` overrides the total dynamical time (default T_OBS).

    Two discretisations, as in `torch_back_integrate`:
      * default -- fixed step COUNT `ns`, timestep t_obs/ns;
      * `dt_step` given -- fixed TIMESTEP dt_step [Myr], step count t_obs/dt_step
        (real-valued, closed by one partial step).  This is what the
        time-parameter fit uses, so the timestep never grows with the fitted
        total time."""
    t_obs = T_OBS if t_obs is None else t_obs
    with torch.no_grad():
        zb, wb = torch_back_integrate(torch.tensor(z_obs), torch.tensor(w_obs),
                                      t_obs, torch.tensor(float(h)), sim,
                                      n_steps=2000 if ns is None else ns,
                                      dt_step_myr=dt_step)
    return zb.numpy(), wb.numpy()


def joint_fit(make_flow, z_obs, w_obs, h_init, n_epochs=300, ns=1000, n_sub=500,
              lr_flow=5e-2, lr_h=2.0, seed=0):
    """Joint MLE of (flow params, h).  make_flow() -> nn.Module with .log_prob(z,w).
    Returns (flow, h_fit, h_history)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    flow = make_flow()
    h = nn.Parameter(torch.tensor(float(h_init), dtype=torch.float64))
    opt = torch.optim.Adam([{'params': flow.parameters(), 'lr': lr_flow},
                            {'params': [h],               'lr': lr_h}])
    hist = []
    for _ in range(n_epochs):
        idx = rng.choice(z_obs.size, size=min(n_sub, z_obs.size), replace=False)
        zb, wb = torch_back_integrate(torch.tensor(z_obs[idx]), torch.tensor(w_obs[idx]),
                                      T_OBS, h, sim, n_steps=ns)
        loss = -flow.log_prob(zb, wb).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([h], max_norm=50.0)
        opt.step()
        with torch.no_grad():
            h.clamp_(50.0, 800.0)
        hist.append(h.item())
    return flow, h.item(), hist


# ---- profile-likelihood / plotting helpers ---------------------------------
def gauss_pdf(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (math.sqrt(2 * math.pi) * s)


def cross(hs, d, level, side):
    """Linear-interp h where (Delta)NLL curve `d` crosses `level`, walking
    outward from the minimum in the +/- `side` direction (1 sigma <-> level=0.5)."""
    k = int(np.argmin(d))
    rng_ = range(k, len(hs) - 1) if side > 0 else range(k, 0, -1)
    for i in rng_:
        j = i + 1 if side > 0 else i - 1
        if (d[i] - level) * (d[j] - level) <= 0 and d[j] != d[i]:
            f = (level - d[i]) / (d[j] - d[i])
            return hs[i] + f * (hs[j] - hs[i])
    return hs[-1] if side > 0 else hs[0]


def mixture_pdf(comps, weights):
    """Return f(x) = sum_k weight_k * N(x; mu_k, s_k).  comps = [(mu, s), ...]."""
    def f(x):
        return sum(wk * gauss_pdf(x, mu, s) for wk, (mu, s) in zip(weights, comps))
    return f


def _evolve(z0, w0, seed_evolve=0):
    Zs, Ws, _ = sim.evolve_record(z0, w0, t_obs_myr=T_OBS, n_steps=12000, n_snaps=10,
                                  seed=seed_evolve, use_satellite=False, h_pc_dm=H_TRUE)
    return Zs[-1].astype(np.float64), Ws[-1].astype(np.float64)


def make_mixture_w_data(N=2000, comps=((0.0, 20.0),), weights=(1.0,),
                        z_sigma=None, seed=0, seed_evolve=0):
    """Snapshot from f0 = N(0, z_sigma) in z  x  Gaussian mixture in w, evolved to
    t_obs (no satellite).  Used by the non-Gaussian tests (11 core+wings,
    12 bimodal)."""
    z_sigma = Z0_SIGMA if z_sigma is None else z_sigma
    rng = np.random.default_rng(seed)
    z0 = rng.normal(0.0, z_sigma, N)
    idx = rng.choice(len(comps), size=N, p=np.asarray(weights) / np.sum(weights))
    mu = np.array([comps[i][0] for i in idx]); sw = np.array([comps[i][1] for i in idx])
    w0 = rng.normal(mu, sw)
    return _evolve(z0, w0, seed_evolve)


def small_realnvp_factory(z_obs, w_obs, n_couplings=4, hidden=6, h_std=300.0):
    """A make_flow() closure returning a SMALL RealNVP (few couplings, tiny hidden)
    with standardisation fixed from a rough back-integration.  Also returns the
    trainable parameter count."""
    from fitter import ZWNormalizingFlow
    n_params = sum(p.numel() for p in ZWNormalizingFlow(n_couplings, hidden).parameters())

    def make_flow():
        f = ZWNormalizingFlow(n_couplings=n_couplings, hidden=hidden).double()
        zi, wi = backint(z_obs, w_obs, h_std, ns=1000)
        f.set_standardization(zi, wi)
        return f
    return make_flow, n_params


# default ladder of RealNVP capacities for the non-Gaussian sweeps
DEFAULT_NF_SIZES = [("NVP c2h4", 2, 4), ("NVP c4h6", 4, 6),
                    ("NVP c6h12", 6, 12), ("NVP c8h64", 8, 64)]


def run_nf_sweep_test(tid, z_obs, w_obs, true_pdf_z, true_pdf_w,
                      sizes=None, inits=(150.0, 250.0, 350.0), lr_flow=1e-3):
    """Fit SEVERAL RealNVP flows of increasing capacity to the same
    (non-Gaussian) data, jointly with h.  For each capacity record:
      * the recovered-h spread over `inits` (how well h is constrained), and
      * the fitted f0 marginals on a COMMON grid (how well the flow mimics the
        true f0 / data), plus their total-variation misfit to the true f0.
    Only normalizing flows are used -- no mixture / parametric model.
    `sizes` = list of (label, n_couplings, hidden)."""
    sizes = DEFAULT_NF_SIZES if sizes is None else sizes
    zref, wref = backint(z_obs, w_obs, H_TRUE)      # common grid + data histogram
    mid = inits[len(inits) // 2]
    out, zg, wg, true_z, true_w = [], None, None, None, None
    for label, n_c, n_h in sizes:
        make_flow, npar = small_realnvp_factory(z_obs, w_obs, n_c, n_h)
        fits, flow_mid = {}, None
        for hi in inits:
            flow, hf, _ = joint_fit(make_flow, z_obs, w_obs, hi, lr_flow=lr_flow)
            fits[hi] = hf
            if hi == mid:
                flow_mid = flow
        zg, mz, wg, mw = fitted_marginals(flow_mid, zref, wref)   # common grid (from zref,wref)
        if true_z is None:
            true_z, true_w = true_pdf_z(zg), true_pdf_w(wg)
        tv_w = 0.5 * float(np.sum(np.abs(mw - true_w)) * (wg[1] - wg[0]))
        tv_z = 0.5 * float(np.sum(np.abs(mz - true_z)) * (zg[1] - zg[0]))
        hs = [fits[hi] for hi in inits]
        out.append(dict(label=label, n=npar, hs=hs, spread=float(max(hs) - min(hs)),
                        mz=mz, mw=mw, tv_w=tv_w, tv_z=tv_z))
    return dict(id=tid, sizes=out, inits=list(inits), h_true=H_TRUE,
                zg=zg, wg=wg, true_z=true_z, true_w=true_w,
                z0_samples=zref, w0_samples=wref)


def plot_nf_sweep(res, title=None):
    import matplotlib.pyplot as plt
    sizes = res["sizes"]
    cols = plt.cm.viridis(np.linspace(0.0, 0.82, len(sizes)))
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    for a, grid, marg, samp, true, lab, unit in [
            (ax[0, 0], "wg", "mw", "w0_samples", "true_w", "w_0", "km/s"),
            (ax[0, 1], "zg", "mz", "z0_samples", "true_z", "z_0", "pc")]:
        a.hist(res[samp], bins=45, density=True, alpha=0.30, color="0.6", label="data (back-int)")
        a.plot(res[grid], res[true], "k--", lw=2.4, label="true $f_0$")
        for s, c in zip(sizes, cols):
            a.plot(res[grid], s[marg], color=c, lw=1.8, label=f'{s["label"]} ({s["n"]}p)')
        a.set_xlabel(fr"${lab}$ [{unit}]"); a.set_ylabel("density"); a.legend(fontsize=7)
    ax[0, 0].set_title("$w_0$ marginal: flow fits vs true $f_0$ and data")
    ax[0, 1].set_title("$z_0$ marginal")

    ns = [s["n"] for s in sizes]
    ax[1, 0].plot(ns, [s["spread"] for s in sizes], "s-", color="C0", ms=7)
    ax[1, 0].set_xscale("log"); ax[1, 0].set_xlabel("bijection parameters")
    ax[1, 0].set_ylabel(r"recovered-$h$ spread over inits [pc]")
    ax[1, 0].set_title("h constraint LOOSENS with more parameters")
    for s in sizes:
        ax[1, 0].annotate(s["label"].replace("NVP ", ""), (s["n"], s["spread"]),
                          fontsize=7, textcoords="offset points", xytext=(0, 6))

    ax[1, 1].plot(ns, [s["tv_w"] for s in sizes], "o-", color="C3", ms=7, label=r"$w_0$ misfit")
    ax[1, 1].plot(ns, [s["tv_z"] for s in sizes], "^--", color="C2", ms=6, label=r"$z_0$ misfit")
    ax[1, 1].set_xscale("log"); ax[1, 1].set_xlabel("bijection parameters")
    ax[1, 1].set_ylabel("total-variation misfit to true $f_0$")
    ax[1, 1].set_title("shape misfit to true $f_0$ vs #parameters"); ax[1, 1].legend(fontsize=9)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def run_density_test(tid, z_obs, w_obs, make_flow, true_pdf_z, true_pdf_w, n_params,
                     inits=(180, 350), lr_flow=5e-2, refit_lr=None,
                     n_profile=13, profile_epochs=400, warm_epochs=1500,
                     profile_mode="refit"):
    """Generic flow-as-f0 test: joint fit (flow params + h) from two inits, a
    profile likelihood in h, and 1-D marginals with the true f0 overlaid.
    Returns the standard result dict used by plot_density_test.  Works with ANY
    flow exposing .log_prob(z,w).

    profile_mode:
      'refit' -> re-fit the flow at each h (warm up/down sweeps); general.
      'gauss' -> closed-form profile for a diagonal-Gaussian model: the optimal
                 NLL at fixed h is the Gaussian entropy of the back-integrated
                 cloud, N*(log 2pi + log sigma_z + log sigma_w + 1).  Exact and
                 noise-free -- use it whenever make_flow is a diagonal affine map.
    """
    refit_lr = lr_flow if refit_lr is None else refit_lr
    N = z_obs.size
    fits, flows = {}, {}
    for hi in inits:
        flow, hf, hist = joint_fit(make_flow, z_obs, w_obs, hi, lr_flow=lr_flow)
        fits[hi] = dict(h=hf, hist=hist); flows[hi] = flow
    h_est = float(np.mean([fits[hi]["h"] for hi in inits]))

    hgrid = np.linspace(h_est - 90, h_est + 90, n_profile)
    if profile_mode == "gauss":
        nll = np.empty(len(hgrid))
        for i, h in enumerate(hgrid):
            zb, wb = backint(z_obs, w_obs, h)
            nll[i] = N * (math.log(2 * math.pi) + math.log(zb.std())
                          + math.log(wb.std()) + 1.0)
        noise = 0.0
    else:
        def _refit(flow, zb, wb, epochs, lr=refit_lr):
            opt = torch.optim.Adam(flow.parameters(), lr=lr)
            zt, wt = torch.tensor(zb), torch.tensor(wb)
            for _ in range(epochs):
                opt.zero_grad(); loss = -flow.log_prob(zt, wt).mean(); loss.backward()
                torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0); opt.step()
            with torch.no_grad():
                return float((-flow.log_prob(zt, wt)).mean())

        clouds = [backint(z_obs, w_obs, h) for h in hgrid]
        fu = make_flow(); _refit(fu, *clouds[0], warm_epochs)
        up = N * np.array([_refit(fu, *clouds[i], profile_epochs) for i in range(len(hgrid))])
        fd = make_flow(); _refit(fd, *clouds[-1], warm_epochs)
        dn = N * np.array([_refit(fd, *clouds[i], profile_epochs)
                           for i in range(len(hgrid) - 1, -1, -1)])[::-1]
        nll = np.minimum(up, dn); noise = float(np.abs(up - dn).max())
    dnll = nll - nll.min()
    h_min = float(hgrid[dnll.argmin()])
    lo, hi_ = cross(hgrid, dnll, 0.5, -1), cross(hgrid, dnll, 0.5, +1)
    sig = 0.5 * (hi_ - lo)

    flow0 = flows[inits[0]]
    zb, wb = backint(z_obs, w_obs, fits[inits[0]]["h"])
    zg, mz, wg, mw = fitted_marginals(flow0, zb, wb)

    return dict(id=tid, n_params=n_params, inits=list(inits), fits=fits, h_est=h_est,
                h_true=H_TRUE, prof_h=hgrid, prof_dnll=dnll, noise=noise, h_min=h_min,
                sig=sig, lo=lo, hi=hi_, z0_samples=zb, w0_samples=wb,
                zg=zg, mz=mz, wg=wg, mw=mw, true_z=true_pdf_z(zg), true_w=true_pdf_w(wg))


def plot_density_test(res, title=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for hi in res["inits"]:
        f = res["fits"][hi]
        ax[0, 0].plot(f["hist"], label=f'init h={hi} -> {f["h"]:.1f}')
    ax[0, 0].axhline(res["h_true"], color="k", ls="--", label=f'truth {res["h_true"]:.0f}')
    ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel(r"$h_{\rm dm}$ [pc]")
    ax[0, 0].set_title(f"joint ({res['n_params']}-param flow $f_0$)+$h$")
    ax[0, 0].legend()

    ax[0, 1].plot(res["prof_h"], res["prof_dnll"], "o-", color="C3", ms=4,
                  label="profile $\\Delta$NLL(h)")
    ax[0, 1].axhline(0.5, color="k", ls=":", label=r"$\Delta$NLL=0.5 (1$\sigma$)")
    ax[0, 1].axvspan(res["lo"], res["hi"], color="C2", alpha=0.15,
                     label=fr'{res["h_min"]:.0f}$\pm${res["sig"]:.0f} pc')
    ax[0, 1].axvline(res["h_true"], color="k", ls="--", label=f'truth {res["h_true"]:.0f}')
    ax[0, 1].set_ylim(0, 6); ax[0, 1].set_xlabel(r"$h_{\rm dm}$ [pc]")
    ax[0, 1].set_ylabel(r"$\Delta$NLL$_{\rm total}$ [nats]")
    ax[0, 1].set_title("profile likelihood (re-fit flow at each $h$)")
    ax[0, 1].legend(fontsize=8)

    for a, samp, grid, marg, true, lab, unit in [
            (ax[1, 0], res["z0_samples"], res["zg"], res["mz"], res["true_z"], "z_0", "pc"),
            (ax[1, 1], res["w0_samples"], res["wg"], res["mw"], res["true_w"], "w_0", "km/s")]:
        a.hist(samp, bins=45, density=True, alpha=0.35, color="C0",
               label="back-integrated samples")
        a.plot(grid, true, "k--", lw=2, label="true $f_0$ marginal")
        a.plot(grid, marg, color="C3", lw=2, label="fitted $f_0$ marginal")
        a.set_xlabel(fr"${lab}$ [{unit}]"); a.set_ylabel("density"); a.legend(fontsize=8)
    ax[1, 0].set_title(r"$z_0$ marginal"); ax[1, 1].set_title(r"$w_0$ marginal")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def fitted_marginals(flow, zb, wb, pad=0.15, ng=220):
    """1-D marginals of a fitted 2-D density `flow` by grid integration.
    Returns (zg, m_z, wg, m_w)."""
    zlo, zhi = zb.min(), zb.max(); wlo, whi = wb.min(), wb.max()
    zg = np.linspace(zlo - (zhi - zlo) * pad, zhi + (zhi - zlo) * pad, ng)
    wg = np.linspace(wlo - (whi - wlo) * pad, whi + (whi - wlo) * pad, ng)
    ZG, WG = np.meshgrid(zg, wg)                       # shape (len(wg), len(zg))
    with torch.no_grad():
        p = np.exp(flow.log_prob(torch.tensor(ZG.ravel()),
                                 torch.tensor(WG.ravel())).numpy()).reshape(ZG.shape)
    dz, dw = zg[1] - zg[0], wg[1] - wg[0]
    return zg, p.sum(axis=0) * dw, wg, p.sum(axis=1) * dz


# ---- per-fit phase-space scatter (fitted-f0 RESIDUAL at t=0 and evolved) ----
def fit_all_models_for_scatter(z, w, true_pdf_z, true_pdf_w, n_show=4000,
                               ns=2000, seed=0):
    """Fit EACH model (analytic linear + every NVP and residual ladder member)
    to the observed cloud (z, w), then colour a per-star scatter by the RESIDUAL
    ``flow - true`` at t=0 and evolved to t_obs.

    Colouring by the fitted density alone (the old behaviour) makes the two
    panels IDENTICAL: by Liouville the phase density is conserved along each
    orbit, so the same colour just rides forward to the sheared cloud and nothing
    visibly changes between t=0 and t_obs.  The residual against the TRUE f_0 is
    what changes: it is a smooth structureless blob at t=0 but, because the truth
    evolves under H_TRUE while the flow value is conserved at the fit's own h,
    differential (anharmonic) phase mixing winds it into a spiral by t_obs.

    Each fit's ``h`` is the PROFILE-LIKELIHOOD minimum on the shared grid (the
    SAME estimator as run_profile_comparison / the PG profile cells), so the
    labelled h matches the upper fits.  (The previous version used joint_fit,
    whose h is under-converged and tracks its 250 pc init -> ~247 pc regardless
    of the model, which is why it disagreed with the profile-minimum h above.)

    Returns per model ``z0, w0`` (t=0 support of the fitted f_0), ``r0`` (t=0
    residual) and ``rt`` (t_obs residual).  `n_show` sub-samples the scatter; the
    fit still uses all stars."""
    def true2d(zz, ww):
        return true_pdf_z(np.asarray(zz)) * true_pdf_w(np.asarray(ww))

    rng = np.random.default_rng(seed)
    idx = (rng.choice(z.size, size=n_show, replace=False)
           if z.size > n_show else np.arange(z.size))
    zs, ws = z[idx], w[idx]
    N = z.size

    # profile grid + back-integrated clouds shared with run_profile_comparison,
    # so the fitted h here is the exact same profile minimum as the PG cells.
    hgrid = np.linspace(200.0, 300.0, 25)
    clouds = [backint(z, w, h) for h in hgrid]
    zi, wi = backint(z, w, 300.0, ns=1000)

    # TRUE f_0 evolved to t_obs at the shown stars: back-integrate at H_TRUE
    # (true dynamics) and read the true density there.  Same for every model.
    z0t, w0t = backint(zs, ws, H_TRUE, ns=ns)
    true_obs = true2d(z0t, w0t)

    def _resid(flow_dens_fn, h):
        z0, w0 = backint(zs, ws, h, ns=ns)            # t=0 support of fitted f_0
        fdens = flow_dens_fn(z0, w0)
        # flow density at the t_obs stars == fdens (Liouville-conserved), so the
        # t_obs residual is fdens minus the EVOLVED truth.
        return z0, w0, fdens - true2d(z0, w0), fdens - true_obs

    models = []

    # analytic linear = closed-form diagonal Gaussian at its profile minimum
    ap = _analytic_gauss_profile(z, w, hgrid, N)
    h_an = float(hgrid[int(ap.argmin())])
    zb, wb = backint(zs, ws, h_an, ns=ns)
    an_dens = gauss_pdf(zb, zb.mean(), zb.std()) * gauss_pdf(wb, wb.mean(), wb.std())
    models.append(dict(kind="analytic", label="linear 4p", n=4, h=h_an, z0=zb,
                       w0=wb, r0=an_dens - true2d(zb, wb), rt=an_dens - true_obs))

    # both NF kinds x every parameter count: profile-min h + a flow refit there
    for kind, ladder in [("NVP", nvp_ladder(zi, wi)),
                         ("residual", residual_ladder(zi, wi))]:
        for label, make, npar, lr in ladder:
            dnll = _profile_curve(make, z, w, clouds, hgrid, N, lr)
            k = int(dnll.argmin()); h = float(hgrid[k])
            flow = make(); _refit_once(flow, *clouds[k], lr)

            def _fd(z0, w0, flow=flow):
                with torch.no_grad():
                    return np.exp(flow.log_prob(torch.tensor(z0),
                                                torch.tensor(w0)).numpy())
            z0, w0, r0, rt = _resid(_fd, h)
            models.append(dict(kind=kind, label=label, n=npar, h=h,
                               z0=z0, w0=w0, r0=r0, rt=rt))
    return dict(id="PGscatter", models=models, z_obs=zs, w_obs=ws, h_true=H_TRUE)


def plot_phase_space_scatter(res, title=None, s=5):
    """Grid of RESIDUAL-coloured phase-space scatters, one ROW per fit, with the
    fitted-f_0 residual (flow - true) at t=0 (left) and evolved to t_obs (right).
    A diverging colour scale centred at 0 makes the sign of the misfit visible;
    the t=0 panel is a smooth blob, the t_obs panel a wound-up spiral.  `res` =
    output of fit_all_models_for_scatter."""
    import matplotlib.pyplot as plt
    if isinstance(res, str):
        raise TypeError("pass the dict from fit_all_models_for_scatter(z, w)")
    models = res["models"]; z, w = res["z_obs"], res["w_obs"]
    n = len(models)
    fig, ax = plt.subplots(n, 2, figsize=(11.5, 2.9 * n), squeeze=False)
    for i, m in enumerate(models):
        vmax = max(float(np.abs(m["r0"]).max()),
                   float(np.abs(m["rt"]).max())) or 1.0
        cols = [(m["z0"], m["w0"], m["r0"], r"$t=0$: flow $-$ true"),
                (z,        w,        m["rt"], r"$t=400$ Myr: flow $-$ true (evolved)")]
        sc = None
        for j, (zz, ww, rr, ttl) in enumerate(cols):
            o = np.argsort(np.abs(rr))                # large |residual| on top
            a = ax[i, j]
            sc = a.scatter(zz[o], ww[o], c=rr[o], s=s, cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, linewidths=0, rasterized=True)
            a.set_xlabel(r"$z$ [pc]")
            if i == 0:
                a.set_title(ttl, fontsize=11)
        fig.colorbar(sc, ax=ax[i, 1], fraction=0.046, pad=0.02,
                     label=r"flow $-$ true")
        ax[i, 0].set_ylabel(f'{m["kind"]} {m["label"]}\n'
                            fr'$h={m["h"]:.0f}$ pc' + "\n\n$w$ [km/s]", fontsize=9)
        ax[i, 1].set_ylabel(r"$w$ [km/s]")
    fig.suptitle(title or "Fitted-$f_0$ residual (flow $-$ true) of each fit — "
                 rf'truth $h={res["h_true"]:.0f}$ pc; '
                 "smooth at $t=0$, wound into a spiral at $t=400$ Myr",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


# ---- true & fitted f0 as overlaid FUNCTIONS + data, per fit -----------------
def fit_all_models_for_density(z, w, true_pdf_z, true_pdf_w, n_show=4000,
                               ng=120, pad=0.15, ns=2000, ns_grid=800, seed=0):
    """Fit EACH model (analytic linear + every NVP and residual ladder member)
    to the observed cloud, then evaluate -- as smooth FUNCTIONS on a grid -- both
    the TRUE f_0 and the FITTED f_0, at t=0 AND evolved to t_obs (t=400 Myr), plus
    the data points at both times, so a per-fit panel can overlay all three.

    The flow only ever yields a density, but the back-integration map is
    symplectic (Liouville, |det|=1), so the evolved density is just f_0 at the
    back-integrated pre-image: f_t(z,w) = f_0(backint_h(z,w)).  We evaluate the
    true f_0 there with the TRUE h and each fitted f_0 with that fit's own h.
    Each h is the PROFILE-LIKELIHOOD minimum on the shared grid (the SAME
    estimator as run_profile_comparison / the PG profile cells)."""
    def true2d(zz, ww):
        return true_pdf_z(np.asarray(zz)) * true_pdf_w(np.asarray(ww))

    rng = np.random.default_rng(seed)
    idx = (rng.choice(z.size, size=n_show, replace=False)
           if z.size > n_show else np.arange(z.size))
    zs, ws = z[idx], w[idx]
    N = z.size

    # profile grid + back-integrated clouds shared with run_profile_comparison,
    # so the fitted h here is the exact same profile minimum as the PG cells.
    hgrid = np.linspace(200.0, 300.0, 25)
    clouds = [backint(z, w, h) for h in hgrid]
    zi, wi = backint(z, w, 300.0, ns=1000)

    # t=0 grid (from the back-integrated data extent) and t_obs grid (from the
    # observed data extent).  Both true grids are model-independent.
    zb0, wb0 = backint(zs, ws, H_TRUE, ns=ns)
    zg0, wg0 = _axis_grid(zb0, pad, ng), _axis_grid(wb0, pad, ng)
    zgt, wgt = _axis_grid(zs, pad, ng), _axis_grid(ws, pad, ng)
    ZG0, WG0 = np.meshgrid(zg0, wg0)
    ZGt, WGt = np.meshgrid(zgt, wgt)

    true0 = true2d(ZG0, WG0)
    zbt, wbt = backint(ZGt.ravel(), WGt.ravel(), H_TRUE, ns=ns_grid)
    truet = true2d(zbt, wbt).reshape(ZGt.shape)

    def evolve_grid(dens0, h):
        """Density dens0 (defined at t=0) evaluated on the t_obs grid: by
        Liouville that is dens0 at the back-integrated pre-image."""
        z0, w0 = backint(ZGt.ravel(), WGt.ravel(), h, ns=ns_grid)
        return dens0(z0, w0).reshape(ZGt.shape)

    models = []

    # analytic linear = closed-form diagonal Gaussian at its profile minimum
    ap = _analytic_gauss_profile(z, w, hgrid, N)
    h_an = float(hgrid[int(ap.argmin())])
    zb, wb = backint(zs, ws, h_an, ns=ns)
    mz, sz, mw, sw = zb.mean(), zb.std(), wb.mean(), wb.std()
    dens_an = lambda zz, ww: gauss_pdf(zz, mz, sz) * gauss_pdf(ww, mw, sw)
    models.append(dict(kind="analytic", label="linear 4p", n=4, h=h_an,
                       z0=zb, w0=wb,
                       fit0=dens_an(ZG0, WG0), fitt=evolve_grid(dens_an, h_an)))

    # both NF kinds x every parameter count: profile-min h + a flow refit there
    for kind, ladder in [("NVP", nvp_ladder(zi, wi)),
                         ("residual", residual_ladder(zi, wi))]:
        for label, make, npar, lr in ladder:
            dnll = _profile_curve(make, z, w, clouds, hgrid, N, lr)
            k = int(dnll.argmin()); h = float(hgrid[k])
            flow = make(); _refit_once(flow, *clouds[k], lr)
            dens0 = _flow_density_fn(flow)
            fit0 = dens0(ZG0.ravel(), WG0.ravel()).reshape(ZG0.shape)
            z0, w0 = backint(zs, ws, h, ns=ns)
            models.append(dict(kind=kind, label=label, n=npar, h=h,
                               z0=z0, w0=w0,
                               fit0=fit0, fitt=evolve_grid(dens0, h)))

    return dict(id="PGdensity", models=models, zg0=zg0, wg0=wg0,
                zgt=zgt, wgt=wgt, true0=true0, truet=truet,
                z_obs=zs, w_obs=ws, h_true=H_TRUE)


def plot_phase_space_density(res, title=None, s=4, n_lev=5, zlim=None, wlim=None):
    """One ROW per fit, TWO columns (t=0 and t=400 Myr).  In EACH panel the TRUE
    f_0 (black solid contours) and the FITTED f_0 (red dashed contours) are drawn
    as overlaid density FUNCTIONS on the SAME shared contour levels, with the data
    points scattered underneath -- so true vs fit vs data is all on one plot.  The
    t=0 column shows f_0 itself; the t=400 Myr column shows both functions evolved
    (true with the true h, fit with its own h), where differential phase mixing
    winds them up.  `res` = the unified run_profile_comparison result.

    `zlim`/`wlim` = (lo, hi) axis windows applied to every panel, to zoom into the
    central structure (corners carry little information).  Default = full grid."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if isinstance(res, str):
        raise TypeError("pass the dict from fit_all_models_for_density(z, w)")
    # the 2-D panel would be unwieldy with the whole ladder stacked, so show only
    # the featured subset here (the 1-D marginal panel shows every fit)
    models = [m for m in res["models"] if m.get("featured", True)]
    zg0, wg0, zgt, wgt = res["zg0"], res["wg0"], res["zgt"], res["wgt"]
    true0, truet = res["true0"], res["truet"]
    zo, wo = res["z_obs"], res["w_obs"]
    lv = np.linspace(0.1, 0.9, n_lev)
    n = len(models)
    fig, ax = plt.subplots(n, 2, figsize=(11.5, 3.4 * n), squeeze=False)
    for i, m in enumerate(models):
        cols = [(zg0, wg0, true0, m["fit0"], m["z0"], m["w0"], r"$t=0$"),
                (zgt, wgt, truet, m["fitt"], zo, wo, r"$t=400$ Myr")]
        for j, (zg, wg, tg, fg, zd, wd, ttl) in enumerate(cols):
            a = ax[i, j]
            a.scatter(zd, wd, s=s, c="0.55", alpha=0.25, linewidths=0,
                      rasterized=True, zorder=1)
            levels = float(tg.max()) * lv
            a.contour(zg, wg, tg, levels=levels, colors="k",
                      linewidths=1.3, zorder=3)
            a.contour(zg, wg, fg, levels=levels, colors="crimson",
                      linewidths=1.3, linestyles="--", zorder=4)
            a.set_xlim(*(zlim if zlim is not None else (zg[0], zg[-1])))
            a.set_ylim(*(wlim if wlim is not None else (wg[0], wg[-1])))
            a.set_xlabel(r"$z$ [pc]")
            if i == 0:
                a.set_title(ttl, fontsize=11)
        ax[i, 0].set_ylabel(f'{m["kind"]} {m["label"]}\n'
                            fr'$h={m["h"]:.0f}$ pc' + "\n\n$w$ [km/s]", fontsize=9)
        ax[i, 1].set_ylabel(r"$w$ [km/s]")
    handles = [Line2D([0], [0], color="k", lw=1.3, label=r"true $f_0$"),
               Line2D([0], [0], color="crimson", lw=1.3, ls="--",
                      label=r"fitted $f_0$"),
               Line2D([0], [0], marker="o", color="none", markerfacecolor="0.55",
                      markersize=5, label="data")]
    ax[0, 0].legend(handles=handles, loc="upper right", fontsize=8,
                    framealpha=0.9)
    fig.suptitle(title or "True (black) vs fitted (red) $f_0$ with data, per fit "
                 rf'— truth $h={res["h_true"]:.0f}$ pc; f_0 at $t=0$ and evolved '
                 "to $t=400$ Myr", fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


def plot_phase_space_density_legend():
    """Standalone key for plot_phase_space_density: which contour is the TRUE
    f_0, which is the FITTED f_0, and what the points are.  Draw it in its own
    cell so the meaning of the colours/linestyles is unambiguous."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="k", lw=1.6,
               label=r"true $f_0$  (black solid contours)"),
        Line2D([0], [0], color="crimson", lw=1.6, ls="--",
               label=r"fitted $f_0$  (red dashed contours)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.55",
               markersize=6, label="data (stars)"),
    ]
    fig, ax = plt.subplots(figsize=(5.2, 1.4))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", fontsize=11, frameon=True,
              title="Key for the True-vs-fitted $f_0$ figure")
    fig.tight_layout()
    return fig


# ---- residual of true f0 vs fitted flow, at t=0 and evolved to t_obs --------
def _axis_grid(x, pad, ng, q=None):
    """Grid spanning the data with a `pad` fractional margin.  If `q` is given the
    extent is the [q, 100-q] percentile band instead of the raw min/max, so a few
    extreme wing stars do not blow the window out -- this concentrates grid
    resolution on the central bulk (matters for the heavy-tailed multi-width f0)."""
    if q is None:
        lo, hi = float(np.min(x)), float(np.max(x))
    else:
        lo, hi = (float(v) for v in np.percentile(x, [q, 100.0 - q]))
    m = (hi - lo) * pad
    return np.linspace(lo - m, hi + m, ng)


# ---- per-fit TRUE vs FITTED 1-D marginals, at t=0 and evolved to t_obs -------
def _flow_density_fn(flow):
    """A numpy density callable dens(z0, w0) -> array for a fitted flow."""
    def dens(z0, w0):
        with torch.no_grad():
            return np.exp(flow.log_prob(torch.tensor(np.asarray(z0, dtype=np.float64)),
                                        torch.tensor(np.asarray(w0, dtype=np.float64))).numpy())
    return dens


def _grid_marginals(dens, zg, wg):
    """1-D z- and w-marginals of a 2-D density callable on the grid zg x wg."""
    ZG, WG = np.meshgrid(zg, wg)                       # shape (len(wg), len(zg))
    p = dens(ZG.ravel(), WG.ravel()).reshape(ZG.shape)
    return p.sum(0) * (wg[1] - wg[0]), p.sum(1) * (zg[1] - zg[0])


def _evolved_marginals(dens, zg, wg, h, ns):
    """Marginals of the density `dens` (defined at t=0) EVOLVED to t_obs: by
    Liouville the evolved density is dens(backint_h(z, w)), so back-integrate the
    t_obs grid to t=0 at scale height h and evaluate dens there."""
    def edens(z, w):
        z0, w0 = backint(np.asarray(z), np.asarray(w), h, ns=ns)
        return dens(z0, w0)
    return _grid_marginals(edens, zg, wg)


def fit_all_models_distributions(z, w, true_pdf_z, true_pdf_w, inits=250.0,
                                 ng=120, ns=1200, pad=0.15, seed=0):
    """For EVERY model (analytic linear + each NVP and residual ladder member):
    joint-fit (f_0, h), then compute the z- and w-marginals of the TRUE f_0 and
    the FITTED f_0, both at t=0 AND evolved to t_obs.  Returns a dict with a
    per-model list of marginals plus shared true / data curves for overlay."""
    def true2d(Z, W):
        return true_pdf_z(np.asarray(Z)) * true_pdf_w(np.asarray(W))

    # shared grids: t=0 from the H_TRUE back-integrated cloud, t_obs from the data
    z0d, w0d = backint(z, w, H_TRUE)
    zg0, wg0 = _axis_grid(z0d, pad, ng), _axis_grid(w0d, pad, ng)
    zgt, wgt = _axis_grid(z, pad, ng), _axis_grid(w, pad, ng)
    true_z0, true_w0 = true_pdf_z(zg0), true_pdf_w(wg0)
    # truth evolved to t_obs (same for every model -> compute once)
    true_zt, true_wt = _evolved_marginals(true2d, zgt, wgt, H_TRUE, ns)

    def _pack(kind, label, npar, h_fit, dens):
        fz0, fw0 = _grid_marginals(dens, zg0, wg0)
        fzt, fwt = _evolved_marginals(dens, zgt, wgt, h_fit, ns)
        return dict(kind=kind, label=label, n=npar, h=h_fit,
                    fit_z0=fz0, fit_w0=fw0, fit_zt=fzt, fit_wt=fwt)

    models = []

    # analytic linear = closed-form diagonal Gaussian at its profile-min h
    idx = (np.random.default_rng(seed).choice(z.size, 6000, replace=False)
           if z.size > 6000 else np.arange(z.size))
    hgrid = np.linspace(200.0, 320.0, 19)
    ap = _analytic_gauss_profile(z[idx], w[idx], hgrid, idx.size)
    h_an = float(hgrid[int(ap.argmin())])
    zb, wb = backint(z, w, h_an)
    dens_an = (lambda z0, w0, m=(zb.mean(), zb.std(), wb.mean(), wb.std()):
               gauss_pdf(np.asarray(z0), m[0], m[1]) * gauss_pdf(np.asarray(w0), m[2], m[3]))
    models.append(_pack("analytic", "linear 4p", 4, h_an, dens_an))

    # both NF kinds x every parameter count
    zi, wi = backint(z, w, 300.0, ns=1000)
    for kind, ladder in [("NVP", nvp_ladder(zi, wi)),
                         ("residual", residual_ladder(zi, wi))]:
        for label, make, npar, lr in ladder:
            flow, h_fit, _ = joint_fit(make, z, w, inits, lr_flow=lr, seed=seed)
            models.append(_pack(kind, label, npar, h_fit, _flow_density_fn(flow)))

    return dict(id="PGdist", models=models, h_true=H_TRUE,
                zg0=zg0, wg0=wg0, true_z0=true_z0, true_w0=true_w0,
                zgt=zgt, wgt=wgt, true_zt=true_zt, true_wt=true_wt,
                z0_data=z0d, w0_data=w0d, z_data=z, w_data=w)


def plot_all_models_distributions(res, title=None):
    """Grid of true-vs-fitted 1-D marginals, one ROW per fit and four columns:
    (z, w) at t=0 and (z, w) evolved to t_obs.  True f_0 dashed, fitted solid,
    data histogram faint."""
    import matplotlib.pyplot as plt
    models = res["models"]
    n = len(models)
    fig, ax = plt.subplots(n, 4, figsize=(17, 2.7 * n), squeeze=False)
    col = [("zg0", "true_z0", "fit_z0", "z0_data", r"$z_0$ [pc]", r"$z$ marginal, $t=0$"),
           ("wg0", "true_w0", "fit_w0", "w0_data", r"$w_0$ [km/s]", r"$w$ marginal, $t=0$"),
           ("zgt", "true_zt", "fit_zt", "z_data",  r"$z$ [pc]",    r"$z$ marginal, $t=400$ Myr"),
           ("wgt", "true_wt", "fit_wt", "w_data",  r"$w$ [km/s]",  r"$w$ marginal, $t=400$ Myr")]
    for i, m in enumerate(models):
        for j, (gk, tk, fk, dk, xl, ttl) in enumerate(col):
            a = ax[i, j]
            a.hist(res[dk], bins=60, density=True, color="0.75", alpha=0.6,
                   label="data" if (i == 0 and j == 0) else None)
            a.plot(res[gk], res[tk], "k--", lw=1.8,
                   label="true $f_0$" if (i == 0 and j == 0) else None)
            a.plot(res[gk], m[fk], color="C3", lw=1.8,
                   label="fitted $f_0$" if (i == 0 and j == 0) else None)
            a.set_xlabel(xl)
            if i == 0:
                a.set_title(ttl, fontsize=10)
        ax[i, 0].set_ylabel(f'{m["kind"]} {m["label"]}\n'
                            fr'$h={m["h"]:.0f}$ pc' + "\n\ndensity", fontsize=9)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(title or "True vs fitted $f_0$ marginals per fit — "
                 rf'$t=0$ and evolved to $t=400$ Myr (truth $h={res["h_true"]:.0f}$ pc)',
                 fontsize=13, y=1.0)
    fig.tight_layout()
    return fig


def fit_spiral_residual(z, w, true_pdf_z, true_pdf_w, kind="NVP", size=1,
                        h_init=250.0, ng=160, ns=1500, pad=0.30, seed=0):
    """Fit a flow f_0 to the observed cloud (z, w), then evaluate the 2-D TRUE and
    FLOW densities on a common grid at t=0 and at t_obs, plus their residuals.

    At t=0 the true f_0 and the fitted flow are compared directly (they should
    nearly coincide -- a structureless residual).  At t_obs the truth is evolved
    with the true dynamics (H_TRUE) and the model with its FITTED h, both by the
    density-conserving back-integration map, so any t=0 misfit -- and any h
    offset -- is wound up by differential (anharmonic) phase mixing into a
    SPIRAL residual in the (z, w) plane.

    kind in {'NVP','residual'}; size = ladder index 0..3 (smaller = stiffer flow
    -> a more coherent, cleaner spiral)."""
    def true2d(Z, W):
        return true_pdf_z(Z) * true_pdf_w(W)

    zi, wi = backint(z, w, 300.0, ns=1000)
    ladder = nvp_ladder(zi, wi) if kind == "NVP" else residual_ladder(zi, wi)
    label, make, npar, lr = ladder[size]
    flow, h_fit, _ = joint_fit(make, z, w, h_init, lr_flow=lr, seed=seed)

    def _flow_dens(zz, ww, shape):
        with torch.no_grad():
            return np.exp(flow.log_prob(torch.tensor(zz),
                                        torch.tensor(ww)).numpy()).reshape(shape)

    # t=0: grid on the (H_TRUE) back-integrated cloud; no integration needed
    z0d, w0d = backint(z, w, H_TRUE)
    zg0, wg0 = _axis_grid(z0d, pad, ng), _axis_grid(w0d, pad, ng)
    ZG0, WG0 = np.meshgrid(zg0, wg0)
    true0 = true2d(ZG0, WG0)
    flow0 = _flow_dens(ZG0.ravel(), WG0.ravel(), ZG0.shape)

    # t=obs: grid on the observed cloud; back-integrate to t=0 for each density
    zgt, wgt = _axis_grid(z, pad, ng), _axis_grid(w, pad, ng)
    ZGt, WGt = np.meshgrid(zgt, wgt)
    zr, wr = ZGt.ravel(), WGt.ravel()
    z0T, w0T = backint(zr, wr, H_TRUE, ns=ns)          # truth: true dynamics
    trueT = true2d(z0T, w0T).reshape(ZGt.shape)
    z0F, w0F = backint(zr, wr, h_fit, ns=ns)           # model: fitted h
    flowT = _flow_dens(z0F, w0F, ZGt.shape)

    return dict(id="spiral", kind=kind, label=label, n=npar, h_fit=h_fit,
                h_true=H_TRUE, zg0=zg0, wg0=wg0, true0=true0, flow0=flow0,
                zgt=zgt, wgt=wgt, trueT=trueT, flowT=flowT, z_obs=z, w_obs=w)


def _select_model(res, kind="NVP", size=0):
    """Pick one fitted model from a unified run_profile_comparison result: the
    `size`-th ladder member (0 = stiffest) of the requested `kind`."""
    matches = [m for m in res["models"] if m["kind"] == kind]
    if not matches:
        raise ValueError(f"no {kind!r} models in result; kinds present: "
                         f"{sorted({m['kind'] for m in res['models']})}")
    return matches[min(size, len(matches) - 1)]


def plot_spiral_residual(res, kind="NVP", size=0, title=None, zlim=None, wlim=None):
    """2x3 grid: rows = (t=0, t_obs); columns = (true vs flow overlay, FITTED
    phase-space density heatmap, signed residual flow-true).  The t_obs row shows
    the fitted DF -- and the residual -- wound into a phase-space spiral.

    Consumes the SINGLE-fit result from run_profile_comparison (X.run) and simply
    SELECTS one already-fitted model (`kind`, ladder index `size`); it does not
    re-fit.  size=0 is the stiffest flow (cleanest spiral).  `zlim`/`wlim` =
    (lo, hi) windows to zoom the heatmaps into the central structure."""
    import matplotlib.pyplot as plt
    m = _select_model(res, kind, size)
    d = dict(zg0=res["zg0"], wg0=res["wg0"], true0=res["true0"], flow0=m["fit0"],
             zgt=res["zgt"], wgt=res["wgt"], trueT=res["truet"], flowT=m["fitt"],
             kind=m["kind"], label=m["label"], h_fit=m["h"], h_true=res["h_true"])
    res = d
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    rows = [(r"$t=0$ (initial DF)", "zg0", "wg0", "true0", "flow0",
             r"$z_0$ [pc]", r"$w_0$ [km/s]"),
            (r"$t=400$ Myr (observed frame)", "zgt", "wgt", "trueT", "flowT",
             r"$z$ [pc]", r"$w$ [km/s]")]
    for i, (rlab, zk, wk, tk, fk, xl, yl) in enumerate(rows):
        zg, wg, true, flow = res[zk], res[wk], res[tk], res[fk]
        resid = flow - true

        a = ax[i, 0]
        a.pcolormesh(zg, wg, true, cmap="Greys", shading="auto")
        a.contour(zg, wg, flow, levels=7, colors="C3", linewidths=0.9)
        a.set_title(f"{rlab}\ntrue $f_0$ (grey) vs flow (red)")

        a = ax[i, 1]
        im = a.pcolormesh(zg, wg, flow, cmap="viridis", vmin=0.0,
                          shading="gouraud", rasterized=True)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label="density")
        a.set_title("fitted phase-space DF (flow)")

        a = ax[i, 2]
        m = float(np.abs(resid).max()) or 1.0
        im = a.pcolormesh(zg, wg, resid, cmap="RdBu_r", vmin=-m, vmax=m, shading="auto")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        a.set_title("residual  flow $-$ true")

        for j in range(3):
            ax[i, j].set_xlabel(xl); ax[i, j].set_ylabel(yl)
            if zlim is not None:
                ax[i, j].set_xlim(*zlim)
            if wlim is not None:
                ax[i, j].set_ylim(*wlim)
    fig.suptitle(title or (f'Phase-space residual: true $f_0$ vs {res["kind"]} '
                 f'{res["label"]} flow  '
                 rf'($h_{{\rm fit}}={res["h_fit"]:.0f}$, truth {res["h_true"]:.0f} pc)'
                 "\nsmooth at $t=0$ → wound into a spiral at $t=400$ Myr"),
                 fontsize=13)
    fig.tight_layout()
    return fig


# ==========================================================================
# Profile-likelihood comparison across NF KINDS and parameter counts.
#   Kind 1: NVP / affine-coupling flows (RealNVP family).
#   Kind 2: residual flows  y -> y + V tanh(W y + b)  (non-coupling; the exact
#           2x2 Jacobian determinant is used since dim=2).  Width m scales the
#           parameter count (5*m per layer) cleanly from a handful to ~1000.
#   Plus an analytic linear bijection (diagonal Gaussian, closed-form profile).
# ==========================================================================
from experiments.exp_B_tiny_nn import TinyCoupling as _TinyCoupling


class TinyNVP(nn.Module):
    """K affine-coupling layers with Linear(1,2) heads (4 params each) + fixed
    standardisation.  K=1 -> 4 params, K=2 -> 8 params."""
    def __init__(self, K):
        super().__init__()
        self.layers = nn.ModuleList([_TinyCoupling(flip=(i % 2 == 1)) for i in range(K)])
        self.register_buffer("mu", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("sig", torch.ones(2, dtype=torch.float64))

    def set_standardization(self, z, w):
        self.mu = torch.tensor([float(np.mean(z)), float(np.mean(w))], dtype=torch.float64)
        self.sig = torch.tensor([float(np.std(z)), float(np.std(w))], dtype=torch.float64)

    def log_prob(self, z, w):
        xy = (torch.stack([z, w], 1) - self.mu) / self.sig
        ld = xy.new_zeros(xy.shape[0])
        for L in self.layers:
            xy, d = L(xy); ld = ld + d
        lb = -0.5 * (xy ** 2).sum(1) - math.log(2 * math.pi)
        return lb + ld - torch.log(self.sig).sum()


class ResidualFlow(nn.Module):
    """Single residual layer  y -> y + V tanh(W y + b)  in the data->base
    direction, on standardised coords.  Non-coupling architecture; width m gives
    5*m trainable parameters.  Exact 2x2 Jacobian determinant (dim=2)."""
    def __init__(self, m):
        super().__init__()
        self.W = nn.Parameter(0.1 * torch.randn(m, 2, dtype=torch.float64))
        self.V = nn.Parameter(0.1 * torch.randn(2, m, dtype=torch.float64))
        self.b = nn.Parameter(torch.zeros(m, dtype=torch.float64))
        self.register_buffer("mu", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("sig", torch.ones(2, dtype=torch.float64))

    def set_standardization(self, z, w):
        self.mu = torch.tensor([float(np.mean(z)), float(np.mean(w))], dtype=torch.float64)
        self.sig = torch.tensor([float(np.std(z)), float(np.std(w))], dtype=torch.float64)

    def log_prob(self, z, w):
        y = (torch.stack([z, w], 1) - self.mu) / self.sig      # (N,2)
        a = y @ self.W.T + self.b                              # (N,m)
        t = torch.tanh(a)
        y2 = y + t @ self.V.T                                  # (N,2)
        hp = 1.0 - t ** 2                                      # (N,m)
        VW = torch.einsum('ik,nk,kj->nij', self.V, hp, self.W)  # (N,2,2)
        J = VW + torch.eye(2, dtype=torch.float64)
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        logdet = torch.log(torch.abs(det) + 1e-8)
        lb = -0.5 * (y2 ** 2).sum(1) - math.log(2 * math.pi)
        return lb + logdet - torch.log(self.sig).sum()


def _std_maker(ctor, zi, wi):
    def make():
        f = ctor(); f.set_standardization(zi, wi); return f
    return make


def _diag_affine_maker():
    """Maker for the 4-param diagonal-affine flow x=a*u+b (exact for a Gaussian;
    its profile is the closed-form analytic).  No standardisation buffer."""
    from experiments.exp_A_linear_bijection import DiagonalAffineFlow
    return DiagonalAffineFlow(a_init=(Z0_SIGMA, W0_SIGMA)).double()


def rigid_ladder(zi, wi):
    """(label, make_flow, n_params, lr) for the RIGID maps that reliably recover h
    with a resolved well: a 4-param diagonal-affine (exact Gaussian) plus small
    ZERO-init tiny-coupling flows TinyNVP(K) = 4K params (K=2,4,8 -> 8,16,32 p).
    These start at identity and cannot overfit the phase-mixed structure, so their
    profile-likelihood minima sit at the truth (unlike the flexible RealNVP)."""
    lad = [("affine 4p", _diag_affine_maker, 4, 3e-2)]
    for K in (2, 4, 8):
        lad.append((f"{4*K}p", _std_maker(lambda K=K: TinyNVP(K).double(), zi, wi),
                    4 * K, 5e-3))
    return lad


def flexible_ladder(zi, wi, configs=None):
    """(label, make_flow, n_params, lr) for the FLEXIBLE full-RealNVP flows
    (ZWNormalizingFlow, random-init MLP heads) -- shown to DEMONSTRATE that
    flexibility overfits the finite sample and loosens/edge-pins h.  `configs` is a
    list of (n_couplings, hidden) tuples; default ~76 / 304 / 1014 params.  The
    non-Gaussian notebooks pass a longer list to probe the 100-5000 param range
    (where a flexible flow can start to fit the true shape but over-fits h more)."""
    from fitter import ZWNormalizingFlow
    configs = [(2, 4), (8, 4), (3, 16)] if configs is None else configs
    lad = []
    for c, h in configs:
        npar = sum(p.numel() for p in ZWNormalizingFlow(c, h).parameters())
        lad.append((f"{npar}p",
                    _std_maker(lambda c=c, h=h: ZWNormalizingFlow(c, h).double(), zi, wi),
                    npar, 1e-3))
    lad.sort(key=lambda t: t[2])                        # ascending #params
    return lad


# kept for backward-compatibility with any old callers (unused by the notebooks)
def nvp_ladder(zi, wi):
    return rigid_ladder(zi, wi)[1:] + flexible_ladder(zi, wi)


def residual_ladder(zi, wi):
    return [(f"{5*m}p", _std_maker(lambda m=m: ResidualFlow(m).double(), zi, wi), 5 * m, 3e-4)
            for m in (2, 12, 60)]


def _profile_curve(make, z, w, clouds, hgrid, N, lr, warm=800, epochs=250, seed=0):
    """The textbook procedure: JOINT-fit (h, Omega), then PROFILE outward from it.

    1. `joint_fit` gradient-descends h and Omega together -> the central h_hat and
       its Omega_hat (the joint MLE).
    2. Starting from Omega_hat at the grid point nearest h_hat, step h outward in
       BOTH directions, re-optimising Omega at each fixed h (warm-started from the
       neighbour), giving the profile NLL_p(h) = min_Omega NLL(h, Omega).

    Profiling *outward from the joint optimum* keeps Omega in the correct basin, so
    the well is centred on h_hat and the flexible flows no longer slide into the
    over-fit grid-EDGE basins (the old up-from-200 / down-from-300 min-envelope did).
    Returns Delta-NLL(h) on `hgrid`: its argmin is h_hat and the Delta-NLL=0.5
    half-width is the 1-sigma uncertainty.  `seed` fixes the flow init (reproducible)."""
    import copy
    torch.manual_seed(seed)
    hgrid = np.asarray(hgrid)

    def refit(flow, zb, wb, ep):
        opt = torch.optim.Adam(flow.parameters(), lr=lr)
        zt, wt = torch.tensor(zb), torch.tensor(wb)
        for _ in range(ep):
            opt.zero_grad(); loss = -flow.log_prob(zt, wt).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0); opt.step()
        with torch.no_grad():
            return float((-flow.log_prob(zt, wt)).mean())

    # 1. joint MLE of (h, Omega) -> central h_hat + Omega_hat (subsample is fine
    #    here: only used to locate the basin and warm-start the profile)
    jflow, h_hat, _ = joint_fit(make, z, w, 250.0, n_epochs=300,
                                n_sub=min(N, 3000), lr_flow=lr, seed=seed)
    state = copy.deepcopy(jflow.state_dict())
    k0 = int(np.argmin(np.abs(hgrid - h_hat)))

    # 2. profile OUTWARD from Omega_hat at k0, re-optimising Omega at each fixed h
    nll = np.full(len(hgrid), np.nan)
    fu = make(); fu.load_state_dict(state); nll[k0] = N * refit(fu, *clouds[k0], warm)
    for i in range(k0 + 1, len(hgrid)):                      # sweep up
        nll[i] = N * refit(fu, *clouds[i], epochs)
    fd = make(); fd.load_state_dict(state); refit(fd, *clouds[k0], warm)
    for i in range(k0 - 1, -1, -1):                          # sweep down
        nll[i] = N * refit(fd, *clouds[i], epochs)
    return nll - nll.min()


def _analytic_gauss_profile(z, w, hgrid, N):
    """Closed-form profile for the analytic linear (diagonal-Gaussian) bijection."""
    nll = np.array([(lambda c: N * (math.log(2 * math.pi) + math.log(c[0].std())
                                    + math.log(c[1].std()) + 1.0))(backint(z, w, h))
                    for h in hgrid])
    return nll - nll.min()


def run_profile_comparison(tid, z, w, true_pdf_z, true_pdf_w, include_analytic=True,
                           w_components=None, hgrid=None,
                           corr_inits=(160.0, 250.0, 340.0),
                           ng=100, pad=0.2, ns_grid=500, n_show=4000, seed=0,
                           n_feat=3, grid_q=1.0, cloud_ns=2000, flex_configs=None):
    """SINGLE fit per model, then every view derived from that one fit.

    For each model (analytic linear + each NVP and residual ladder member):

      1. profile h_dm -- at each h on the grid the bijection params phi are
         re-optimised, giving NLL_p(h) = min_phi NLL(h, phi).  This is exactly
         "jointly fit (h, phi), then vary h near the minimum re-fitting phi to
         map the uncertainty" (Wilks Delta-NLL = 0.5).
      2. h_best = argmin NLL_p(h)  (the joint MLE of h);
      3. refit phi ONCE at h_best -> the single best-fit flow;
      4. from THAT one flow, evaluate every downstream view -- 1-D marginals and
         2-D density, at t=0 AND evolved to t_obs -- plus the init-guess -> final-h
         correlation.

    So the profile plot, the init-guess map, the 1-D distribution panels, the 2-D
    phase-space-density panels and the spiral residual all read the SAME fit; no
    plot re-fits.  Returns one dict consumed by every plot_* below."""
    def true2d(Z, W):
        return true_pdf_z(np.asarray(Z)) * true_pdf_w(np.asarray(W))

    N = z.size
    zi, wi = backint(z, w, 300.0, ns=1000)
    hgrid = np.linspace(200.0, 300.0, 25) if hgrid is None else np.asarray(hgrid)
    clouds = [backint(z, w, h, ns=cloud_ns) for h in hgrid]

    # shared display grids: t=0 from the H_TRUE back-integrated cloud, t_obs from
    # the observed data.  All true curves live on these grids (model-independent).
    rng = np.random.default_rng(seed)
    idx = (rng.choice(N, size=n_show, replace=False) if N > n_show else np.arange(N))
    zs, ws = z[idx], w[idx]
    zb0, wb0 = backint(z, w, H_TRUE)                         # full cloud -> grid extent
    zg0, wg0 = _axis_grid(zb0, pad, ng, grid_q), _axis_grid(wb0, pad, ng, grid_q)
    zgt, wgt = _axis_grid(z, pad, ng, grid_q), _axis_grid(w, pad, ng, grid_q)
    ZG0, WG0 = np.meshgrid(zg0, wg0)
    ZGt, WGt = np.meshgrid(zgt, wgt)
    dz0, dw0 = zg0[1] - zg0[0], wg0[1] - wg0[0]
    dzt, dwt = zgt[1] - zgt[0], wgt[1] - wgt[0]

    true0 = true2d(ZG0, WG0)
    zbt, wbt = backint(ZGt.ravel(), WGt.ravel(), H_TRUE, ns=ns_grid)
    truet = true2d(zbt, wbt).reshape(ZGt.shape)              # truth evolved (true h)
    true_z0, true_w0 = true0.sum(0) * dw0, true0.sum(1) * dz0
    true_zt, true_wt = truet.sum(0) * dwt, truet.sum(1) * dzt

    def eval_model(dens0, h, kind, label, npar, dnll, corr):
        """Every view of ONE fitted density dens0 at its own h."""
        fit0 = dens0(ZG0.ravel(), WG0.ravel()).reshape(ZG0.shape)
        z0e, w0e = backint(ZGt.ravel(), WGt.ravel(), h, ns=ns_grid)   # Liouville pre-image
        fitt = dens0(z0e, w0e).reshape(ZGt.shape)
        zb, wb = backint(zs, ws, h)                          # t=0 support scatter
        return dict(kind=kind, label=label, n=npar, h=h, dnll=dnll, corr=corr,
                    fit0=fit0, fitt=fitt, z0=zb, w0=wb,
                    fit_z0=fit0.sum(0) * dw0, fit_w0=fit0.sum(1) * dz0,
                    fit_zt=fitt.sum(0) * dwt, fit_wt=fitt.sum(1) * dzt)

    def featured_indices(n, k):
        """`k` representative ladder positions: smallest, largest, and (k-2) evenly
        spaced in between -- so the per-row panels span the #-parameter range."""
        if k >= n:
            return list(range(n))
        return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})

    kinds = {}
    models = []
    for kind_name, short, ladder in [
            ("rigid maps", "rigid", rigid_ladder(zi, wi)),
            ("flexible RealNVP", "flex", flexible_ladder(zi, wi, flex_configs))]:
        curves = []
        feat = featured_indices(len(ladder), n_feat)         # subset for the 2-D panel & init-map
        for j, (label, make, npar, lr) in enumerate(ladder):
            dnll = _profile_curve(make, z, w, clouds, hgrid, N, lr, seed=seed)  # joint+profile
            k = int(dnll.argmin()); h_best = float(hgrid[k]) # step 2: h_best
            is_feat = j in feat
            # the init-guess correlation re-fits (joint_fit) per start -> expensive,
            # so compute it only for ONE representative member per kind (the smallest,
            # feat[0]); the init-map then contrasts one rigid vs one flexible flow.
            # Every member is still refit once + evaluated (1-D marginal panel shows
            # the WHOLE ladder; 2-D panel shows the featured subset).
            corr = ([(hi, C_joint(make, z, w, hi, lr)) for hi in corr_inits]
                    if j == feat[0] else None)
            flow = make(); _refit_once(flow, *clouds[k], lr)  # steps 3-4 (ALL)
            m = eval_model(_flow_density_fn(flow), h_best, short, label, npar, dnll, corr)
            m["featured"] = is_feat
            models.append(m)
            curves.append(dict(label=label, n=npar, lr=lr, dnll=dnll, corr=corr,
                               zg=zg0, wg=wg0, mz=m["fit_z0"], mw=m["fit_w0"]))
        kinds[kind_name] = curves

    analytic = None
    if include_analytic:
        analytic = _analytic_gauss_profile(z, w, hgrid, N)
        h_an = float(hgrid[int(analytic.argmin())])
        zb, wb = backint(z, w, h_an)
        m_an = (zb.mean(), zb.std(), wb.mean(), wb.std())
        dens_an = (lambda zz, ww, m=m_an: gauss_pdf(np.asarray(zz), m[0], m[1])
                   * gauss_pdf(np.asarray(ww), m[2], m[3]))
        am = eval_model(dens_an, h_an, "analytic", "linear 4p", 4, analytic, None)
        am["featured"] = True
        models.insert(0, am)                                 # analytic = first row

    comp_w = ([(float(wt), gauss_pdf(wg0, mu, s)) for wt, mu, s in w_components]
              if w_components else None)
    return dict(id=tid, hgrid=hgrid, kinds=kinds, analytic=analytic, h_true=H_TRUE,
                corr_inits=list(corr_inits), z0_samples=zb0, w0_samples=wb0,
                zg=zg0, wg=wg0, true_z=true_z0, true_w=true_w0, comp_w=comp_w,
                # unified per-model views + shared grids (dist / density / spiral)
                models=models, zg0=zg0, wg0=wg0, zgt=zgt, wgt=wgt,
                true0=true0, truet=truet, true_z0=true_z0, true_w0=true_w0,
                true_zt=true_zt, true_wt=true_wt,
                z0_data=zb0, w0_data=wb0, z_data=z, w_data=w,
                z_obs=zs, w_obs=ws)


def C_joint(make, z, w, h_init, lr, n_epochs=200, ns=800, n_sub=2500):
    """Joint fit endpoint h for the init-guess -> fitted-h map.  Must be CONVERGED
    (enough epochs): h enters only anharmonically so its gradient is weak and it
    moves slowly -- too few epochs and the endpoint just tracks h_init (a spurious
    slope-1 line).  With enough epochs the slope collapses to 0 (h -> truth from
    any start), which is the real convergence signature."""
    return joint_fit(make, z, w, h_init, n_epochs=n_epochs, ns=ns, n_sub=n_sub,
                     lr_flow=lr)[1]


def _refit_once(flow, zb, wb, lr, epochs=1200):
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    zt, wt = torch.tensor(zb), torch.tensor(wb)
    for _ in range(epochs):
        opt.zero_grad(); loss = -flow.log_prob(zt, wt).mean(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0); opt.step()


def fitted_h_summary(res):
    """Text table of the fitted h_dm for every model: profile minimum with a
    1-sigma error bar from the Delta-NLL = 0.5 profile half-width (Wilks)."""
    hg = res["hgrid"]

    def row(label, d):
        hm = hg[int(d.argmin())]
        lo, hi = cross(hg, d, 0.5, -1), cross(hg, d, 0.5, +1)
        sig = 0.5 * (hi - lo)
        grid_lim = (lo <= hg[0] + 1e-6) or (hi >= hg[-1] - 1e-6)
        flat = d.max() < 0.5                       # never rises to 1 sigma in-window
        if flat:
            note = "  <- flat (well < 0.5 nat): h unconstrained"
        elif hm <= hg[1] or hm >= hg[-2]:
            note = "  <- min at grid edge (true min outside %.0f-%.0f)" % (hg[0], hg[-1])
        elif grid_lim:
            note = "  (sigma grid-limited: lower bound)"
        else:
            note = ""
        return f"     {label:<8}: h_dm = {hm:3.0f} +/- {sig:2.0f} pc   (well {d.max():5.0f} nats){note}"

    lines = [f"Fitted h_dm = profile minimum +/- 1sigma (Delta NLL = 0.5)   "
             f"[truth = {res['h_true']:.0f} pc, grid {hg[0]:.0f}-{hg[-1]:.0f} pc]"]
    if res["analytic"] is not None:
        lines.append(row("analytic", res["analytic"]))
    for name in res["kinds"]:
        lines.append(f"  {name}:")
        for c in res["kinds"][name]:
            lines.append(row(c["label"], c["dnll"]))
    return "\n".join(lines)


# ---- per-axis drawing helpers (shared by the combined and the split plots) ---
def _draw_profile_axes(res, axmap):
    """Draw the profile-likelihood Delta-NLL(h) curves, one axis per NF kind."""
    import matplotlib.pyplot as plt
    kinds = res["kinds"]; hg = res["hgrid"]
    for name, a in axmap.items():
        curves = kinds[name]
        cols = plt.cm.viridis(np.linspace(0.0, 0.82, len(curves)))
        for cdat, col in zip(curves, cols):
            a.plot(hg, cdat["dnll"], "-", color=col, lw=1.8,
                   label=f'{cdat["n"]}p (min {hg[cdat["dnll"].argmin()]:.0f})')
            a.plot(hg[cdat["dnll"].argmin()], 0.0, "o", color=col, ms=6, zorder=5)
        if res["analytic"] is not None:
            a.plot(hg, res["analytic"], "k--", lw=2.0,
                   label=f'analytic linear 4p (min {hg[res["analytic"].argmin()]:.0f})')
            a.plot(hg[res["analytic"].argmin()], 0.0, "k*", ms=13, zorder=6)
        a.axhline(0.5, color="0.4", ls=":", label=r"$\Delta$NLL=0.5")
        a.axvline(res["h_true"], color="k", ls="-", lw=0.8)
        a.set_ylim(0, 12 if res["analytic"] is not None else 80)
        a.set_xlabel(r"$h_{\rm dm}$ [pc]")
        a.set_ylabel(r"profile $\Delta$NLL [nats]")
        a.set_title(f"profile likelihood — {name} kind")
        a.legend(fontsize=7, ncol=2)


def _draw_init_map(res, axm):
    """Draw the init-guess -> fitted-h map (does the fit follow its start?)."""
    import matplotlib.pyplot as plt
    kinds = res["kinds"]
    for name, ls in [(list(kinds)[0], "-"), (list(kinds)[1], "--")]:
        curves = kinds[name]
        cols = plt.cm.viridis(np.linspace(0.0, 0.82, len(curves)))
        for cdat, col in zip(curves, cols):
            if not cdat.get("corr"):          # only featured members carry the map
                continue
            xs = [c[0] for c in cdat["corr"]]; ys = [c[1] for c in cdat["corr"]]
            axm.plot(xs, ys, ls, color=col, marker="o", ms=4, lw=1.2,
                     label=f'{cdat["n"]}p')
    lo, hi = min(res["corr_inits"]), max(res["corr_inits"])
    axm.plot([lo, hi], [lo, hi], ":", color="0.6", label="final = init (unconstrained)")
    axm.axhline(res["h_true"], color="k", ls="--", label=f'truth {res["h_true"]:.0f}')
    axm.set_xlabel("initial guess $h_{\\rm init}$ [pc]")
    axm.set_ylabel("fitted $h_{\\rm dm}$ [pc]")
    axm.set_title("init-guess vs fitted h  (solid=%s, dashed=%s)" % tuple(kinds))
    axm.legend(fontsize=8)


def _draw_well_depth(res, axd):
    """Draw profile well depth vs #parameters (deeper well = tighter h)."""
    kinds = res["kinds"]
    for name, mk in [(list(kinds)[0], "o-"), (list(kinds)[1], "s--")]:
        ns = [c["n"] for c in kinds[name]]
        depth = [float(c["dnll"].max()) for c in kinds[name]]
        axd.plot(ns, depth, mk, label=name)
    if res["analytic"] is not None:
        axd.plot([4], [float(res["analytic"].max())], "k*", ms=13, label="analytic linear")
    axd.set_xscale("log"); axd.set_yscale("log")
    axd.set_xlabel("bijection parameters")
    axd.set_ylabel(r"profile well depth [nats]")
    axd.set_title("deeper well = tighter h")
    axd.legend(fontsize=8)


def _draw_dist_row(res, ax_w, ax_z):
    """Draw the distribution-fit row: true f0 + generating gaussians + data + best fits."""
    kinds = res["kinds"]; names = list(kinds)
    for a, grid, mk_key, samp, true, lab, unit, comps in [
            (ax_w, "wg", "mw", "w0_samples", "true_w", "w_0", "km/s", res.get("comp_w")),
            (ax_z, "zg", "mz", "z0_samples", "true_z", "z_0", "pc", None)]:
        a.hist(res[samp], bins=60, density=True, alpha=0.30, color="0.6", label="data (back-int)")
        if comps:
            for wt, arr in comps:
                a.plot(res[grid], wt * arr, ls=":", lw=1.4, color="C4",
                       label="generating gaussians" if wt == comps[0][0] else None)
        a.plot(res[grid], res[true], "k--", lw=2.4, label="true $f_0$")
        for nm, col in zip(names, ["C0", "C1"]):
            big = kinds[nm][-1]
            a.plot(big[grid], big[mk_key], color=col, lw=1.8,
                   label=f'{nm} fit ({big["n"]}p)')
        a.set_xlabel(fr"${lab}$ [{unit}]"); a.set_ylabel("density"); a.legend(fontsize=7)
    ax_w.set_title("$w_0$ marginal: true $f_0$, generating gaussians, data, best fits")
    ax_z.set_title("$z_0$ marginal")


def plot_profile_comparison(res, title=None, show_dist=False):
    import matplotlib.pyplot as plt
    print(fitted_h_summary(res))
    kinds = res["kinds"]
    nrows = 3 if show_dist else 2
    fig, ax = plt.subplots(nrows, 2, figsize=(14, 5 * nrows))
    _draw_profile_axes(res, {name: ax[0, i] for i, name in enumerate(kinds)})
    _draw_init_map(res, ax[1, 0])
    _draw_well_depth(res, ax[1, 1])
    if show_dist:
        _draw_dist_row(res, ax[2, 0], ax[2, 1])
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_profiles(res, title=None):
    """h_dm results table + the profile-likelihood panels and the well-depth
    summary.  The init-guess panel is intentionally omitted -- that lives in
    plot_init_guess_map so it can be shown separately at the end of the notebook."""
    import matplotlib.pyplot as plt
    print(fitted_h_summary(res))
    kinds = res["kinds"]
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    _draw_profile_axes(res, {name: fig.add_subplot(gs[0, i]) for i, name in enumerate(kinds)})
    _draw_well_depth(res, fig.add_subplot(gs[1, :]))
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_init_guess_map(res, title=None):
    """The init-guess -> fitted-h map on its own (the initial-guess analysis)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    _draw_init_map(res, ax)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def _azimuthal_residual(H, zc, wc, s, smooth=0.0):
    """Given a 2-D density H on grid centres (zc, wc), return (H_smoothed,
    H_smoothed - azimuthal average).  The average is taken in scaled coords
    w_s = w*s (so the cloud is roughly round) and subtracted via a *smooth*
    radial interpolation, so no concentric-ring artifacts appear.  `smooth` is an
    optional Gaussian-filter width (in cells) applied first to tame Poisson noise
    so a ~10%% spiral signal is visible above the shot noise."""
    if smooth > 0:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, smooth)
    ZZ, WW = np.meshgrid(zc, wc * s, indexing="ij")
    R = np.sqrt(ZZ ** 2 + WW ** 2)
    nr = max(len(zc) // 2, 6)
    rbins = np.linspace(0.0, float(R.max()), nr + 1)
    rc = 0.5 * (rbins[:-1] + rbins[1:])
    idx = np.clip(np.digitize(R.ravel(), rbins) - 1, 0, nr - 1)
    prof = np.array([H.ravel()[idx == k].mean() if np.any(idx == k) else np.nan
                     for k in range(nr)])
    good = ~np.isnan(prof)
    Hmean = np.interp(R.ravel(), rc[good], prof[good]).reshape(H.shape)
    return H, H - Hmean


def plot_simulated_data(z_obs, w_obs, true_pdf_z, true_pdf_w, title=None,
                        nbins=110, smooth=1.4, zlim=None, wlim=None):
    """The simulated snapshot as phase-space heatmaps, 2x2:
      * rows    = the two times  (t=0 back-integrated cloud, t=400 Myr observed);
      * columns = absolute phase-space density, and the residual = density minus
        its azimuthal (radial) average.
    The residual cancels the smooth mean and exposes the phase-space SPIRAL that
    anharmonic (differential) phase mixing winds into the cloud by t=400 Myr
    (~structureless for the equilibrium t=0 cloud).  The analytic joint Gaussian
    f_0 the data are sampled from is overlaid on the density panels (white
    contours; evolved through the true back-integration map at t=400 Myr), and its
    own spiral arms are traced on the residual panels (black contours) so the data
    spiral is unmistakable.  For a clean spiral, feed a dense draw (N >~ 60k); a
    light Gaussian smoothing tames the shot noise below the ~10%% spiral signal."""
    import matplotlib.pyplot as plt

    def true2d(Z, W):
        return true_pdf_z(Z) * true_pdf_w(W)

    z0d, w0d = backint(z_obs, w_obs, H_TRUE)         # t=0 cloud (true h)
    times = [(r"$t=0$", z0d, w0d, None),
             (r"$t=400$ Myr", z_obs, w_obs, H_TRUE)]

    fig, ax = plt.subplots(2, 2, figsize=(13.5, 11.5))
    for i, (tlab, zz, ww, hev) in enumerate(times):
        pz, pw = 0.08 * np.ptp(zz), 0.08 * np.ptp(ww)
        ez = np.linspace(zz.min() - pz, zz.max() + pz, nbins + 1)
        ew = np.linspace(ww.min() - pw, ww.max() + pw, nbins + 1)
        H, _, _ = np.histogram2d(zz, ww, bins=[ez, ew], density=True)
        zc = 0.5 * (ez[:-1] + ez[1:]); wc = 0.5 * (ew[:-1] + ew[1:])
        s = zz.std() / ww.std()
        Hs, resid = _azimuthal_residual(H, zc, wc, s, smooth=smooth)

        # analytic joint Gaussian on the same grid (evolved if hev is set) +
        # its own azimuthal residual (the pristine spiral, to trace over the data)
        ZG, WG = np.meshgrid(zc, wc, indexing="ij")
        if hev is None:
            A = true2d(ZG, WG)
        else:
            z0g, w0g = backint(ZG.ravel(), WG.ravel(), hev, ns=1500)
            A = true2d(z0g, w0g).reshape(ZG.shape)
        _, Ares = _azimuthal_residual(A, zc, wc, s)

        a = ax[i, 0]
        im = a.pcolormesh(zc, wc, Hs.T, cmap="magma", shading="gouraud", rasterized=True)
        a.contour(zc, wc, A.T, levels=6, colors="w", linewidths=0.8, alpha=0.85)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label="density")
        a.set_title(f"{tlab} — phase-space density\n(white = analytic joint Gaussian)")

        a = ax[i, 1]
        m = float(np.percentile(np.abs(resid), 99.5)) or 1.0
        im = a.pcolormesh(zc, wc, resid.T, cmap="RdBu_r", vmin=-m, vmax=m,
                          shading="gouraud", rasterized=True)
        am = float(np.abs(Ares).max()) or 1.0
        a.contour(zc, wc, Ares.T, levels=[-0.4 * am, 0.4 * am],
                  colors="k", linewidths=0.7, alpha=0.55)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label="density − azimuthal mean")
        a.set_title(f"{tlab} — residual: phase-space spiral\n(black = analytic spiral arms)")

        for j in range(2):
            ax[i, j].set_xlabel("$z$ [pc]"); ax[i, j].set_ylabel("$w$ [km/s]")
            if zlim is not None:
                ax[i, j].set_xlim(*zlim)
            if wlim is not None:
                ax[i, j].set_ylim(*wlim)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ============================================================================
#   TIME-PARAMETER EXTENSION -- joint fit of (h_dm, Delta t)
# ----------------------------------------------------------------------------
#   The Hamiltonian flow integrates over a total dynamical time
#       Delta t = N_steps * dt   (timestep dt, N_steps leapfrog steps).
#
#   PARAMETRISATION (changed 2026-09-17).  The timestep `dt` is what leapfrog
#   accuracy/stability constrains, so it is now PINNED IN ADVANCE at a value
#   known to be small enough (DT_STEP <= DT_STEP_MAX, i.e. >~ 160 steps per
#   vertical oscillation) and the free fit parameter is the *step count*
#       N_steps = Delta t / DT_STEP,
#   hence the total time.  N_steps is real-valued: the integrator takes
#   floor(N_steps) full steps plus one closing partial step, so the likelihood
#   stays a smooth, differentiable function of the total time while every step
#   taken is the same, safely small dt.  (The old scheme did the opposite --
#   fixed step COUNT, dt = Delta t / N_steps free -- which let the timestep
#   grow with the fitted time and mixed the derivative of the discretisation
#   error into dNLL/d(Delta t).)
#
#   Delta t is optimised jointly with h_dm and the NF bijection parameters,
#   subject to the physical lower bound  Delta t >= DT_MIN.
#   The data were generated at Delta t = T_OBS (= DT_TRUE); the question is
#   whether the fit recovers it.
# ============================================================================
DT_TRUE = T_OBS          # total time the data were generated at [Myr]
DT_MIN  = 200.0          # lower bound on the fitted total time [Myr]
DT_MAX  = 700.0          # numerical upper clamp [Myr]

# --- fixed leapfrog timestep (set in advance, never fitted) -----------------
# The vertical oscillation period is 2*pi/NU ~ 81.5 Myr, so DT_STEP = 0.5 Myr is
# ~163 steps per period and ~1/52 of the leapfrog stability limit 2/NU = 26 Myr.
# The profiled statistic log(sigma_z) + log(sigma_w) is converged to < 2e-6 nats
# per star at this step (vs dt = 0.0625 Myr), i.e. < 0.05 nats at N = 20000 --
# four orders of magnitude below the features of the likelihood.
DT_STEP_MAX = 1.0        # hard ceiling on the timestep [Myr]
DT_STEP     = 0.5        # the timestep actually used [Myr]  (<= DT_STEP_MAX)
N_STEPS_TRUE = DT_TRUE / DT_STEP        # = 800 steps at the true total time


def _gauss_profile_nll(z, w, h, dt, ns=None, dt_step=DT_STEP):
    """Closed-form min-over-(diagonal-Gaussian flow) NLL of the cloud obtained by
    back-integrating (z, w) a total time `dt` at scale height `h`.  For a Gaussian
    f0 this diagonal-Gaussian optimum equals the fully-converged NF profile
    minimum, so it is the *exact* profile NLL(h, dt).

    `dt_step` pins the leapfrog timestep (step count = dt/dt_step, the new
    parametrisation).  Passing `dt_step=None` falls back to the old fixed-step-
    count scheme with `ns` steps."""
    zb, wb = backint(z, w, h, ns=ns, t_obs=dt, dt_step=dt_step)
    return z.size * (math.log(2 * math.pi) + math.log(zb.std())
                     + math.log(wb.std()) + 1.0)


def _profiled_nll_torch(zt, wt, h, dt, dt_step=DT_STEP):
    """Same statistic, torch/differentiable, for a torch h and total time dt."""
    zb, wb = torch_back_integrate(zt, wt, dt, h, sim, dt_step_myr=dt_step)
    return (math.log(2 * math.pi) + torch.log(zb.std())
            + torch.log(wb.std()) + 1.0)


NVP_STEPS = 150          # Adam steps per evaluation (converged; see docstring)
NVP_LR = 0.05


def _nvp_profile_nll(z, w, h, dt, dt_step=DT_STEP, n_steps=NVP_STEPS,
                     lr=NVP_LR, K=1, ret_flow=False):
    """Profiled NLL for the 4-parameter **TinyNVP(1)** bijection -- one RealNVP
    coupling with a non-linear (tanh) Linear(1,2) head.  The non-linear
    counterpart of `_gauss_profile_nll`, with the same parameter count.

    There is no closed form here, so the four parameters are fitted numerically
    at every call.  Two things make that affordable:

      * the flow's fixed standardisation is set to the back-integrated cloud's
        mean and sd -- which IS the diagonal-affine maximum likelihood -- and the
        coupling is zero-initialised, so the optimisation *starts exactly at the
        affine optimum* and can only improve on it.  `n_steps=0` reproduces
        `_gauss_profile_nll` to the last digit;
      * 4 parameters over a few thousand points converge in ~50 Adam steps; 150
        agrees with 400 to <0.01 nats.

    Cost is ~1.6x the closed form (the back-integration still dominates).
    Returns the NLL, or (NLL, flow) with `ret_flow`."""
    zb, wb = backint(z, w, h, t_obs=dt, dt_step=dt_step)
    flow = TinyNVP(int(K)).double()
    flow.set_standardization(zb, wb)
    zt, wt = torch.tensor(zb), torch.tensor(wb)
    if n_steps:
        opt = torch.optim.Adam(flow.parameters(), lr=lr)
        for _ in range(int(n_steps)):
            loss = -flow.log_prob(zt, wt).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        nll = float(-flow.log_prob(zt, wt).sum())
    return (nll, flow) if ret_flow else nll


def _zw_profile_nll(z, w, h, dt, dt_step=DT_STEP, c=1, hid=8, n_adam=400,
                    lr=2e-2, lbfgs=200, ret_flow=False):
    """Profiled NLL for a ZWNormalizingFlow(c, hid) -- the RealNVP flows of the
    two-width ladder (c=1, hid=8 -> 106 params).  Same start-at-the-affine-optimum
    trick as `_nvp_profile_nll` (standardise on the back-integrated cloud; the
    coupling's last layer is zero-init), then Adam + an LBFGS polish.

    The hidden layers ARE randomly initialised, so the torch seed is pinned per
    call: the objective must be a deterministic function of (h, dt) or
    Nelder-Mead chases init noise.  Benchmarked on the two-width data (4000
    stars, 1 thread): ~0.75 ms / Adam step at c=1, 2.65 ms at 20000 stars."""
    from fitter import ZWNormalizingFlow
    zb, wb = backint(z, w, h, t_obs=dt, dt_step=dt_step)
    torch.manual_seed(0)
    flow = ZWNormalizingFlow(int(c), int(hid)).double()
    flow.set_standardization(zb, wb)
    zt, wt = torch.tensor(zb), torch.tensor(wb)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    for _ in range(int(n_adam)):
        loss = -flow.log_prob(zt, wt).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if lbfgs:
        lb = torch.optim.LBFGS(list(flow.parameters()), max_iter=int(lbfgs),
                               history_size=25, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            l = -flow.log_prob(zt, wt).mean()
            l.backward()
            return l
        try:
            lb.step(closure)
        except Exception:                        # LBFGS can trip on flat dirs
            pass
    with torch.no_grad():
        nll = float(-flow.log_prob(zt, wt).sum())
    return (nll, flow) if ret_flow else nll


#: the bijections this notebook fits, by name.
#:   affine -- diagonal affine, 4 params, CLOSED-FORM optimum (exact, free)
#:   nvp    -- TinyNVP(1),  4 params, one RealNVP coupling, fitted numerically
#:   nvp16  -- TinyNVP(4), 16 params, four couplings, fitted numerically
#:   zw106  -- ZWNormalizingFlow(1, 8), 106 params; the hopping (4000-star)
#:             budget.  zw106_full is the SAME model with the budget the
#:             full-sample refine/profile needs (inner fit converges ~4x slower
#:             per nat at 20000 stars).  Pass it as `flow_full=`.
#: The Adam budget is part of the DEFINITION of the nvp objectives: a flexible
#: flow keeps improving with more steps, so the profiled NLL is only a function
#: of (h, Delta t) once the budget is pinned.  600 steps for the 16-parameter
#: flow captures 81% of the 1500-step gain, and the 1.1-nat remainder is small
#: next to the 100-300-nat comb barriers that decide the fit.
PROFILE_NLL = {
    "affine": _gauss_profile_nll,
    "nvp": _nvp_profile_nll,
    "nvp16": _partial(_nvp_profile_nll, K=4, n_steps=600, lr=5e-3),
    "zw106": _partial(_zw_profile_nll, c=1, hid=8, n_adam=400),
    "zw106_full": _partial(_zw_profile_nll, c=1, hid=8, n_adam=3000),
}


SCAN_H_LADDER = (120.0, 220.0, 320.0, 420.0, 520.0)


def dealias_scan(z_obs, w_obs, h, dt_min=DT_MIN, dt_max=DT_MAX, step=4.0,
                 n_sub=4000, dt_step=DT_STEP, refine=4, seed=0,
                 h_ladder=SCAN_H_LADDER):
    """Stage 0 of the fit: a coarse GLOBAL sweep over (h_dm, Delta t).

    NLL(Delta t) is a *comb* -- back-integrating by the wrong total time leaves
    the cloud partly wound up, and the residual width oscillates with the ~41 Myr
    breathing period of the non-equilibrium blob, giving ~16 local minima
    separated by 100-300-nat barriers between 100 and 700 Myr.  No local
    (gradient) step can cross those, so the optimiser is handed the right alias
    first by evaluating the closed-form profiled NLL on a comb-resolving grid.

    The sweep runs over a coarse LADDER of h as well, not just at the caller's
    h: which tooth is deepest depends on h (the teeth drift by about
    -0.05 Myr/pc), and at a badly wrong h the true tooth is genuinely not the
    global one -- scanning Delta t alone at h = 500 pc picks 442 Myr, not 400,
    and the descent then sticks there with h running to its clamp.  Pass
    `h_ladder=None` (or a 1-element ladder) for the 1-D scan.

    Cheap: one back-integration of `n_sub` stars per grid point, and the SAME
    stars at every point, so the subsampling noise is common-mode and the comb
    survives.  Returns (h_best, dt_best)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(z_obs.size, size=min(n_sub, z_obs.size), replace=False)
    zs, ws = z_obs[idx], w_obs[idx]
    hs = [float(h)] if not h_ladder else [float(x) for x in h_ladder]

    def _scan(hgrid, tgrid):
        v = np.array([[_gauss_profile_nll(zs, ws, hh, float(t), dt_step=dt_step)
                       for t in tgrid] for hh in hgrid])
        i, j = np.unravel_index(v.argmin(), v.shape)
        return float(hgrid[i]), float(tgrid[j])

    grid = np.arange(float(dt_min), float(dt_max) + 1e-9, float(step))
    h_b, t_b = _scan(hs, grid)
    if refine and refine > 1:                    # local refinement around the pick
        fine = np.arange(max(dt_min, t_b - step), min(dt_max, t_b + step) + 1e-9,
                         step / float(refine))
        h_b, t_b = _scan([h_b], fine)
    return h_b, t_b


# ============================================================================
#   BASIN HOPPING -- a global optimiser that USES the initial guess
# ----------------------------------------------------------------------------
#   `dealias_scan` finds the right comb tooth by brute force, but it IGNORES the
#   caller's (h_init, dt_init) entirely: the h ladder and the Delta t grid are
#   fixed, so every start produces a bit-identical fit and "converges from any
#   initial guess" becomes untestable.  This is the alternative that actually
#   optimises: it starts AT the initial guess and searches outward.
#
#   Why the obvious things do not work here:
#     * a local optimiser stays in the tooth it started in (100-300-nat walls);
#     * a downhill-only search over the tooth DEPTHS stalls, because the comb
#       envelope is not monotonic -- there are deep decoys at 348 and 446 Myr
#       (63 and 58 nats), one tooth either side of the truth at 402, with
#       ~190-nat ridges in between;
#     * randomised smoothing of the comb was tried and failed (hyperopt stage 2).
#
#   Basin hopping handles exactly this shape: local minimisation, then a jump in
#   PARAMETER space, then re-minimise.  The jump crosses barriers for free
#   because it re-descends where it lands instead of having to climb out.
# ============================================================================
def _nm_local(f, h, dt, bounds, h_scale=25.0, dt_scale=1.5, maxiter=120,
              path=None):
    """Nelder-Mead confined to the current basin.

    The initial simplex is deliberately ANISOTROPIC (h_scale >> dt_scale) to
    match the likelihood, which is ~32x stiffer in Delta t than in h_dm; a
    symmetric simplex spends every evaluation chasing the sharp direction."""
    from scipy.optimize import minimize
    (hlo, hhi), (tlo, thi) = bounds
    x0 = np.array([float(np.clip(h, hlo, hhi)), float(np.clip(dt, tlo, thi))])
    simplex = np.array([x0, x0 + [h_scale, 0.0], x0 + [0.0, dt_scale]])
    simplex[:, 0] = np.clip(simplex[:, 0], hlo, hhi)
    simplex[:, 1] = np.clip(simplex[:, 1], tlo, thi)

    def g(x):
        if not (hlo <= x[0] <= hhi and tlo <= x[1] <= thi):
            return 1e12
        return f(x[0], x[1])

    # `path` (a list) collects the simplex best after every iteration, so the
    # minimisation can be DRAWN as the walk it is rather than a straight arrow.
    cb = None
    if path is not None:
        cb = lambda xk: path.append((float(xk[0]), float(xk[1])))
    r = minimize(g, x0, method="Nelder-Mead", callback=cb,
                 options=dict(initial_simplex=simplex, maxiter=maxiter,
                              xatol=0.02, fatol=1e-4))
    return float(r.x[0]), float(r.x[1]), float(r.fun)


# ----------------------------------------------------------------------------
#   JOINT local minimisation: h, Delta t and EVERY flow parameter together
# ----------------------------------------------------------------------------
class JointFlow(nn.Module):
    """The bijection with ALL its parameters free, for the joint fit: a learnable
    diagonal affine map (mean, log-scale -- the 4 parameters the profile fit
    eliminated in closed form) followed by optional coupling layers.

    The affine part is stored RELATIVE to a reference (the back-integrated
    cloud's mean/sd at the start of each local fit): mu = mu_ref + sig_ref*m,
    sig = sig_ref*exp(s), so every parameter is O(1) and m = s = 0 IS the affine
    optimum at the starting point.

    `mirror=True` restricts the density to the z-MIRROR-SYMMETRIC family,
    p(z, w) = [q(z, w) + q(-z, w)] / 2 (q = the unrestricted flow): a normalised
    density for any q, exactly symmetric under z -> -z.  See
    experiments/mirror_f0.py for why this identifies Delta t."""
    def __init__(self, layers=(), mirror=False):
        super().__init__()
        self.mirror = bool(mirror)
        self.layers = nn.ModuleList(layers)
        self.m = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.s = nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.register_buffer("mu_ref", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("sig_ref", torch.ones(2, dtype=torch.float64))

    def set_ref(self, zb, wb):
        """Re-centre on a new cloud: affine part reset to that cloud's ML
        optimum (m = s = 0); coupling layers untouched (warm start)."""
        with torch.no_grad():
            self.mu_ref = torch.stack([zb.mean(), wb.mean()]).detach().clone()
            self.sig_ref = torch.stack([zb.std(), wb.std()]).detach().clone()
            self.m.zero_(); self.s.zero_()

    @property
    def mu(self):
        return self.mu_ref + self.sig_ref * self.m

    @property
    def sig(self):
        return self.sig_ref * torch.exp(self.s)

    def _log_q(self, z, w):
        xy = (torch.stack([z, w], 1) - self.mu) / self.sig
        ld = xy.new_zeros(xy.shape[0])
        for L in self.layers:
            xy, d = L(xy); ld = ld + d
        lb = -0.5 * (xy ** 2).sum(1) - math.log(2 * math.pi)
        return lb + ld - torch.log(self.sig).sum()

    def log_prob(self, z, w):
        if self.mirror:
            return torch.logaddexp(self._log_q(z, w), self._log_q(-z, w)) - math.log(2.0)
        return self._log_q(z, w)


def make_joint_flow(flow, seed=0):
    """JointFlow for a flow name; a trailing 'm' (e.g. 'zw114m') = the same
    couplings restricted to the z-mirror-symmetric family."""
    mirror = flow.endswith("m") and flow != "m"
    return JointFlow(_joint_layers(flow[:-1] if mirror else flow, seed),
                     mirror=mirror).double()


def _joint_layers(flow, seed=0):
    """Coupling layers per flow name (the affine part is always in JointFlow).
      affine -- none: 4 params (affine only)
      nvp    -- one TinyCoupling: 4 + 4 = 8 params (the profile 'nvp' fit is the
                same model, with its affine part solved in closed form)
      zw114  -- ZWNormalizingFlow(3, 4) couplings: 4 + 114 params
    (a trailing 'm' -- mirror-symmetric -- is handled by `make_joint_flow`)"""
    torch.manual_seed(seed)
    if flow.endswith("m") and flow[:-1] in ("affine", "nvp", "zw114"):
        flow = flow[:-1]
    if flow == "affine":
        return []
    if flow == "nvp":
        return [l.double() for l in TinyNVP(1).layers]
    if flow == "zw114":
        from fitter import ZWNormalizingFlow
        return list(ZWNormalizingFlow(3, 4).double().layers)
    raise ValueError(flow)


H_SCALE = 10.0     # h enters the joint optimiser in units of 10 pc (O(1) curvature)


def _joint_fisher(flow, zb, wb):
    """Excess Fisher information of the flow density, standardised by the
    CLOUD's sd (data-defined, so the penalty cannot be gamed through the
    learnable scale); gradient kept through the points into h and Delta t."""
    lp = flow.log_prob(zb, wb)
    gz, gw = torch.autograd.grad(lp.sum(), (zb, wb), create_graph=True)
    sz, sw = zb.detach().std(), wb.detach().std()
    return ((gz * sz) ** 2 + (gw * sw) ** 2).mean() - 2.0


def _joint_local(zt, wt, h, dt, flow, bounds, dt_step=DT_STEP, max_iter=80,
                 tol=1e-4, lam=0.0, n_pen=4000, path=None, counter=None,
                 pen="fisher"):
    """ONE minimisation of the (penalised) NLL over (h, N_steps, all flow params)
    together, from (h, dt), with the flow's couplings warm-started (the affine
    part is reset to the starting cloud's optimum).  Full batch, L-BFGS with a
    strong-Wolfe line search, run one iteration at a time so the path can be
    recorded.  `flow` is modified in place.  Returns (h, dt, nll_total)."""
    (hlo, hhi), (tlo, thi) = bounds
    u = torch.nn.Parameter(torch.tensor(float(np.clip(h, hlo, hhi)) / H_SCALE,
                                        dtype=torch.float64))
    nst = torch.nn.Parameter(torch.tensor(float(np.clip(dt, tlo, thi)) / dt_step,
                                          dtype=torch.float64))
    with torch.no_grad():
        zb0, wb0 = torch_back_integrate(zt, wt, nst * dt_step, u * H_SCALE, sim,
                                        dt_step_myr=dt_step)
    flow.set_ref(zb0, wb0)
    ip = slice(0, min(int(n_pen), zt.numel()))

    def total():
        hh = torch.clamp(u * H_SCALE, hlo, hhi)
        nn_ = torch.clamp(nst, tlo / dt_step, thi / dt_step)
        zb, wb = torch_back_integrate(zt, wt, nn_ * dt_step, hh, sim,
                                      dt_step_myr=dt_step)
        J = -flow.log_prob(zb, wb).sum()
        if lam > 0:
            R = (_joint_fisher(flow, zb[ip], wb[ip]) if pen == "fisher" else
                 sum((p ** 2).sum() for p in flow.layers.parameters()))
            J = J + lam * zt.numel() * R
        return J

    params = [u, nst] + list(flow.parameters())

    def new_lbfgs():
        return torch.optim.LBFGS(params, max_iter=1, history_size=30,
                                 line_search_fn="strong_wolfe")
    lb = new_lbfgs()

    def closure():
        if counter is not None:
            counter[0] += 1
        lb.zero_grad()
        J = total()
        J.backward()
        return J

    # Stop only when the objective AND the position in (h, Delta t) have both
    # stopped changing for 3 consecutive iterations.  A rule on the objective
    # alone stops EARLY: along the h-Delta t degeneracy ridge (-0.05 Myr/pc) the
    # objective changes by < 1e-4 nats per iteration while h is still sliding --
    # measured, it halted 120 pc short of the minimum (0.65 nats too high);
    # without the early stop the same L-BFGS lands on the closed-form profile
    # optimum to 0.01 pc.
    prev, prev_x, quiet, restarted = None, None, 0, False
    for _ in range(int(max_iter)):
        try:
            J = float(lb.step(closure))
        except Exception:
            break
        x = (float(np.clip(u.item() * H_SCALE, hlo, hhi)),
             float(np.clip(nst.item() * dt_step, tlo, thi)))
        if path is not None:
            path.append(x)
        still = (prev is not None and abs(prev - J) < tol
                 and abs(x[0] - prev_x[0]) < 0.02 and abs(x[1] - prev_x[1]) < 0.002)
        quiet = quiet + 1 if still else 0
        if quiet >= 3:
            # Looks converged -- but a failed line search ALSO makes no move.
            # Restart once with a fresh L-BFGS history and stop only if that
            # makes no progress either (measured: 1 of 32 affine runs stalled
            # 9.5 pc short, 0.046 nats high, without this).
            if restarted:
                break
            restarted, quiet = True, 0
            lb = new_lbfgs()
        prev, prev_x = J, x
    J = float(total().detach())          # grad mode on: the Fisher term needs it
    return (float(np.clip(u.item() * H_SCALE, hlo, hhi)),
            float(np.clip(nst.item() * dt_step, tlo, thi)), J)


def joint_hessian(z, w, h, dt, flow, state, lam=0.0, pen="fisher",
                  dt_step=DT_STEP, n_pen=4000):
    """Hessian of the (penalised) NLL in nats over ALL parameters -- (h, N_steps)
    and every JointFlow parameter -- at a joint optimum.  The (h, Delta t)
    covariance is the inverse of the Schur complement, i.e. marginalised over
    the flow parameters: the local curvature of the profile likelihood obtained
    without fixing anything.  Zero-curvature flow directions (symmetries) are
    removed by a pseudo-inverse.  Returns dict(sig_h, sig_dt, corr, is_min, ...)."""
    jf = make_joint_flow(flow)
    jf.load_state_dict(state)
    zt, wt = torch.tensor(z), torch.tensor(w)
    hh = torch.tensor(float(h), dtype=torch.float64, requires_grad=True)
    nst = torch.tensor(float(dt) / dt_step, dtype=torch.float64, requires_grad=True)
    P = [hh, nst] + list(jf.parameters())
    zb, wb = torch_back_integrate(zt, wt, nst * dt_step, hh, sim, dt_step_myr=dt_step)
    J = -jf.log_prob(zb, wb).sum()
    if lam > 0:
        ip = slice(0, min(int(n_pen), zt.numel()))
        R = (_joint_fisher(jf, zb[ip], wb[ip]) if pen == "fisher" else
             sum((p ** 2).sum() for p in jf.layers.parameters()))
        J = J + lam * zt.numel() * R
    g = torch.cat([x.reshape(-1) for x in torch.autograd.grad(J, P, create_graph=True)])
    n = g.numel()
    H = np.empty((n, n))
    for i in range(n):
        row = torch.autograd.grad(g[i], P, retain_graph=True, allow_unused=True)
        H[i] = torch.cat([(r if r is not None else torch.zeros_like(x)).reshape(-1)
                          for r, x in zip(row, P)]).detach().numpy()
    H = 0.5 * (H + H.T)
    Hxx, Hxt, Htt = H[:2, :2], H[:2, 2:], H[2:, 2:]
    ev, V = np.linalg.eigh(Htt)
    keep = ev > 1e-8 * max(ev.max(), 1e-300)
    S = Hxx - Hxt @ ((V[:, keep] / ev[keep]) @ V[:, keep].T) @ Hxt.T
    is_min = bool(np.all(np.linalg.eigvalsh(S) > 0)) and bool(ev.min() > -1e-6 * ev.max())
    cov = np.linalg.inv(S)
    ok = cov[0, 0] > 0 and cov[1, 1] > 0
    return dict(sig_h=math.sqrt(cov[0, 0]) if ok else float("nan"),
                sig_dt=math.sqrt(cov[1, 1]) * dt_step if ok else float("nan"),
                corr=cov[0, 1] / math.sqrt(abs(cov[0, 0] * cov[1, 1])),
                is_min=is_min, flow_min_eig=float(ev.min()),
                flow_max_eig=float(ev.max()), n_dropped=int((~keep).sum()),
                grad_max=float(g.detach().abs().max()), S=S, n_params=n)


def basin_hop_h_dt(z_obs, w_obs, h_init, dt_init, n_iter=60, n_sub=4000,
                   dt_jump=55.0, h_jump=45.0, patience=25, seed=0,
                   dt_min=DT_MIN, dt_max=DT_MAX, dt_step=DT_STEP,
                   h_bounds=(50.0, 800.0), refine_full=True, info=None,
                   flow="affine", flow_full=None, local="nm", lam=0.0,
                   pen="fisher"):
    """Global MLE of (h_dm, Delta t) by basin hopping FROM (h_init, dt_init).

    `local` -- what happens after every jump:
      "nm"    -- Nelder-Mead over (h, Delta t) of the PROFILED likelihood (the
                 flow re-fitted from scratch at every point it tries);
      "joint" -- ONE L-BFGS minimisation of h, Delta t and EVERY flow parameter
                 together (`_joint_local`), with the flow's couplings
                 warm-started from the stored best point.  The stored best
                 point carries its flow weights; a rejected hop's weights are
                 discarded with it.  `lam` > 0 adds the excess-Fisher complexity
                 penalty on f0.

    `flow_full` (a PROFILE_NLL key, default = `flow`) is the objective of the
    full-sample refine -- for a numerically-fitted flow it carries the larger
    inner budget the 20000-star fit needs.

    TWO objectives, on purpose.  The hopping phase needs thousands of evaluations
    so it runs on a fixed `n_sub`-star subsample -- enough to identify the right
    comb TOOTH, but NOT to locate the minimum inside it: the subsample's own
    optimum sits ~65 pc away in h_dm (the soft direction), which costs ~3.5 nats
    on the full sample.  So the subsample selects the BASIN and the full sample
    then decides where in that basin the answer is (`refine_full`).

    Unlike `dealias_scan`, the initial guess is genuinely used: different starts
    take different routes (30-62 hops, 2200-4300 evaluations in testing) and the
    agreement of their endpoints is therefore a real measurement, not an identity.

    Returns (h_fit, dt_fit, traj) where traj is the list of accepted
    (h, dt, nll) states, starting with the initial guess."""
    rng = np.random.default_rng(seed)
    bounds = (tuple(h_bounds), (float(dt_min), float(dt_max)))

    sub_rng = np.random.default_rng(0)      # the SAME stars at every evaluation
    idx = sub_rng.choice(z_obs.size, size=min(int(n_sub), z_obs.size),
                         replace=False)
    zs, ws = z_obs[idx], w_obs[idx]
    n_eval = [0]

    if local == "joint":
        zst, wst = torch.tensor(zs), torch.tensor(ws)
        jflow = make_joint_flow(flow, seed)
        keep = {"state": None}               # flow weights OF THE STORED POINT

        def minimise(h0, t0, path, full=False):
            if keep["state"] is not None:
                jflow.load_state_dict(keep["state"])
            zt_, wt_ = ((torch.tensor(z_obs), torch.tensor(w_obs)) if full
                        else (zst, wst))
            return _joint_local(zt_, wt_, h0, t0, jflow, bounds, dt_step=dt_step,
                                max_iter=400 if full else 80, lam=lam, pen=pen,
                                path=path, counter=n_eval if not full else m_full)

        def remember():
            keep["state"] = {k: v.detach().clone()
                             for k, v in jflow.state_dict().items()}
    else:
        prof = PROFILE_NLL[flow]            # 'affine' (closed form) or 'nvp'

        def f_sub(h, dt):
            n_eval[0] += 1
            return prof(zs, ws, h, dt, dt_step=dt_step)

        def minimise(h0, t0, path, full=False):
            return _nm_local(f_sub, h0, t0, bounds, path=path)

        def remember():
            pass
    m_full = [0]

    mp0 = [(float(h_init), float(dt_init))]
    h, dt, fx = minimise(h_init, dt_init, mp0)
    remember()
    mp0.append((h, dt))
    traj = [(float(h_init), float(dt_init), None), (h, dt, fx)]
    # `hops` keeps the two halves of every hop SEPARATE -- the jump and the
    # minimisation that follows it -- because collapsing them into one segment
    # is what made the trajectory figure unreadable.
    hops = [dict(kind="init", frm=(float(h_init), float(dt_init)),
                 prop=(float(h_init), float(dt_init)), land=(h, dt), nll=fx,
                 accepted=True, mpath=mp0)]
    best = (h, dt, fx)
    stale = 0
    for _ in range(int(n_iter)):
        # mixture proposal: mostly 1-3 teeth, occasionally 4x that, so the walker
        # can refine locally AND escape a deep decoy 45-55 Myr from the truth.
        ddt = (rng.uniform(-1, 1) * 4.0 * dt_jump if rng.random() < 0.25
               else rng.normal(0.0, dt_jump))
        ch = float(np.clip(h + rng.normal(0.0, h_jump), *bounds[0]))
        cdt = float(np.clip(dt + ddt, *bounds[1]))
        mp = [(ch, cdt)]
        nh, ndt, nf = minimise(ch, cdt, mp)
        mp.append((nh, ndt))
        acc = nf < fx                                 # greedy acceptance
        hops.append(dict(kind="hop", frm=(h, dt), prop=(ch, cdt),
                         land=(nh, ndt), nll=nf, accepted=bool(acc),
                         mpath=mp))
        if acc:
            h, dt, fx = nh, ndt, nf
            remember()                       # the new stored point's flow
        traj.append((h, dt, fx))
        if nf < best[2] - 1e-9:
            best = (nh, ndt, nf); stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    h, dt = best[0], best[1]
    if info is not None:
        info.update(h_hop=h, dt_hop=dt, n_hops=len(traj) - 2,
                    n_eval_sub=n_eval[0], hops=hops)

    if refine_full:
        mpr = [(h, dt)]
        if local == "joint":
            h, dt, J_full = minimise(h, dt, mpr, full=True)
            remember()
            m_eval = m_full
            if info is not None:
                info.update(J_full=J_full, flow_state=keep["state"])
        else:
            m_eval = [0]
            prof_full = PROFILE_NLL[flow_full or flow]

            def f_full(hh, tt):
                m_eval[0] += 1
                return prof_full(z_obs, w_obs, hh, tt, dt_step=dt_step)
            h, dt, _ = _nm_local(f_full, h, dt, bounds, h_scale=20.0,
                                 dt_scale=0.8, maxiter=200, path=mpr)
        mpr.append((h, dt))
        traj.append((h, dt, None))
        if info is not None:
            info.update(n_eval_full=m_eval[0])
            info["hops"] = info["hops"] + [
                dict(kind="refine", frm=(best[0], best[1]),
                     prop=(best[0], best[1]), land=(h, dt), nll=None,
                     accepted=True, mpath=mpr)]
    return h, dt, traj


def joint_fit_h_dt(make_flow, z_obs, w_obs, h_init, dt_init,
                   n_epochs=400, n_sub=20000, dt_step=DT_STEP,
                   lr_flow=2e-2, lr_h=8.0, lr_dt=4.0, warm=150,
                   sched="cosine", grad_clip=30.0, resample=True,
                   jitter0=0.0, jitter_end=0.0, n_jit=1,
                   scan=True, scan_step=4.0, scan_sub=4000, scan_refine=4,
                   scan_h=SCAN_H_LADDER,
                   polish=150, polish_lr=0.25, polish_sub=None,
                   dt_min=DT_MIN, dt_max=DT_MAX, seed=0, ns=None, info=None):
    """Joint MLE of (flow params, h_dm, Delta t) in the fixed-timestep
    parametrisation: the timestep is pinned at `dt_step` and the free parameter
    is the real-valued STEP COUNT  N_steps = Delta t / dt_step.

    Three stages (all individually switchable).  THE DEFAULTS BELOW ARE THE ONES
    THE HYPER-PARAMETER SEARCH SELECTED -- `experiments/hyperopt_time.py`, which
    scored candidates by the full-sample NLL at their landing point over a spread
    of starts: global sweep + full-batch descent at lr_h = 8 pc/epoch + full-batch
    polish reaches the same optimum (to 1e-4 nats) from every start, while every
    purely local setting tried lands ~245 nats above it:

      0. `scan`   -- global de-aliasing sweep (`dealias_scan`) over a coarse
                     (h ladder) x (Delta t comb) grid, which replaces the caller's
                     (h_init, dt_init) by the best coarse pair.  Without it the fit
                     can only reach the alias it starts in; with a 1-D sweep at
                     h_init alone it can still pick the wrong tooth when h_init is
                     far off (see `dealias_scan`).  The h ladder is deliberately
                     coarse (100 pc spacing), so the descent still has to converge
                     h over tens of pc.
      1. descent  -- Adam on (flow, h, N_steps) for `n_epochs` epochs, full batch
                     by default: the h_dm gradient is ~0.05 nats/pc for the whole
                     20000-star sample, below the gradient noise of a fresh
                     minibatch, so with resampling h random-walks instead of
                     descending.  Optionally with *randomised smoothing* -- the
                     total time jittered by eps ~ N(0, sigma_j^2) (`n_jit`
                     antithetic samples), sigma_j annealed jitter0 -> jitter_end,
                     a graduated-non-convexity attempt at the comb.  KEPT BUT OFF
                     BY DEFAULT: the search found it does not work (the comb's
                     envelope is not smooth enough), and applying it after the
                     sweep is actively harmful -- it throws the fit back out of a
                     well less than a Myr wide.
      2. `polish` -- full-batch, low-lr, no-jitter Adam to land on the exact
                     minimum.

    `lr_dt` is quoted in Myr/epoch (converted internally to steps/epoch).
    `resample=False` freezes ONE minibatch for the whole descent, which makes the
    objective deterministic -- it matters because the h_dm gradient is so shallow
    (~3 nats at 70 pc from the truth) that fresh-minibatch noise swamps it and h
    then just tracks its initial guess.  Delta t is clamped to [dt_min, dt_max].
    `ns` is accepted and ignored (old signature).
    Returns (flow, h_fit, dt_fit, h_hist, dt_hist, loss_hist)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    flow = make_flow()
    zt_all, wt_all = torch.tensor(z_obs), torch.tensor(w_obs)

    # ---- stage 0: global de-aliasing sweep ---------------------------------
    h0, dt0 = float(h_init), float(dt_init)
    if scan:
        h0, dt0 = dealias_scan(z_obs, w_obs, h_init, dt_min=dt_min, dt_max=dt_max,
                               step=scan_step, n_sub=scan_sub, dt_step=dt_step,
                               refine=scan_refine, seed=seed, h_ladder=scan_h)
    if info is not None:
        info.update(dt_scan=dt0, h_scan=h0, n_epochs=n_epochs, polish=polish)

    h = nn.Parameter(torch.tensor(float(h0), dtype=torch.float64))
    # THE free time parameter: the (real-valued) number of leapfrog steps
    nst = nn.Parameter(torch.tensor(dt0 / dt_step, dtype=torch.float64))
    n_lo, n_hi = dt_min / dt_step, dt_max / dt_step

    def _cloud(idx, eps=0.0):
        return torch_back_integrate(zt_all[idx], wt_all[idx], nst * dt_step + eps,
                                    h, sim, dt_step_myr=dt_step)

    # ---- warm up the flow only, at the fixed initial (h, Delta t) -----------
    optf = torch.optim.Adam(flow.parameters(), lr=lr_flow)
    for _ in range(warm):
        idx = rng.choice(z_obs.size, size=min(n_sub, z_obs.size), replace=False)
        with torch.no_grad():
            zb, wb = _cloud(idx)
        optf.zero_grad(); (-flow.log_prob(zb, wb).mean()).backward(); optf.step()

    hh, dd, ll = [], [], []

    def _descend(epochs, lrs, use_sched, sub, jit0, jit1, njit):
        if epochs <= 0:
            return
        opt = torch.optim.Adam([{'params': flow.parameters(), 'lr': lrs[0]},
                                {'params': [h],   'lr': lrs[1]},
                                {'params': [nst], 'lr': lrs[2] / dt_step}])
        sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
               if use_sched == "cosine" else None)
        fixed_idx = (None if (resample or sub >= z_obs.size) else
                     rng.choice(z_obs.size, size=sub, replace=False))
        for e in range(epochs):
            sig = (jit0 * (jit1 / jit0) ** (e / max(epochs - 1, 1))
                   if (jit0 > 0 and jit1 > 0) else 0.0)
            idx = (np.arange(z_obs.size) if sub >= z_obs.size else
                   fixed_idx if fixed_idx is not None else
                   rng.choice(z_obs.size, size=sub, replace=False))
            opt.zero_grad()
            # randomised smoothing: average the loss over antithetic jitters
            # (offsets clipped so the jittered total time stays in the box)
            offs = ([0.0] if sig <= 0 else
                    [s * e_ for e_ in rng.normal(0.0, sig, max(njit // 2, 1))
                     for s in (1.0, -1.0)][:max(njit, 1)])
            t_now = nst.item() * dt_step
            offs = [float(np.clip(o, 1.0 - t_now, dt_max - t_now)) for o in offs]
            loss_val = 0.0
            for off in offs:
                zb, wb = _cloud(idx, float(off))
                lo = -flow.log_prob(zb, wb).mean() / len(offs)
                lo.backward()
                loss_val += float(lo.detach())
            if grad_clip:
                torch.nn.utils.clip_grad_norm_([h], grad_clip)
                torch.nn.utils.clip_grad_norm_([nst], grad_clip / dt_step)
            opt.step()
            if sch is not None:
                sch.step()
            with torch.no_grad():
                h.clamp_(50.0, 800.0); nst.clamp_(n_lo, n_hi)
            hh.append(h.item()); dd.append(nst.item() * dt_step); ll.append(loss_val)

    _descend(n_epochs, (lr_flow, lr_h, lr_dt), sched,
             min(n_sub, z_obs.size), jitter0, jitter_end, n_jit)
    _descend(polish, (lr_flow * polish_lr, lr_h * polish_lr, lr_dt * polish_lr),
             "cosine", min(polish_sub or z_obs.size, z_obs.size), 0.0, 0.0, 1)

    return flow, h.item(), nst.item() * dt_step, hh, dd, ll


# old name kept: the step count IS the time parameter now
joint_fit_h_nsteps = joint_fit_h_dt


def _parab_sigma(xgrid, dnll, level=2.0):
    """1-sigma half-width (Delta NLL = 0.5) from a local parabola around the
    profile minimum, plus the sub-grid position of the minimum (the parabola's
    vertex).  The fit window is every grid point within `level` nats of the
    minimum (at least the 5 nearest), so it adapts to wells of very different
    widths -- the Delta t well is a fraction of a Myr wide while the h_dm well is
    tens of pc, and a fixed +-2-node window mis-measures one or the other.
    Returns (sigma, x_min); sigma is NaN if the window is not convex."""
    xgrid = np.asarray(xgrid, dtype=float); dnll = np.asarray(dnll, dtype=float)
    k = int(np.argmin(dnll))
    sel = np.where(dnll - dnll[k] <= level)[0]
    if sel.size < 5:                                  # widen to the 5 nearest
        lo, hi = max(k - 2, 0), min(k + 2, len(xgrid) - 1)
        sel = np.arange(lo, hi + 1)
    xs, ys = xgrid[sel], dnll[sel]
    if len(xs) < 3:
        return float("nan"), float(xgrid[k])
    a, b, _ = np.polyfit(xs - xgrid[k], ys, 2)
    if a <= 0:
        return float("nan"), float(xgrid[k])
    return float(math.sqrt(0.5 / a)), float(xgrid[k] - b / (2.0 * a))


_GRID_CTX = {}


def _grid_init(z, w, dt_step, threads=None, flow="affine"):
    if threads:                       # pool workers: one thread each
        torch.set_num_threads(int(threads))
    _GRID_CTX.update(z=z, w=w, dt_step=dt_step, flow=flow)


def _grid_worker(pt):
    g = _GRID_CTX
    return PROFILE_NLL[g.get("flow", "affine")](
        g["z"], g["w"], pt[0], pt[1], dt_step=g["dt_step"])


def _fit_worker(job):
    """One joint fit, in a pool worker (module level so it is picklable)."""
    ((hi, dti), kw, tag), common_kw = job
    g = _GRID_CTX
    z, w = g["z"], g["w"]
    info = {}
    flow, hf, dtf, hh, dd, ll = joint_fit_h_dt(_diag_affine_maker, z, w, hi, dti,
                                               info=info, **common_kw, **kw)
    return dict(h_init=hi, dt_init=dti, h_fit=hf, dt_fit=dtf, tag=tag,
                h_hist=hh, dt_hist=dd, loss_hist=ll, **info,
                nf=_flow_params(flow),
                nll=_gauss_profile_nll(z, w, hf, dtf,
                                       dt_step=common_kw["dt_step"]))


def _flow_params(flow):
    """The fitted bijection's parameters as a plain dict, so they survive into
    the cached result alongside the Hamiltonian-flow ones.  The bijection here is
    `DiagonalAffineFlow` -- x = a*u + b on each axis, 4 numbers -- whose `a` is
    stored as log a; anything else is returned raw, by parameter name."""
    out = {}
    with torch.no_grad():
        for name, p in flow.named_parameters():
            v = p.detach().cpu().numpy().ravel()
            if name == "log_a" and v.size == 2:
                out["a_z"], out["a_w"] = float(np.exp(v[0])), float(np.exp(v[1]))
            elif name == "b" and v.size == 2:
                out["b_z"], out["b_w"] = float(v[0]), float(v[1])
            else:
                for i, x in enumerate(v):
                    out[f"{name}[{i}]" if v.size > 1 else name] = float(x)
    return out


def grid_nll(z, w, hs, dts, dt_step=DT_STEP, workers=None, flow="affine"):
    """Exact profiled NLL on the (Delta t) x (h) grid, parallel over grid points.
    Falls back to serial if a process pool cannot be created."""
    pts = [(float(h), float(d)) for d in dts for h in hs]
    workers = (max((os.cpu_count() or 2) - 2, 1)) if workers is None else workers
    vals = None
    if workers > 1:
        try:
            from multiprocessing import Pool
            with Pool(workers, initializer=_grid_init,
                      initargs=(z, w, dt_step, 1, flow)) as p:
                vals = p.map(_grid_worker, pts, chunksize=8)
        except Exception:
            vals = None
    if vals is None:
        _grid_init(z, w, dt_step, flow=flow)
        vals = [_grid_worker(pt) for pt in pts]
    return np.asarray(vals, dtype=float).reshape(len(dts), len(hs))


def run_time_parameter(tid, z, w, dt_min=DT_MIN,
                       inits=((250, 400), (220, 300), (290, 520), (240, 250)),
                       extra_inits=(), extra_fit_kw=None, extra_tag="local-only",
                       dt_step=DT_STEP, seed=0, fit_kw=None,
                       h_half=100.0, n_h=25, dt_half=8.0, dt_fine=0.25,
                       comb_step=1.0, comb_max=DT_MAX, workers=None,
                       hgrid=None, dtgrid=None, ns_grid=None, ns_fit=None):
    """Joint (h_dm, Delta t) study: ONE tuned joint fit per starting point, then
    the profile likelihood built AROUND the resulting minimum.

    Order of operations (the same "joint-fit first, then profile" discipline the
    h-only notebooks use -- a fixed pre-chosen grid would put the error bars
    wherever the grid happens to be sampled):

      1. `joint_fit_h_dt` from every start in `inits` + `extra_inits`.  Each fit
         does the global de-aliasing sweep of Delta t, then joint Adam descent on
         (flow, h, N_steps), then a full-batch polish.  The landing with the
         lowest full-sample NLL is (h_hat, Delta t_hat).
      2. The alias COMB: exact profiled NLL(Delta t) at h_hat across the whole
         allowed range at `comb_step` Myr -- this is the structure that defeats a
         purely local optimiser (~16 local minima, 100-300-nat barriers).
      3. A 2-D ZOOM of the exact profile NLL(h, Delta t) centred on
         (h_hat, Delta t_hat): h_hat +- `h_half` pc x Delta t_hat +- `dt_half` Myr
         at `dt_fine` Myr, fine enough to resolve a well whose 1-sigma half-width
         is a fraction of a Myr.  The 1-D profiles and the Delta NLL = 0.5 error
         bars come from that zoom.

    `dt_min` is the free lower bound on Delta t (clamps the fit AND floors both
    grids), so lowering it to 0 exposes what the fit does when the total time is
    allowed to vanish.  `extra_inits` adds below-the-well starts for that case.
    `hgrid`/`dtgrid`/`ns_grid`/`ns_fit` are accepted for signature compatibility
    and ignored."""
    fit_kw = dict(fit_kw or {})
    jobs = ([(ini, fit_kw, "tuned") for ini in inits]
            + [(ini, dict(fit_kw, **(extra_fit_kw or {})), extra_tag)
               for ini in extra_inits])
    jobs = [(j, dict(dt_step=dt_step, dt_min=dt_min, seed=seed)) for j in jobs]
    nw = (max((os.cpu_count() or 2) - 2, 1)) if workers is None else workers
    traj = None
    if nw > 1 and len(jobs) > 1:
        try:
            from multiprocessing import Pool
            with Pool(min(nw, len(jobs)), initializer=_grid_init,
                      initargs=(z, w, dt_step, 1)) as p:
                traj = p.map(_fit_worker, jobs, chunksize=1)
        except Exception:
            traj = None
    if traj is None:
        _grid_init(z, w, dt_step)
        traj = [_fit_worker(j) for j in jobs]

    best = min([t for t in traj if t["tag"] == "tuned"] or traj,
               key=lambda t: t["nll"])
    h_hat, dt_hat = best["h_fit"], best["dt_fit"]

    # ---- 2. the alias comb at h_hat ----------------------------------------
    comb_dt = np.arange(float(dt_min), float(comb_max) + 1e-9, float(comb_step))
    comb = grid_nll(z, w, [h_hat], comb_dt, dt_step=dt_step,
                    workers=workers).ravel()

    # ---- 3. the 2-D zoom around (h_hat, Delta t_hat) ------------------------
    hgrid = np.linspace(h_hat - h_half, h_hat + h_half, int(n_h))
    lo = max(float(dt_min), dt_hat - dt_half)
    dtgrid = np.arange(lo, dt_hat + dt_half + 1e-9, float(dt_fine))
    NLL = grid_nll(z, w, hgrid, dtgrid, dt_step=dt_step, workers=workers)

    ref = NLL.min()
    NLL = NLL - ref
    comb = comb - ref
    prof_dt = NLL.min(axis=1)          # profile over h  -> Delta NLL(Delta t)
    prof_h  = NLL.min(axis=0)          # profile over dt -> Delta NLL(h)
    sig_dt, dt_best = _parab_sigma(dtgrid, prof_dt)
    sig_h,  h_best  = _parab_sigma(hgrid,  prof_h)

    return dict(id=tid, hgrid=hgrid, dtgrid=dtgrid, NLL=NLL,
                comb_dt=comb_dt, comb_nll=comb,
                h_best=h_best, dt_best=dt_best, prof_dt=prof_dt, prof_h=prof_h,
                sig_dt=sig_dt, sig_h=sig_h, traj=traj, N=int(z.size),
                h_true=H_TRUE, dt_true=DT_TRUE, dt_min=float(dt_min),
                dt_step=float(dt_step), n_steps_best=dt_best / float(dt_step),
                h_hat=h_hat, dt_hat=dt_hat, fit_kw=fit_kw)


def hdt_summary(res):
    """Text summary of the joint (h_dm, Delta t) fit."""
    ds = res.get("dt_step", DT_STEP)
    lines = [f"Joint (h_dm, Delta t) MLE   "
             f"[truth: h = {res['h_true']:.0f} pc, Delta t = {res['dt_true']:.0f} Myr; "
             f"bound Delta t >= {res['dt_min']:.0f} Myr]",
             f"  parametrisation      : dt = {ds:g} Myr FIXED, free parameter = step "
             f"count N  (Delta t = N*dt); truth N = {res['dt_true']/ds:.0f}",
             f"  profile minimum      : h_dm    = {res['h_best']:.1f} +/- {res['sig_h']:.1f} pc",
             f"                         Delta t = {res['dt_best']:.2f} +/- {res['sig_dt']:.2f} Myr"
             f"   (N = {res.get('n_steps_best', float('nan')):.1f} steps)",
             "  joint-fit landings (flow + h + N, from each start):"]
    tuned = [t for t in res["traj"] if t.get("tag", "tuned") == "tuned"]
    for t in res["traj"]:
        tag = t.get("tag", "tuned")
        lines.append(f"     init (h={t['h_init']:>3.0f}, dt={t['dt_init']:>3.0f}) "
                     f"-> (h={t['h_fit']:7.2f}, dt={t['dt_fit']:7.2f})"
                     + ("" if tag == "tuned" else f"   [{tag}]"))
    hs = np.array([t["h_fit"] for t in tuned])
    ds_ = np.array([t["dt_fit"] for t in tuned])
    lines.append(f"     spread across starts: sd(h) = {hs.std():.2f} pc, "
                 f"sd(Delta t) = {ds_.std():.2f} Myr")
    lines.append("  NOTE: the total time Delta t is pinned FAR more tightly than h_dm "
                 f"(sigma_dt/dt = {res['sig_dt']/res['dt_best']*100:.3f}% vs "
                 f"sigma_h/h = {res['sig_h']/res['h_best']*100:.1f}%).")
    return "\n".join(lines)


def plot_hdt_landscape(res, title=None):
    """Two views of the same exact profile likelihood.

    LEFT -- the alias comb: Delta NLL(Delta t) at h_hat across the whole allowed
    range.  A forest of local minima separated by 100-300-nat barriers, with one
    razor-sharp global tooth at the truth.  This is what a purely local optimiser
    cannot cross, and why the fit starts with a global de-aliasing sweep.
    RIGHT -- the 2-D zoom around the joint minimum, with the truth, the profile
    minimum and the landing point of every start."""
    import matplotlib.pyplot as plt
    H, D, Z = res["hgrid"], res["dtgrid"], res["NLL"]
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.2),
                           gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    a = ax[0]
    a.plot(res["comb_dt"], res["comb_nll"], "-", color="0.25", lw=1.0)
    a.plot(res["dt_best"], 0.0, "P", color="r", mec="k", ms=11, zorder=6,
           label="global minimum %.1f Myr" % res["dt_best"])
    a.axvline(res["dt_true"], color="k", lw=0.9, ls="--", label="truth 400")
    for t in res["traj"]:
        a.plot(t["dt_init"], np.interp(t["dt_init"], res["comb_dt"], res["comb_nll"]),
               "o", color="C0", mec="k", ms=6, zorder=5)
    a.plot([], [], "o", color="C0", mec="k", ms=6, label="starting points")
    a.set_xlabel(r"total dynamical time $\Delta t$ [Myr]")
    a.set_ylabel(r"profile $\Delta$NLL at $\hat{h}$ [nats]")
    a.set_title(r"the $\Delta t$ alias comb (exact profile, $h=\hat{h}$)")
    a.legend(fontsize=8); a.grid(alpha=0.25)

    b = ax[1]
    top = float(min(Z.max(), 60.0))
    cs = b.contourf(H, D, np.clip(Z, 0, top), levels=np.linspace(0, top, 25),
                    cmap="viridis_r", extend="max")
    fig.colorbar(cs, ax=b, label=r"profile $\Delta$NLL [nats]")
    b.contour(H, D, Z, levels=[0.5, 2, 8, 30], colors="w", linewidths=0.7, alpha=0.7)
    b.plot(res["h_true"], res["dt_true"], "*", color="w", mec="k", ms=20,
           label="truth (%.0f, %.0f)" % (res["h_true"], res["dt_true"]), zorder=6)
    b.plot(res["h_best"], res["dt_best"], "P", color="r", mec="k", ms=12,
           label="profile min (%.1f, %.1f)" % (res["h_best"], res["dt_best"]), zorder=6)
    cols = plt.cm.autumn(np.linspace(0.0, 0.75, len(res["traj"])))
    for t, c in zip(res["traj"], cols):
        if t.get("tag", "tuned") == "tuned":
            b.plot(t["h_fit"], t["dt_fit"], "s", color=c, mec="k", ms=7, zorder=7)
    b.plot([], [], "s", color="orange", mec="k", ms=7, label="landings (all starts)")
    b.set_xlabel(r"$h_{\rm dm}$ [pc]")
    b.set_ylabel(r"$\Delta t$ [Myr]")
    b.set_xlim(H.min(), H.max()); b.set_ylim(D.min(), D.max())
    b.legend(fontsize=8, loc="upper right", framealpha=0.9)
    b.set_title("zoom on the minimum")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_hdt_profiles(res, title=None):
    """1-D profile likelihoods: Delta NLL(Delta t) (min over h) and Delta NLL(h)
    (min over Delta t), each with the Delta NLL = 0.5 (1-sigma) line."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # Delta t profile
    a = ax[0]
    a.plot(res["dtgrid"], res["prof_dt"], "C0-", lw=1.9)
    a.plot(res["dt_best"], 0.0, "C0o", ms=7)
    a.axhline(0.5, color="0.4", ls=":", label=r"$\Delta$NLL=0.5")
    a.axvline(res["dt_true"], color="k", lw=0.9, label="truth 400")
    if res["dtgrid"].min() < res["dt_min"]:
        a.axvspan(res["dtgrid"].min(), res["dt_min"], color="0.85")
        a.axvline(res["dt_min"], color="r", ls=":", lw=1.4,
                  label=r"bound $\geq %.0f$" % res["dt_min"])
    a.set_ylim(0, float(min(res["prof_dt"].max(), 30)))
    a.set_xlabel(r"$\Delta t$ [Myr]"); a.set_ylabel(r"profile $\Delta$NLL [nats]")
    a.set_title(r"profile over $h$:  $\Delta t = %.2f \pm %.2f$ Myr"
                % (res["dt_best"], res["sig_dt"]))
    a.legend(fontsize=8)
    # h profile
    b = ax[1]
    b.plot(res["hgrid"], res["prof_h"], "C1-", lw=1.9)
    b.plot(res["h_best"], 0.0, "C1o", ms=7)
    b.axhline(0.5, color="0.4", ls=":", label=r"$\Delta$NLL=0.5")
    b.axvline(res["h_true"], color="k", lw=0.9, label="truth 250")
    b.set_ylim(0, float(min(res["prof_h"].max(), 30)))
    b.set_xlabel(r"$h_{\rm dm}$ [pc]"); b.set_ylabel(r"profile $\Delta$NLL [nats]")
    b.set_title(r"profile over $\Delta t$:  $h_{\rm dm} = %.1f \pm %.1f$ pc"
                % (res["h_best"], res["sig_h"]))
    b.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_hdt_trajectories(res, title=None):
    """Per-epoch joint-optimisation traces of both parameters, one line per start.
    Epoch 0 is AFTER the global de-aliasing sweep, so every Delta t trace starts
    on the right comb tooth; the dashed vertical line is where the minibatch
    descent hands over to the full-batch polish."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    cols = plt.cm.autumn(np.linspace(0.0, 0.75, len(res["traj"])))
    for t, c in zip(res["traj"], cols):
        ep = np.arange(len(t["h_hist"]))
        tag = t.get("tag", "tuned")
        ls = "-" if tag == "tuned" else "--"
        lab = "init (%.0f, %.0f)" % (t["h_init"], t["dt_init"])
        if tag != "tuned":
            lab += f" [{tag}]"
        ax[0].plot(ep, t["h_hist"], color=c, lw=1.4, ls=ls, label=lab)
        ax[1].plot(ep, t["dt_hist"], color=c, lw=1.4, ls=ls)
    nep = res["traj"][0].get("n_epochs")
    if nep and res["traj"][0].get("polish"):
        for a in ax:
            a.axvline(nep, color="0.5", ls="--", lw=1.0)
    ax[0].axhline(res["h_true"], color="k", ls="--", lw=1.0, label="truth 250")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel(r"$h_{\rm dm}$ [pc]")
    ax[0].set_title(r"$h_{\rm dm}$ optimisation"); ax[0].legend(fontsize=8)
    ax[1].axhline(res["dt_true"], color="k", ls="--", lw=1.0, label="truth 400")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel(r"$\Delta t$ [Myr]")
    ax[1].set_title(r"$\Delta t$ optimisation (after the de-aliasing sweep)")
    ax[1].legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


# ============================================================================
#   THE LIKELIHOOD LANDSCAPE THE OPTIMISER ACTUALLY WALKED OVER
# ----------------------------------------------------------------------------
#   `res["NLL"]` is a ZOOM a few Myr across, placed around the minimum so that
#   the error bars measure the likelihood rather than the grid.  To draw the
#   optimisation PATHS we need the opposite: the whole prior box, comb and all,
#   wide enough to contain every starting point and every de-aliasing jump.
#   That is ~10^4 back-integrations, so it is a separate, separately-cached step.
# ============================================================================
def _landscape_cache_path(res, n_sub):
    return os.path.join(os.path.dirname(__file__), os.pardir, "results",
                        f"{res.get('id', 'PGT')}_landscape_N{int(res['N'])}"
                        f"_dtmin{int(res['dt_min'])}_sub{int(n_sub)}.pkl")


def hdt_path_landscape(res, z_obs, w_obs, n_sub=4000, dh=12.0, ddt=2.0,
                       pad_h=70.0, dt_max=None, workers=None, cache=True,
                       force=False, verbose=True):
    """Exact profiled NLL on a grid wide enough to contain the whole fit.

    Evaluated on a fixed `n_sub`-star SUBSAMPLE -- the same stars at every grid
    point, so the sampling noise is common-mode and the comb survives -- then
    rescaled to the full sample (the statistic is N x a per-star quantity, so the
    rescaling is exact up to that common-mode noise).  This is ~5x cheaper than
    the 20000-star grid and visually identical; the minimum drawn on top always
    comes from the full-sample profile in `res`, never from this grid.

    Returns dict(hgrid, dtgrid, NLL, n_sub) with NLL shaped (len(dtgrid),
    len(hgrid)) and already offset so its own minimum is 0."""
    path = _landscape_cache_path(res, n_sub)
    if cache and not force and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    tr = res["traj"]
    hs_seen = ([t["h_init"] for t in tr] + [t["h_fit"] for t in tr]
               + [t.get("h_scan", t["h_init"]) for t in tr]
               + [res["h_true"], res["h_best"]])
    h_lo = max(50.0, min(hs_seen) - pad_h)
    h_hi = min(800.0, max(hs_seen) + pad_h)
    dt_hi = float(dt_max if dt_max is not None else DT_MAX)
    hgrid = np.arange(h_lo, h_hi + 1e-9, float(dh))
    dtgrid = np.arange(float(res["dt_min"]), dt_hi + 1e-9, float(ddt))

    rng = np.random.default_rng(0)
    idx = rng.choice(z_obs.size, size=min(int(n_sub), z_obs.size), replace=False)
    zs, ws = z_obs[idx], w_obs[idx]
    if verbose:
        print(f"landscape grid: {len(hgrid)} x {len(dtgrid)} = "
              f"{len(hgrid)*len(dtgrid)} points on {zs.size} stars ...", flush=True)
    Z = grid_nll(zs, ws, hgrid, dtgrid, dt_step=res.get("dt_step", DT_STEP),
                 workers=workers)
    Z = (Z - Z.min()) * (float(res["N"]) / zs.size)     # -> full-sample nats

    out = dict(hgrid=hgrid, dtgrid=dtgrid, NLL=Z, n_sub=int(zs.size))
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out


def _comb_teeth(Z, half=3):
    """The comb teeth: local minima of the likelihood PROFILED over h_dm, i.e.
    of `Z.min(axis=1)` along Delta t, each returned with the h that attains it.

    Deliberately not a 2-D neighbourhood test.  The surface is a razor comb in
    Delta t but almost flat in h, so 2-D local minima land wherever the grid's
    own sampling noise dips -- scattered dots implying structure in h that is not
    there.  The teeth are a property of the profile, and the profile is what a
    local optimiser is actually trapped by.

    `half` is the half-width in grid points of the window a point must be the
    minimum of (default +-3, i.e. +-6 Myr at the default 2 Myr spacing -- well
    inside the ~20 Myr tooth spacing).  Returns (dt_index, h_index) pairs."""
    prof = Z.min(axis=1)
    n = prof.size
    keep = []
    for i in range(half, n - half):
        win = prof[i - half:i + half + 1]
        if prof[i] == win.min() and prof[i] < min(prof[i - 1], prof[i + 1]):
            keep.append(i)
    return np.array([(i, int(Z[i].argmin())) for i in keep], dtype=int) \
        if keep else np.empty((0, 2), dtype=int)


def hdt_stiffness(res, level=4.0):
    """Curvature of the profile likelihood at its minimum: how much better
    Delta t is measured than h_dm, and along which direction the two trade off.

    A 2-D quadratic is least-squares fitted to every zoom-grid point within
    `level` nats of the minimum, giving the Hessian of the NLL in (h, Delta t).
    It is then rescaled to FRACTIONAL coordinates (h/h_true, Delta t/Delta t_true)
    -- the parameters have different units, so only a dimensionless comparison
    means anything -- and diagonalised.

    Returns a dict with the two sigmas, their ratio, the correlation, the
    stiff/soft eigen-directions and the degeneracy slope d(Delta t)/dh."""
    H, D, Z = res["hgrid"], res["dtgrid"], res["NLL"]
    j, i = np.unravel_index(Z.argmin(), Z.shape)
    HH, DD = np.meshgrid(H, D)
    m = Z < Z.min() + float(level)
    x, y, z = HH[m] - H[i], DD[m] - D[j], Z[m] - Z.min()
    A = np.c_[x * x, x * y, y * y, x, y, np.ones_like(x)]
    c = np.linalg.lstsq(A, z, rcond=None)[0]
    Hess = np.array([[2 * c[0], c[1]], [c[1], 2 * c[2]]])
    cov = np.linalg.inv(Hess)
    S = np.diag([res["h_true"], res["dt_true"]])          # -> fractional units
    Hd = S @ Hess @ S
    w, v = np.linalg.eigh(Hd)                             # ascending: soft first
    covd = np.linalg.inv(Hd)
    soft = v[:, 0] * (1.0 if v[0, 0] >= 0 else -1.0)      # point it along +h
    return dict(
        hess=Hess, cov=cov, n_points=int(m.sum()),
        sigma_h=float(np.sqrt(cov[0, 0])), sigma_dt=float(np.sqrt(cov[1, 1])),
        frac_h=float(np.sqrt(covd[0, 0])), frac_dt=float(np.sqrt(covd[1, 1])),
        ratio=float(np.sqrt(covd[0, 0] / covd[1, 1])),
        corr=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
        curv_ratio=float(Hd[1, 1] / Hd[0, 0]), cond=float(w[1] / w[0]),
        soft_dir=soft, stiff_dir=v[:, 1],
        slope=float(soft[1] * res["dt_true"] / (soft[0] * res["h_true"])))


def hdt_stiffness_text(res, level=4.0):
    """`hdt_stiffness` as a short human-readable block."""
    s = hdt_stiffness(res, level)
    return "\n".join([
        f"curvature of the profile likelihood at the minimum "
        f"({s['n_points']} grid points within {level:.0f} nats)",
        f"  sigma(h_dm)   = {s['sigma_h']:7.1f} pc   = {s['frac_h']*100:6.2f} % of the truth",
        f"  sigma(Delta t)= {s['sigma_dt']:7.2f} Myr  = {s['frac_dt']*100:6.3f} % of the truth",
        f"  -> Delta t is measured {s['ratio']:.0f}x better in relative terms",
        f"     (the NLL curves {s['curv_ratio']:.0f}x more steeply in fractional",
        f"      Delta t than in fractional h; the precision ratio is its sqrt)",
        f"  correlation(h, Delta t) = {s['corr']:+.3f}   "
        f"-> a ridge of slope {s['slope']:+.3f} Myr/pc",
        f"  anisotropy (stiff/soft eigenvalue) = {s['cond']:.0f}",
        f"  stiff direction (d h/h, d t/t) = "
        f"({s['stiff_dir'][0]:+.3f}, {s['stiff_dir'][1]:+.3f})  -- essentially pure Delta t",
        f"  soft  direction (d h/h, d t/t) = "
        f"({s['soft_dir'][0]:+.3f}, {s['soft_dir'][1]:+.3f})  -- essentially pure h_dm",
    ])


def plot_hdt_paths(res, z_obs, w_obs, title=None, land=None, zoom_h=70.0,
                   zoom_dt=6.0, stiffness=True, **grid_kw):
    """THE figure: the (h_dm, Delta t) profile likelihood as a heatmap -- x is
    h_dm, y is the total time Delta t -- with the optimisation path of every
    starting point drawn on top of it.

    Each start contributes three things:
      o  the initial guess;
      -> a dashed arrow to where the global de-aliasing sweep moved it (absent
         for runs made without the sweep, where the arrow has zero length);
      -  the solid per-epoch descent track, ending in a square at the landing.

    White carets on the Delta t axis are the comb teeth -- the local minima each
    of which traps a purely local optimiser.  The inset zooms on the global
    minimum so the landings, which coincide at the scale of the main panel, can
    be told apart from the truth; with `stiffness` it also draws the principal
    axes of the likelihood there, which is where the h_dm/Delta t degeneracy
    lives."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    land = land if land is not None else hdt_path_landscape(res, z_obs, w_obs,
                                                            **grid_kw)
    H, D, Z = land["hgrid"], land["dtgrid"], land["NLL"]
    fig, ax = plt.subplots(figsize=(11.6, 8.4))

    # Z is (len(D), len(H)); x = h, y = Delta t -> no transpose needed
    pc = ax.pcolormesh(H, D, Z + 1.0, cmap="viridis_r", shading="nearest",
                       norm=LogNorm(vmin=1.0, vmax=max(Z.max(), 10.0)))
    cb = fig.colorbar(pc, ax=ax, pad=0.015)
    cb.set_label(r"profile $\Delta$NLL + 1 [nats, full sample]")

    # The teeth are a Delta t structure -- at fixed Delta t the surface is an
    # almost flat valley in h -- so they are marked ON THE Delta t AXIS.  Drawing
    # them at the h that happens to minimise each tooth would paint grid noise as
    # if it were structure in h.
    loc = _comb_teeth(Z)
    if len(loc):
        x0 = H.min() + 0.012 * (H.max() - H.min())
        for i in loc[:, 0]:
            ax.axhline(D[i], color="w", lw=0.6, ls=(0, (3, 4)), alpha=0.30,
                       zorder=3)
        ax.plot(np.full(len(loc), x0), D[loc[:, 0]], marker=5, ls="none",
                color="w", mec="k", mew=0.5, ms=9, zorder=4,
                label=f"local minima in $\\Delta t$ — the comb teeth ({len(loc)})")

    tr = res["traj"]
    cols = plt.cm.cool(np.linspace(0.0, 1.0, max(len(tr), 2)))
    for t, c in zip(tr, cols):
        tuned = t.get("tag", "tuned") == "tuned"
        hsc = t.get("h_scan", t["h_init"]); dsc = t.get("dt_scan", t["dt_init"])
        ax.plot(t["h_init"], t["dt_init"], "o", color=c, mec="k", mew=0.8,
                ms=10, zorder=6)
        if abs(dsc - t["dt_init"]) > 1e-9 or abs(hsc - t["h_init"]) > 1e-9:
            ax.annotate("", xy=(hsc, dsc), xytext=(t["h_init"], t["dt_init"]),
                        zorder=5, arrowprops=dict(arrowstyle="-|>", color=c,
                                                  lw=1.8, ls=(0, (5, 2)),
                                                  shrinkA=5, shrinkB=1))
        ax.plot(t["h_hist"], t["dt_hist"], "-" if tuned else "--", color=c,
                lw=1.8, zorder=5)
        ax.plot(t["h_fit"], t["dt_fit"], "s" if tuned else "X", color=c,
                mec="k", mew=0.8, ms=9, zorder=7)

    ax.plot(res["h_true"], res["dt_true"], "*", color="w", mec="k", ms=22,
            zorder=8, label="truth (%.0f pc, %.0f Myr)"
                            % (res["h_true"], res["dt_true"]))
    ax.plot(res["h_best"], res["dt_best"], "P", color="r", mec="k", ms=13,
            zorder=8, label="global minimum (%.1f pc, %.2f Myr)"
                            % (res["h_best"], res["dt_best"]))
    ax.plot([], [], "o", color="0.6", mec="k", ms=9, label="initial guess")
    ax.plot([], [], "-", color="0.6", lw=1.8, label="descent path")
    ax.plot([], [], "s", color="0.6", mec="k", ms=8, label="landing")
    if any(t.get("tag", "tuned") != "tuned" for t in tr):
        ax.plot([], [], "X", color="0.6", mec="k", ms=9,
                label="landing, sweep DISABLED")

    ax.set_xlabel(r"dark-matter scale height $h_{\rm dm}$ [pc]")
    ax.set_ylabel(r"total dynamical time $\Delta t$ [Myr]")
    ax.set_xlim(H.min(), H.max()); ax.set_ylim(D.min(), D.max())
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92, ncol=2)

    # ---- inset: the landings, which overlap at the scale of the main panel ---
    # upper right: with Delta t on y the interesting paths run along the bottom
    # and the middle, so the top-right corner is the free space.
    axi = ax.inset_axes([0.615, 0.615, 0.365, 0.29])
    hz, dz, Zz = res["hgrid"], res["dtgrid"], res["NLL"]
    axi.contourf(hz, dz, Zz, levels=np.linspace(0, min(Zz.max(), 12.0), 20),
                 cmap="viridis_r", extend="max")
    axi.contour(hz, dz, Zz, levels=[0.5, 2.0], colors="w", linewidths=0.7)
    if stiffness:
        s = hdt_stiffness(res)
        # the soft (degenerate) direction, drawn through the minimum
        dh = 0.9 * zoom_h
        ddt = dh * s["slope"]
        axi.annotate("", xy=(res["h_best"] + dh, res["dt_best"] + ddt),
                     xytext=(res["h_best"] - dh, res["dt_best"] - ddt),
                     arrowprops=dict(arrowstyle="<|-|>", color="w", lw=1.4))
        axi.text(0.03, 0.05, f"soft direction\n{s['slope']:+.3f} Myr/pc\n"
                             f"$\\Delta t$ {s['ratio']:.0f}$\\times$ tighter",
                 transform=axi.transAxes, fontsize=7, color="w", va="bottom")
    # landings LAST here: separating them from the truth is the inset's whole job
    axi.plot(res["h_true"], res["dt_true"], "*", color="w", mec="k", ms=17, zorder=6)
    axi.plot(res["h_best"], res["dt_best"], "P", color="r", mec="k", ms=10, zorder=7)
    for t, c in zip(tr, cols):
        if t.get("tag", "tuned") == "tuned":
            axi.plot(t["h_fit"], t["dt_fit"], "s", color=c, mec="k", mew=0.8,
                     ms=8, zorder=9)
    axi.set_xlim(res["h_best"] - zoom_h, res["h_best"] + zoom_h)
    axi.set_ylim(res["dt_best"] - zoom_dt, res["dt_best"] + zoom_dt)
    axi.tick_params(labelsize=8)
    axi.set_title("zoom on the minimum: every landing", fontsize=8.5)
    for sp in axi.spines.values():
        sp.set_color("w"); sp.set_linewidth(1.2)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


HOP_STARTS = ((150, 250), (220, 300), (250, 400), (240, 250), (290, 520),
              (350, 600), (400, 220), (180, 550), (320, 350), (500, 380),
              (120, 430), (250, 100), (250, 0), (250, 690), (600, 60),
              (80, 660))


def _hop_worker(job):
    """One basin-hopping fit in a pool worker (module level so it pickles)."""
    (hi, dti), seed, kw = job
    torch.set_num_threads(1)
    g = _GRID_CTX
    z, w = g["z"], g["w"]
    info = {}
    t0 = time.time()
    h, dt, traj = basin_hop_h_dt(z, w, hi, dti, seed=seed, info=info, **kw)
    ds = kw.get("dt_step", DT_STEP)
    key = kw.get("flow_full") or kw.get("flow", "affine")
    if key in PROFILE_NLL and not kw.get("lam"):
        prof = PROFILE_NLL[key]
        dnll = prof(z, w, h, dt, dt_step=ds) - prof(z, w, H_TRUE, DT_TRUE, dt_step=ds)
    else:                       # penalised / no profile objective: J is in info
        dnll = float("nan")
    return dict(h_init=hi, dt_init=dti, seed=seed, h_fit=h, dt_fit=dt,
                dnll=dnll, traj=traj, wall=time.time() - t0, **info)


def run_hop_study(z, w, starts=HOP_STARTS, seeds=(0, 1), dt_min=0.0,
                  dt_max=700.0, dt_step=DT_STEP, dh=12.0, ddt=2.0, n_sub=4000,
                  workers=None, cache=True, force=False, verbose=True,
                  flow="affine", data_tag="", **kw):
    """The convergence study: one basin-hopping fit per (start, seed), plus a
    landscape grid wide enough to hold every path, cached together.

    This is the experiment `dealias_scan` cannot support -- the sweep discards
    the initial guess, so every start is the same computation and the spread
    across starts is identically zero.  Here each start seeds its own stochastic
    search, the routes genuinely differ, and the agreement of the endpoints is a
    measurement that was free to fail.

    Cached in results/PGT_hop_N{N}.pkl (~25 min for 32 fits on 8 workers, plus
    ~4 min for the grid); `force=True` recomputes."""
    # dt_min belongs in the key: it changes both the search box and the grid, so
    # a dt_min=200 study must not silently reuse the dt_min=0 one.  Likewise
    # `data_tag` (e.g. "2w"): a non-Gaussian dataset must not reuse the PGT cache.
    tag = "" if flow == "affine" else f"_{flow}"
    if kw.get("local", "nm") == "joint":     # joint local fits: separate cache
        tag += "_joint"
    if kw.get("lam"):
        tag += f"_{kw.get('pen', 'fisher')}{kw['lam']:g}"
    pre = f"PGT{data_tag}" if data_tag else "PGT"
    path = os.path.join(os.path.dirname(__file__), os.pardir, "results",
                        f"{pre}_hop_N{int(z.size)}_dtmin{int(dt_min)}{tag}.pkl")
    if cache and not force and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    fit_kw = dict(dt_min=dt_min, dt_max=dt_max, dt_step=dt_step, n_sub=n_sub,
                  flow=flow, **kw)
    jobs = [(tuple(s), int(sd), fit_kw) for s in starts for sd in seeds]
    nw = (max((os.cpu_count() or 2) - 2, 1)) if workers is None else workers
    if verbose:
        print(f"basin hopping: {len(jobs)} fits on {nw} workers ...", flush=True)
    t0 = time.time()
    try:
        from multiprocessing import Pool
        with Pool(min(nw, len(jobs)), initializer=_grid_init,
                  initargs=(z, w, dt_step, 1)) as p:
            runs = p.map(_hop_worker, jobs, chunksize=1)
    except Exception:
        _grid_init(z, w, dt_step)
        runs = [_hop_worker(j) for j in jobs]
    if verbose:
        print(f"  fits done in {time.time()-t0:.0f}s", flush=True)

    hs = [r["h_init"] for r in runs] + [r["h_fit"] for r in runs]
    for r in runs:
        hs += [t[0] for t in r["traj"]]
    hgrid = np.arange(max(50.0, min(hs) - 40.0),
                      min(800.0, max(hs) + 40.0) + 1e-9, float(dh))
    dtgrid = np.arange(float(dt_min), float(dt_max) + 1e-9, float(ddt))
    if verbose:
        print(f"landscape grid {len(hgrid)} x {len(dtgrid)} ...", flush=True)
    rng = np.random.default_rng(0)
    idx = rng.choice(z.size, size=min(n_sub, z.size), replace=False)
    # Background landscape for the path figure.  A flow with no cheap profiled
    # objective (the penalised zw114) gets the AFFINE landscape, flagged in
    # hop["grid_flow"] so the figure can say so.
    grid_flow = flow if (flow in PROFILE_NLL and not kw.get("lam")) else "affine"
    Z = grid_nll(z[idx], w[idx], hgrid, dtgrid, dt_step=dt_step, workers=nw,
                 flow=grid_flow)
    Z = (Z - Z.min()) * (float(z.size) / idx.size)

    hop = dict(id="PGThop", N=int(z.size), runs=runs, hgrid=hgrid,
               dtgrid=dtgrid, NLL=Z, n_sub=int(idx.size), h_true=H_TRUE,
               dt_true=DT_TRUE, dt_step=float(dt_step), dt_min=float(dt_min),
               flow=flow, flow_full=kw.get("flow_full") or flow,
               data_tag=data_tag, local=kw.get("local", "nm"),
               lam=kw.get("lam", 0.0), pen=kw.get("pen", "fisher"),
               grid_flow=grid_flow)
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(hop, f)
    return hop


def _hop_moves(run, tol_h=0.05, tol_dt=0.005):
    """The ACCEPTED states of one run, near-duplicates collapsed.

    A greedy walker sits still on a rejected proposal, and Nelder-Mead re-lands
    a few 1e-5 away when it does move, so the raw trajectory is full of repeats
    that would draw as invisible zero-length segments."""
    out = []
    for t in run["traj"]:
        if (not out or abs(t[0] - out[-1][0]) > tol_h
                or abs(t[1] - out[-1][1]) > tol_dt):
            out.append(t)
    return out


def plot_hdt_hop_paths(hop, title=None, zoom=None, n_paths=3, prof=None,
                       show_rejected=False, seed=0):
    """The convergence figure, with the two halves of a basin-hopping step drawn
    as the different things they are.

    LEFT -- the likelihood landscape.  One hop is:

      * a **dashed straight** arrow, the JUMP: a random step in
        (h_dm, Delta t).  No likelihood is consulted, so it may land anywhere --
        which is how the 100-300-nat barriers between comb teeth are crossed.
        The walker never climbs them, it steps over.
      * a **solid winding** line, the MINIMISATION: the actual Nelder-Mead walk
        (20-60 steps) from where the jump landed down to the bottom of whatever
        tooth it found.  Drawn as the path it is, not a straight arrow.
      * the pair is kept only if that bottom is lower than the one it came from.
        Rejected pairs are faint grey, so the exploration that led nowhere is
        visible too.

    RIGHT -- the (h_dm, Delta t) plane: every run's final fit with the 1-sigma
    error bars of the profile likelihood, the truth as a star, and an inset
    magnifying the landing points.  Pass `prof` (from `C.hop_profile`).

    SQUARE = initial guess, CIRCLE = final fit.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LogNorm

    H, D, Z = hop["hgrid"], hop["dtgrid"], hop["NLL"]
    runs = hop["runs"]
    hf = np.array([r["h_fit"] for r in runs])
    df = np.array([r["dt_fit"] for r in runs])

    firsts, seen = [], set()
    for k, r in enumerate(runs):
        key = (r["h_init"], r["dt_init"])
        if key not in seen:
            seen.add(key); firsts.append(k)
    dist = [((runs[k]["dt_init"] - hop["dt_true"]) / 100.0) ** 2
            + ((runs[k]["h_init"] - hop["h_true"]) / 100.0) ** 2
            for k in firsts]
    ranked = [firsts[i] for i in np.argsort(dist)]
    # a start AT the truth has a zero-length route -- nothing to show
    far = [k for k in ranked
           if abs(runs[k]["h_init"] - hop["h_true"]) > 1.0
           or abs(runs[k]["dt_init"] - hop["dt_true"]) > 1.0] or ranked
    n = int(min(n_paths, len(far)))
    chosen = [far[int(round(t))] for t in np.linspace(0, len(far) - 1, n)]

    fig = plt.figure(figsize=(16.4, 8.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.24,
                          bottom=0.22, top=0.90)
    ax = fig.add_subplot(gs[0, 0])
    axr = fig.add_subplot(gs[0, 1])

    pc = ax.pcolormesh(H, D, Z + 1.0, cmap="viridis_r", shading="nearest",
                       norm=LogNorm(vmin=1.0, vmax=max(Z.max(), 10.0)))
    cb = fig.colorbar(pc, ax=ax, pad=0.015)
    if hop.get("grid_flow", hop.get("flow", "affine")) != hop.get("flow", "affine"):
        cb.set_label("BACKGROUND ONLY: affine (fixed-Gaussian) profile "
                     r"$\Delta$NLL + 1 -- not the fitted objective")
    else:
        cb.set_label(r"profile $\Delta$NLL + 1 [nats, full sample]")

    loc = _comb_teeth(Z)
    if len(loc):
        x0 = H.min() + 0.012 * (H.max() - H.min())
        for i in loc[:, 0]:
            ax.axhline(D[i], color="w", lw=0.5, ls=(0, (3, 5)), alpha=0.18,
                       zorder=3)
        ax.plot(np.full(len(loc), x0), D[loc[:, 0]], marker=5, ls="none",
                color="w", mec="k", mew=0.5, ms=8, zorder=4,
                label=f"comb teeth ({len(loc)} local minima in $\\Delta t$)")

    txt_halo = [pe.Stroke(linewidth=2.4, foreground="k"), pe.Normal()]
    dark = [pe.Stroke(linewidth=3.4, foreground="k", alpha=0.55), pe.Normal()]
    # colours chosen to stand off the viridis background (no greens/blues)
    pal = ["#ff2a2a", "#ff9f1c", "#ff5ec8", "#ffffff", "#c77dff"]
    cols = [pal[i % len(pal)] for i in range(len(chosen))]
    split_ok = all(runs[k].get("hops") for k in chosen)

    for k, c in zip(chosen, cols):
        r = runs[k]
        hops = r.get("hops")
        if not hops:                     # cache predates the jump/min split
            mv = _hop_moves(r)
            ax.plot([m[0] for m in mv], [m[1] for m in mv], "-", color=c,
                    lw=2.0, alpha=0.95, zorder=5, path_effects=dark)
            ax.plot(mv[0][0], mv[0][1], "s", color=c, mec="k", mew=1.3, ms=15,
                    zorder=8)
            continue
        for hp in hops:
            a, p = hp["frm"], hp["prop"]
            mp = hp.get("mpath") or [p, hp["land"]]
            if not hp["accepted"]:
                if show_rejected:
                    ax.annotate("", xy=p, xytext=a, zorder=4,
                                arrowprops=dict(arrowstyle="-|>", color="0.78",
                                                lw=1.0, ls=(0, (2, 2)),
                                                alpha=0.55, shrinkA=3,
                                                shrinkB=3))
                    ax.plot([q[0] for q in mp], [q[1] for q in mp], "-",
                            color="0.78", lw=1.0, alpha=0.55, zorder=4)
                    ax.plot(*hp["land"], "x", color="0.78", ms=5, alpha=0.7,
                            zorder=4)
                continue
            if hp["kind"] == "hop":                       # the JUMP
                ax.annotate("", xy=p, xytext=a, zorder=6,
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.7,
                                            ls=(0, (4, 3)), alpha=0.95,
                                            shrinkA=4, shrinkB=3,
                                            path_effects=dark))
                ax.plot(*p, "^", color=c, mec="k", mew=0.7, ms=9, zorder=7)
            ax.plot([q[0] for q in mp], [q[1] for q in mp], "-", color=c,
                    lw=2.2, alpha=0.95, zorder=6, path_effects=dark)  # the MIN
            ax.plot(*hp["land"], "o", color=c, mec="k", mew=0.6, ms=7, zorder=7)
        x0_, y0_ = hops[0]["frm"]
        ax.plot(x0_, y0_, "s", color=c, mec="k", mew=1.3, ms=15, zorder=9)
        ax.annotate(f"({r['h_init']:.0f}, {r['dt_init']:.0f})", xy=(x0_, y0_),
                    xytext=(10, 9), textcoords="offset points", fontsize=8.5,
                    color="w", fontweight="bold", zorder=11,
                    path_effects=txt_halo)

    ax.plot(hop["h_true"], hop["dt_true"], "*", color="w", mec="k", ms=24,
            zorder=10, label="truth (%.0f pc, %.0f Myr)"
                             % (hop["h_true"], hop["dt_true"]))
    for k, c in zip(chosen, cols):
        ax.plot(runs[k]["h_fit"], runs[k]["dt_fit"], "o", color=c, mec="k",
                mew=1.4, ms=12, zorder=11)

    ax.plot([], [], "s", color="0.55", mec="k", ms=12, label="initial guess")
    if split_ok:
        ax.plot([], [], ls=(0, (4, 3)), color="0.55", lw=1.7,
                label="JUMP — random step, straight, no likelihood")
        ax.plot([], [], "^", color="0.55", mec="k", ms=9,
                label="where the jump landed")
        ax.plot([], [], "-", color="0.55", lw=2.2,
                label="MINIMISATION — the Nelder-Mead walk downhill")
        if show_rejected:
            ax.plot([], [], "-", color="0.78", lw=1.0,
                    label="rejected hop (no improvement)")
    else:
        ax.plot([], [], "-", color="0.55", lw=2.0, label="route (accepted states)")
    ax.plot([], [], "o", color="0.55", mec="k", ms=10, label="final fit")

    # every drawn point must be inside the axes: the walker reaches the h and
    # Delta t clamps, which sit half a grid cell outside the pcolormesh nodes.
    dx = 0.5 * float(H[1] - H[0]) if len(H) > 1 else 1.0
    dy = 0.5 * float(D[1] - D[0]) if len(D) > 1 else 1.0
    ax.set_xlim(H.min() - dx, H.max() + dx)
    ax.set_ylim(D.min() - dy, D.max() + dy)
    ax.set_xlabel(r"dark-matter scale height $h_{\rm dm}$ [pc]")
    ax.set_ylabel(r"total dynamical time $\Delta t$ [Myr]")
    ax.set_title("one hop = a dashed JUMP, then a solid MINIMISATION"
                 if split_ok else "route through the accepted states",
                 fontsize=11)
    # legend BELOW the panel, never on top of the data
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=3, framealpha=0.95, borderaxespad=0.0)

    # ---- right: (h_dm, Delta t) plane -- every fit with its 1-sigma bars ----
    # All runs land within ~0.04 pc / 0.002 Myr of each other, so on the scale of
    # the error bars they coincide; the inset magnifies the landing points so
    # each fit is individually visible.
    order = sorted(range(len(runs)), key=lambda k: (runs[k]["h_init"],
                                                    runs[k]["dt_init"],
                                                    runs[k]["seed"]))
    allc = plt.cm.turbo(np.linspace(0.04, 0.96, len(order)))
    sh = prof["sig_h"] if prof else None
    sd = prof["sig_dt"] if prof else None
    for k, c in zip(order, allc):
        axr.errorbar(hf[k], df[k], xerr=sh, yerr=sd, fmt="o", color=c,
                     mec="k", mew=0.5, ms=7, ecolor=c, elinewidth=1.3,
                     capsize=5, capthick=1.3, alpha=0.9, zorder=6)
    axr.plot(hop["h_true"], hop["dt_true"], "*", color="gold", mec="k", ms=24,
             zorder=9, label="truth (%.0f pc, %.0f Myr)"
                             % (hop["h_true"], hop["dt_true"]))
    axr.errorbar([], [], xerr=[], yerr=[], fmt="o", color="0.6", mec="k", ms=7,
                 ecolor="0.6", capsize=5,
                 label=f"each of the {len(runs)} final fits"
                       + (r" $\pm 1\sigma$" if prof else ""))
    if prof:
        axr.axvline(hop["h_true"], color="0.6", ls=":", lw=1.0, zorder=2)
        axr.axhline(hop["dt_true"], color="0.6", ls=":", lw=1.0, zorder=2)
        cx = 0.5 * (hf.mean() + hop["h_true"])
        cy = 0.5 * (df.mean() + hop["dt_true"])
        axr.set_xlim(cx - 2.3 * sh, cx + 2.3 * sh)
        axr.set_ylim(cy - 2.3 * sd, cy + 2.3 * sd)
        axr.set_title(
            f"every fit vs the truth, with $1\\sigma$ error bars\n"
            f"off by {abs(hf.mean()-hop['h_true'])/sh:.2f}$\\sigma$ in "
            f"$h_{{\\rm dm}}$ ($\\sigma$={sh:.1f} pc), "
            f"{abs(df.mean()-hop['dt_true'])/sd:.2f}$\\sigma$ in "
            f"$\\Delta t$ ($\\sigma$={sd:.2f} Myr)", fontsize=10)
        # inset: the landing points magnified, so the individual fits separate
        ins = axr.inset_axes([0.62, 0.04, 0.35, 0.30])
        for k, c in zip(order, allc):
            ins.plot(hf[k], df[k], "o", color=c, mec="k", mew=0.4, ms=5)
        ph = max(np.ptp(hf), 1e-3) * 0.25
        pdt = max(np.ptp(df), 1e-4) * 0.25
        ins.set_xlim(hf.min() - ph, hf.max() + ph)
        ins.set_ylim(df.min() - pdt, df.max() + pdt)
        ins.tick_params(labelsize=6.5)
        ins.ticklabel_format(useOffset=False)
        ins.set_title(f"zoom: spread {np.ptp(hf):.3f} pc, {np.ptp(df):.4f} Myr",
                      fontsize=7)
        ins.grid(alpha=0.3)
    else:
        axr.set_title(f"the {len(runs)} final fits\n(pass prof= for error bars)",
                      fontsize=11)
    axr.set_xlabel(r"$h_{\rm dm}$ [pc]")
    axr.set_ylabel(r"$\Delta t$ [Myr]")
    axr.grid(alpha=0.3)
    axr.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09),
               ncol=1, framealpha=0.95, borderaxespad=0.0)

    if title:
        fig.suptitle(title, fontsize=13)
    return fig


def hop_profile(hop, z, w, h_half=100.0, n_h=25, dt_half=8.0, dt_fine=0.25,
                comb_step=1.0, workers=None, cache=True, force=False,
                verbose=True):
    """Exact profile likelihood on a fine grid AROUND the basin-hopping minimum.

    Same "fit first, then profile" discipline as everywhere else in this project:
    the grid is placed where the fit landed, never on a pre-chosen range, so the
    error bars measure the likelihood rather than the grid spacing.

    Returns a dict with the same keys `hdt_stiffness`, `_parab_sigma` and the
    plotting helpers expect (hgrid, dtgrid, NLL, prof_h, prof_dt, sig_h, sig_dt,
    h_best, dt_best), computed on the FULL sample."""
    pre = f"PGT{hop['data_tag']}" if hop.get("data_tag") else "PGT"
    path = os.path.join(os.path.dirname(__file__), os.pardir, "results",
                        f"{pre}_hopprof_N{int(z.size)}_dtmin{int(hop['dt_min'])}"
                        f"{'' if hop.get('flow','affine')=='affine' else '_'+hop['flow']}"
                        f"{'_joint' if hop.get('local') == 'joint' else ''}.pkl")
    if cache and not force and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Centre on the LOWEST-NLL landing, not the mean: one run stuck on a decoy
    # tooth (the nvp16 study's h=800 clamp) dragged the mean 65 pc off and put
    # the -1 sigma point outside the grid.
    best = min(hop["runs"], key=lambda r: r["dnll"])
    h_hat, dt_hat = float(best["h_fit"]), float(best["dt_fit"])
    ds = hop.get("dt_step", DT_STEP)
    nw = (max((os.cpu_count() or 2) - 2, 1)) if workers is None else workers

    hgrid = np.linspace(h_hat - h_half, h_hat + h_half, int(n_h))
    lo = max(float(hop["dt_min"]), dt_hat - dt_half)
    dtgrid = np.arange(lo, dt_hat + dt_half + 1e-9, float(dt_fine))
    if verbose:
        print(f"profile grid {len(hgrid)} x {len(dtgrid)} around "
              f"({h_hat:.2f} pc, {dt_hat:.3f} Myr) on {z.size} stars ...",
              flush=True)
    fl = hop.get("flow_full") or hop.get("flow", "affine")   # full-sample budget
    NLL = grid_nll(z, w, hgrid, dtgrid, dt_step=ds, workers=nw, flow=fl)

    comb_dt = np.arange(float(hop["dt_min"]), DT_MAX + 1e-9, float(comb_step))
    comb = grid_nll(z, w, [h_hat], comb_dt, dt_step=ds, workers=nw,
                    flow=fl).ravel()

    ref = NLL.min()
    NLL = NLL - ref
    comb = comb - ref
    prof_dt = NLL.min(axis=1)              # profile over h  -> dNLL(Delta t)
    prof_h = NLL.min(axis=0)               # profile over dt -> dNLL(h)
    sig_dt, dt_best = _parab_sigma(dtgrid, prof_dt)
    sig_h, h_best = _parab_sigma(hgrid, prof_h)

    out = dict(id="PGThopprof", hgrid=hgrid, dtgrid=dtgrid, NLL=NLL,
               comb_dt=comb_dt, comb_nll=comb, prof_h=prof_h, prof_dt=prof_dt,
               sig_h=sig_h, sig_dt=sig_dt, h_best=h_best, dt_best=dt_best,
               h_hat=h_hat, dt_hat=dt_hat, N=int(z.size), h_true=hop["h_true"],
               dt_true=hop["dt_true"], dt_step=float(ds),
               dt_min=float(hop["dt_min"]),
               n_steps_best=dt_best / float(ds))
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out


def hop_fit_table(hop, z=None, w=None):
    """One row per basin-hopping run: the Hamiltonian-flow parameters, the
    search effort, and the four bijection parameters.

    The bijection parameters are the closed-form optimum at the landing point
    (a = sd, b = mean of the back-integrated cloud) -- for the diagonal-affine
    flow that IS the maximum-likelihood fit, verified against an explicit
    numerical minimisation to 3e-11 nats."""
    import pandas as pd
    ds = hop.get("dt_step", DT_STEP)
    rows = []
    for k, r in enumerate(sorted(hop["runs"],
                                 key=lambda r: (r["h_init"], r["dt_init"],
                                                r["seed"]))):
        nf = {}
        if z is not None:
            zb, wb = backint(z, w, r["h_fit"], t_obs=r["dt_fit"], dt_step=ds)
            nf = dict(a_z=float(zb.std()), a_w=float(wb.std()),
                      b_z=float(zb.mean()), b_w=float(wb.mean()))
        rows.append(dict(
            fit=k, h_init=float(r["h_init"]), dt_init=float(r["dt_init"]),
            seed=int(r["seed"]),
            h_hop=float(r.get("h_hop", np.nan)),
            dt_hop=float(r.get("dt_hop", np.nan)),
            h_fit=float(r["h_fit"]), dt_fit=float(r["dt_fit"]),
            N_steps=float(r["dt_fit"]) / ds,
            dh=float(r["h_fit"]) - hop["h_true"],
            ddt=float(r["dt_fit"]) - hop["dt_true"],
            dNLL=float(r["dnll"]), hops=int(r.get("n_hops", 0)),
            evals=int(r.get("n_eval_sub", 0) + r.get("n_eval_full", 0)),
            wall_s=float(r.get("wall", np.nan)), **nf))
    df = pd.DataFrame(rows).set_index("fit")
    df.attrs.update(h_true=hop["h_true"], dt_true=hop["dt_true"], dt_step=ds,
                    N_steps_true=hop["dt_true"] / ds, N=hop["N"],
                    z0_sigma=Z0_SIGMA, w0_sigma=W0_SIGMA)
    return df


def hop_results_table(studies, z, w, cache=True, force=False):
    """Every basin-hopping fit of several studies in ONE DataFrame, one row per
    fit -- the numbers behind figures 2 and 3.

    `studies` maps a figure label to (hop, prof), e.g.
        {"Fig 2 (affine)": (hop, prof), "Fig 3 (NVP)": (hop_nvp, prof_nvp)}.

    Columns, per fit:
      identity   figure, flow, n_flow_params, run, seed
      start      h_init, dt_init
      hopping    h_hop, dt_hop (4000-star subsample), hops, evals, wall_s
      result     h_fit +/- sig_h, dt_fit +/- sig_dt, N_steps = dt_fit/dt_step
      vs truth   dh, ddt, pull_h = dh/sig_h, pull_dt = ddt/sig_dt
      quality    NLL (absolute, full-sample profiled, at the landing point),
                 dNLL (vs the profile-grid minimum), NLL_per_star
      bijection  a_z, a_w, b_z, b_w -- the diagonal-affine part (sd / mean of
                 the back-integrated cloud; for the affine flow this IS the fit,
                 for the NVP flow it is the fixed standardisation);
                 nvp_W_s, nvp_W_t, nvp_c_s, nvp_c_t -- the 4 trainable weights
                 of the NVP coupling (NaN for the affine flow).

    sig_h, sig_dt are the 1-sigma profile-likelihood errors of that study (the
    fits all land on the same point, so every fit carries the same error).
    The bijection needs one back-integration per fit (~1 s each); cached in
    results/PGT_hoptable_*.pkl."""
    import pandas as pd
    frames = []
    for label, (hop, prof) in studies.items():
        fl = hop.get("flow", "affine")
        ds = float(hop.get("dt_step", DT_STEP))
        path = os.path.join(os.path.dirname(__file__), os.pardir, "results",
                            f"PGT_hoptable_N{int(z.size)}_dtmin{int(hop['dt_min'])}"
                            f"{'' if fl == 'affine' else '_' + fl}.pkl")
        if cache and not force and os.path.exists(path):
            with open(path, "rb") as f:
                rows = pickle.load(f)
        else:
            rows = []
            runs = sorted(hop["runs"], key=lambda r: (r["h_init"], r["dt_init"],
                                                      r["seed"]))
            for k, r in enumerate(runs):
                h, dt = float(r["h_fit"]), float(r["dt_fit"])
                zb, wb = backint(z, w, h, t_obs=dt, dt_step=ds)
                nf = dict(a_z=float(zb.std()), a_w=float(wb.std()),
                          b_z=float(zb.mean()), b_w=float(wb.mean()),
                          nvp_W_s=np.nan, nvp_W_t=np.nan,
                          nvp_c_s=np.nan, nvp_c_t=np.nan)
                if fl == "affine":
                    nll = z.size * (math.log(2 * math.pi) + math.log(zb.std())
                                    + math.log(wb.std()) + 1.0)
                    npar = 4
                else:
                    nll, flow = PROFILE_NLL[fl](z, w, h, dt, dt_step=ds,
                                                ret_flow=True)
                    npar = sum(p.numel() for p in flow.parameters())
                    if len(flow.layers) == 1:
                        lin = flow.layers[0].lin
                        Wt = lin.weight.detach().numpy().ravel()
                        bt = lin.bias.detach().numpy().ravel()
                        nf.update(nvp_W_s=float(Wt[0]), nvp_W_t=float(Wt[1]),
                                  nvp_c_s=float(bt[0]), nvp_c_t=float(bt[1]))
                rows.append(dict(
                    run=k, seed=int(r["seed"]),
                    h_init=float(r["h_init"]), dt_init=float(r["dt_init"]),
                    h_hop=float(r.get("h_hop", np.nan)),
                    dt_hop=float(r.get("dt_hop", np.nan)),
                    h_fit=h, dt_fit=dt, N_steps=dt / ds,
                    NLL=float(nll), NLL_per_star=float(nll) / z.size,
                    dNLL=float(r["dnll"]), hops=int(r.get("n_hops", 0)),
                    evals=int(r.get("n_eval_sub", 0) + r.get("n_eval_full", 0)),
                    wall_s=float(r.get("wall", np.nan)),
                    n_flow_params=int(npar), **nf))
            if cache:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as f:
                    pickle.dump(rows, f)
        d = pd.DataFrame(rows)
        d.insert(0, "figure", label)
        d.insert(1, "flow", fl)
        d["sig_h"] = float(prof["sig_h"]) if prof else np.nan
        d["sig_dt"] = float(prof["sig_dt"]) if prof else np.nan
        d["dh"] = d["h_fit"] - hop["h_true"]
        d["ddt"] = d["dt_fit"] - hop["dt_true"]
        d["pull_h"] = d["dh"] / d["sig_h"]
        d["pull_dt"] = d["ddt"] / d["sig_dt"]
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out.index.name = "fit"
    out = out[HOP_RESULT_COLS]
    out.attrs.update(h_true=H_TRUE, dt_true=DT_TRUE, N=int(z.size),
                     z0_sigma=Z0_SIGMA, w0_sigma=W0_SIGMA)
    return out


HOP_RESULT_COLS = [
    "figure", "flow", "n_flow_params", "run", "seed",
    "h_init", "dt_init", "h_hop", "dt_hop",
    "h_fit", "sig_h", "dt_fit", "sig_dt", "N_steps",
    "dh", "ddt", "pull_h", "pull_dt",
    "NLL", "NLL_per_star", "dNLL", "hops", "evals", "wall_s",
    "a_z", "a_w", "b_z", "b_w", "nvp_W_s", "nvp_W_t", "nvp_c_s", "nvp_c_t"]


HOP_HAM_COLS = ["h_init", "dt_init", "seed", "h_hop", "dt_hop", "h_fit",
                "dt_fit", "N_steps", "dh", "ddt", "dNLL", "hops", "evals"]
HOP_NF_COLS = ["h_init", "dt_init", "a_z", "a_w", "b_z", "b_w"]


def hop_summary(hop):
    """Text summary of the basin-hopping convergence study."""
    runs = hop["runs"]
    hf = np.array([r["h_fit"] for r in runs])
    df = np.array([r["dt_fit"] for r in runs])
    nh = np.array([r["n_hops"] for r in runs])
    ne = np.array([r.get("n_eval_sub", 0) + r.get("n_eval_full", 0)
                   for r in runs])
    dn = np.array([r["dnll"] for r in runs])
    hi = sorted({r["h_init"] for r in runs}); di = sorted({r["dt_init"] for r in runs})
    return "\n".join([
        f"Basin hopping from {len(runs)} starting points "
        f"(h_init {min(hi):.0f}-{max(hi):.0f} pc, dt_init {min(di):.0f}-{max(di):.0f} Myr)",
        f"  truth                : h_dm = {hop['h_true']:.0f} pc, "
        f"Delta t = {hop['dt_true']:.0f} Myr",
        f"  landings             : h_dm = {hf.mean():.2f} +/- {hf.std():.3f} pc "
        f"(range {hf.min():.2f}-{hf.max():.2f})",
        f"                         Delta t = {df.mean():.3f} +/- {df.std():.4f} Myr "
        f"(range {df.min():.3f}-{df.max():.3f})",
        f"  reached the optimum  : {int((dn < 0.5).sum())}/{len(runs)}"
        f"   (max dNLL = {dn.max():+.3f} nats)",
        f"  search effort        : {nh.min()}-{nh.max()} hops, "
        f"{ne.min()}-{ne.max()} likelihood evaluations",
        f"  -> the paths DIFFER ({len(set(map(tuple, zip(nh, ne))))} distinct "
        f"(hops, evals) pairs), so the agreement of the endpoints is a",
        f"     measurement.  Contrast `dealias_scan`, which discards the initial",
        f"     guess and makes every start a bit-identical computation.",
    ])


def hdt_fit_table(res, z_obs=None, w_obs=None):
    """One row per fit: the Hamiltonian-flow parameters and the normalizing-flow
    parameters it converged to, as a pandas DataFrame.

    The bijection's parameters are read from the cached fit when it recorded them
    (`nf`).  Results cached before that was added fall back to the CLOSED-FORM
    optimum at the landing point -- for a diagonal-affine bijection and a
    Gaussian f0 the maximum-likelihood a and b are just the standard deviation
    and mean of the back-integrated cloud, which is the same optimum the descent
    converges to, so the two agree to the descent's own tolerance.  The
    `nf_source` column says which was used."""
    import pandas as pd
    ds = res.get("dt_step", DT_STEP)
    best = min(t["nll"] for t in res["traj"])
    rows = []
    for k, t in enumerate(res["traj"]):
        nf, src = t.get("nf"), "fitted"
        if nf is None:
            if z_obs is None:
                nf, src = {}, "unavailable"
            else:
                zb, wb = backint(z_obs, w_obs, t["h_fit"], t_obs=t["dt_fit"],
                                 dt_step=ds)
                nf, src = dict(a_z=float(zb.std()), a_w=float(wb.std()),
                               b_z=float(zb.mean()), b_w=float(wb.mean())), \
                          "closed form"
        rows.append(dict(
            fit=k, tag=t.get("tag", "tuned"),
            h_init=float(t["h_init"]), dt_init=float(t["dt_init"]),
            h_sweep=float(t.get("h_scan", np.nan)),
            dt_sweep=float(t.get("dt_scan", np.nan)),
            h_fit=float(t["h_fit"]), dt_fit=float(t["dt_fit"]),
            N_steps=float(t["dt_fit"]) / ds,
            dh=float(t["h_fit"]) - res["h_true"],
            ddt=float(t["dt_fit"]) - res["dt_true"],
            NLL=float(t["nll"]), dNLL=float(t["nll"]) - best,
            epochs=t.get("n_epochs"), polish=t.get("polish"),
            nf_source=src, **nf))
    df = pd.DataFrame(rows).set_index("fit")
    df.attrs.update(h_true=res["h_true"], dt_true=res["dt_true"],
                    dt_step=ds, N_steps_true=res["dt_true"] / ds,
                    dt_min=res["dt_min"], N=res["N"],
                    z0_sigma=Z0_SIGMA, w0_sigma=W0_SIGMA)
    return df


HDT_HAM_COLS = ["tag", "h_init", "dt_init", "h_sweep", "dt_sweep",
                "h_fit", "dt_fit", "N_steps", "dh", "ddt", "NLL", "dNLL"]
HDT_NF_COLS = ["tag", "a_z", "a_w", "b_z", "b_w", "nf_source"]


# ----------------------------------------------------------------------------
# Joint (h_dm, flow) fit  ->  profile likelihood AROUND that joint minimum
# ----------------------------------------------------------------------------
# Computed by `experiments/joint_profile_fit.py` (one run per case, cached in
# results/JP_{pg,2w}.pkl, figure-ready copy in results/JP_figure.pkl).  This is
# NOT a scan over a fixed 200-300 grid: h_dm descends together with every flow
# parameter first, and the profile grid is then placed AROUND the resulting
# h_hat, re-optimising all the other parameters at each fixed h.
JP_COL = {"affine 4p": "#1f77b4", "8p": "#2ca02c", "16p": "#17becf",
          "204p": "#ff7f0e", "272p": "#d62728", "304p": "#9467bd"}
JP_TITLE = {"pg": "Regular Gaussian $f_0$",
            "2w": "Two-width $f_0$ (core 15 + wings 60)"}


def _jp_parab(hs, d):
    k = int(np.argmin(d))
    if k == 0 or k == len(d) - 1:
        return float(hs[k]), float("nan")
    a, b, _ = np.polyfit(hs[k - 1:k + 2], d[k - 1:k + 2], 2)
    return (float(hs[k]), float("nan")) if a <= 0 else \
           (float(-b / (2 * a)), float(math.sqrt(0.5 / a)))


def plot_joint_profile(path=None, cases=("pg", "2w")):
    """Draw the joint-fit + profile-likelihood figure from results/JP_figure.pkl.

    Rows: 1 the joint descent of h_dm with the flow (several starts);
          2 the profile likelihood around h_hat, full local range (log y);
          3 the same zoomed on the minimum -- the uncertainty curve, 1 sigma at
            Delta-NLL = 0.5;  4 the f0 the same fit recovers (w-marginal, t=0).
    Pure plotting: everything was fitted once and cached, nothing is re-fitted.
    """
    import pickle
    import matplotlib.pyplot as plt
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "results", "JP_figure.pkl")
    with open(path, "rb") as f:
        F = pickle.load(f)
    sel = [c for c in F["cases"] if c["case"] in cases]
    ht, wg = F["h_true"], F["wg"]
    fig, axes = plt.subplots(4, len(sel), figsize=(7.4 * len(sel), 17.5), squeeze=False)

    for col, res in enumerate(sel):
        case = res["case"]
        ax = axes[0][col]
        for m in res["models"]:
            c = JP_COL.get(m["label"], "k")
            for k, hist in enumerate(m["hist"]):
                ax.plot(hist, color=c, lw=1.4, alpha=.85,
                        label=m["label"] if k == 0 else None)
        ax.axhline(ht, color="k", ls="--", lw=1.2, label=f"true $h$ = {ht:.0f}")
        ax.set_xlabel("joint-fit epoch"); ax.set_ylabel(r"$h_{\rm dm}$ [pc]")
        ax.set_title(f"{JP_TITLE[case]}\n1. joint fit of $(h_{{\\rm dm}},\\varphi)$ "
                     r"— $h$ and the flow descend together", fontsize=11)
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.25)

        ax = axes[1][col]
        for m in res["models"]:
            c = JP_COL.get(m["label"], "k")
            edge = int(np.argmin(m["dnll"])) in (0, len(m["dnll"]) - 1)
            lab = m["label"] + ": " + ("min at grid edge — unresolved" if edge else
                                       f"$\\hat h$={m['h_min']:.0f} $\\pm$ "
                                       f"{m['sigma']:.1f} pc")
            ax.plot(m["hgrid"], np.maximum(m["dnll"], 1e-3), "o-", ms=3.4, lw=1.5,
                    color=c, label=lab)
        if res.get("analytic") is not None:
            a = res["analytic"]
            ax.plot(a["hgrid"], np.maximum(a["dnll"], 1e-3), "k:", lw=1.6,
                    label="analytic linear 4p (closed form)")
        ax.axvline(ht, color="k", ls="--", lw=1.2)
        ax.axhline(0.5, color="grey", ls=":", lw=1.4)
        ax.set_yscale("log"); ax.set_ylim(1e-2, 3e3); ax.set_xlim(180, 320)
        ax.set_xlabel(r"$h_{\rm dm}$ [pc]"); ax.set_ylabel(r"$\Delta$NLL [nats] (log)")
        ax.set_title("2. profile likelihood around $\\hat h$ — full range\n"
                     "at each fixed $h$ ALL flow params are re-fitted; "
                     r"grey $=\Delta$NLL$\,{=}\,0.5$", fontsize=11)
        ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=.25)

        ax = axes[2][col]
        ok = [m for m in res["models"]
              if int(np.argmin(m["dnll"])) not in (0, len(m["dnll"]) - 1)]
        for m in ok:
            c = JP_COL.get(m["label"], "k")
            ax.plot(m["hgrid"], m["dnll"], "o-", ms=4, lw=1.6, color=c,
                    label=f"{m['label']}: {m['h_min']:.1f} $\\pm$ {m['sigma']:.1f} pc")
            if np.isfinite(m["sigma"]):
                ax.errorbar([m["h_min"]], [0.], xerr=[[m["sigma"]], [m["sigma"]]],
                            fmt="*", ms=14, color=c, capsize=4, zorder=5, lw=2)
        if res.get("analytic") is not None:
            ax.plot(res["analytic"]["hgrid"], res["analytic"]["dnll"], "k:", lw=1.6,
                    label="analytic linear 4p")
        if res.get("refine") is not None:
            r = res["refine"]; hm, sg = _jp_parab(r["hgrid"], r["dnll"])
            ax.plot(r["hgrid"], r["dnll"], "s--", ms=4, lw=1.4, color="#8b0000",
                    alpha=.85, label=f"{r['label']}, 3-pass envelope: "
                                     f"{hm:.1f} $\\pm$ {sg:.1f} pc")
        ax.axvline(ht, color="k", ls="--", lw=1.4)
        ax.axhline(0.5, color="grey", ls=":", lw=1.4)
        ax.text(ht + .4, 2.6, f"true {ht:.0f}", fontsize=9)
        ax.set_ylim(0, 3)
        ax.set_xlim(min(m["h_min"] for m in ok) - 22, max(m["h_min"] for m in ok) + 22)
        ax.set_xlabel(r"$h_{\rm dm}$ [pc]"); ax.set_ylabel(r"$\Delta$NLL [nats]")
        ax.set_title("3. the uncertainty curve: zoom on the minimum\n"
                     r"$1\sigma$ = half-width at $\Delta$NLL$\,{=}\,0.5$", fontsize=11)
        ax.legend(fontsize=8.5); ax.grid(alpha=.25)

        ax = axes[3][col]
        ax.plot(wg, res["true_w"], "k--", lw=2.2, label="true $f_0$")
        for m in res["models"]:
            ax.plot(wg, m["fit_w"], lw=1.5, color=JP_COL.get(m["label"], "k"),
                    label=f"{m['label']} fit")
        ax.set_yscale("log"); ax.set_ylim(1e-6, 5e-2)
        ax.set_xlabel("$w$ [km/s] at $t=0$"); ax.set_ylabel("$f_0(w)$")
        ax.set_title("4. the $f_0$ recovered by the same fit (marginal in $w$, $t=0$)",
                     fontsize=11)
        ax.legend(fontsize=8.5); ax.grid(alpha=.25)

    fig.suptitle(r"Joint fit of $h_{\rm dm}$ with the flow, then profile likelihood "
                 r"around the joint minimum   ($N=20\,000$ stars)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .975])
    return fig


def joint_profile_table(path=None, cases=("pg", "2w")):
    """Print the h_dm summary behind `plot_joint_profile`."""
    import pickle
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "results", "JP_figure.pkl")
    with open(path, "rb") as f:
        F = pickle.load(f)
    print("%-20s %5s  %8s  %8s  %-14s" %
          ("case / model", "#par", "h_joint", "h_prof", "1sigma (D=0.5)"))
    for res in F["cases"]:
        if res["case"] not in cases:
            continue
        for m in res["models"]:
            edge = int(np.argmin(m["dnll"])) in (0, len(m["dnll"]) - 1)
            print("%-20s %5d  %8.1f  %8.1f  %-14s" %
                  (res["case"] + " / " + m["label"], m["npar"], m["h_hat"],
                   m["h_min"], "unresolved" if edge else f"+/-{m['sigma']:.1f}"))
            if len(m["h_joints"]) > 1:
                print("      joint fits from h_init=%s -> %s" %
                      (np.round(m["h_inits"], 0), np.round(m["h_joints"], 1)))
