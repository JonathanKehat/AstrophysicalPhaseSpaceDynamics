"""Joint fit of (h_dm, Delta t, f0) with a MIRROR-SYMMETRIC f0 prior.

THE PROBLEM.  Phase-space density is conserved (Liouville), so for ANY (h, dt)
the density f0' = f_obs o T_{h,dt} reproduces the snapshot exactly.  A flexible
flow finds it, and every measure of fit to the data is then blind to which
(h, dt) is right (measured: 114-param flow prefers the wrong teeth at 437 / 348
Myr; EMD, held-out likelihood and weight decay do not help).  The extra
information has to be a prior on f0.

THE PRIOR.  f0 is symmetric under z -> -z:  f0(z, w) = f0(-z, w).
Physical content: at t = 0 the disk was NORTH-SOUTH SYMMETRIC.  That holds for
any IMPULSIVE perturbation of a symmetric disk that acts on velocities: an
instantaneous kick moves no star, so the positions are still the symmetric
pre-kick ones, whatever the kick does to w (uniform kick, heating, a
w-dependent kick ...).  It does NOT assume f0 is separable, Gaussian, or
symmetric in w.

WHY IT IDENTIFIES dt (no tuning parameter).  Restrict the model to symmetric
densities, q = (r + r o M)/2 with M: z -> -z.  For data with density p,
E_p[log q] = E_{sym p}[log q], so the best symmetric model is sym p = (p + p o M)/2
and the minimum NLL per star is

        H(p)  +  KL( p || sym p ).

H(p_{h,dt}) is the SAME at every (h, dt) (the back-integration is
volume-preserving), so the objective is a constant plus the KL distance of the
back-integrated cloud from its own mirror image -- zero only where the cloud IS
mirror symmetric.  At a wrong tooth the core is rotated by ~a multiple of a
quarter period (and so is nearly symmetric again) but the high-action stars --
the 60 km/s wings, with a lower frequency -- are rotated by a different angle:
the cloud is a partly wound spiral, which is NOT mirror symmetric, and no
symmetric density can absorb that.  Unlike a penalty (Fisher, l2) there is no
lambda: the restriction is either true of f0 or it is not.

WHEN IT FAILS ON REAL DATA.
  * a perturbation that DISPLACED stars at t = 0 (non-impulsive forcing, a
    vertical breathing kick dw ~ -k z acting over a finite time, a disk that
    was already bent), or several perturbations at different times;
  * a north-south asymmetric selection function / extinction, or an unmodelled
    solar offset z_sun (it must then be a fitted parameter: symmetry is about
    the MIDPLANE);
  * a harmonic potential: quarter-period aliases become exact (the
    identification uses the ANHARMONIC, action-dependent frequency);
  * an f0 that is already phase-mixed (a function of energy only) carries no
    information at all (correctly: dt is then genuinely undefined).

Model = ZW(3,4) couplings (114 params) + learnable affine (4), symmetrised:
flow name "zw114m" in experiments.common (JointFlow(mirror=True)).

Every result is cached in results/MIR_*.pkl (never overwritten).
"""
import math
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import torch

from experiments import common as C

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")
FLOW = "zw114m"
# sample size of the full fit; MIR_N=600 for smoke tests (caches carry N)
N_STARS = int(os.environ.get("MIR_N", 20000))
DS = C.DT_STEP
Z0 = C.Z0_SIGMA

# ================================================================================
#   datasets -- f0 = sum_k a_k N(z; 0, sz_k) N(w; mw_k, sw_k)
# ================================================================================
# FIXED BEFORE ANY RESULT WAS SEEN (2026-10-01).  "2w" is the target dataset
# (= experiments.penalized_f0.load_data); the rest are the blind T5 set.
DATASETS = {
    "2w": dict(comps=[(0.6, Z0, 0.0, 15.0), (0.4, Z0, 0.0, 60.0)],
               h=250.0, dt=400.0, seed=3,
               label="two-width $f_0$ (seed 3, h=250, $\\Delta t$=400)"),
    "2w_seed11": dict(comps=[(0.6, Z0, 0.0, 15.0), (0.4, Z0, 0.0, 60.0)],
                      h=250.0, dt=400.0, seed=11,
                      label="two-width, NEW DATA SEED 11"),
    "2w_h300": dict(comps=[(0.6, Z0, 0.0, 15.0), (0.4, Z0, 0.0, 60.0)],
                    h=300.0, dt=350.0, seed=5,
                    label="two-width, DIFFERENT TRUTH h=300, $\\Delta t$=350"),
    "3w": dict(comps=[(0.5, Z0, 0.0, 15.0), (0.35, Z0, 0.0, 40.0),
                      (0.15, Z0, 0.0, 90.0)],
               h=250.0, dt=400.0, seed=7,
               label="three-width $f_0$ (15/40/90 km/s)"),
    # NON-SEPARABLE: a cold thin component + a hot thick one (z AND w widths
    # differ per component) -- the realistic thin+thick-disk shape.
    "thinthick": dict(comps=[(0.6, 180.0, 0.0, 15.0), (0.4, 450.0, 0.0, 50.0)],
                      h=250.0, dt=400.0, seed=9,
                      label="NON-SEPARABLE thin+thick $f_0$"),
    # regression (T1): the Gaussian notebook's data
    "gauss": dict(comps=[(1.0, Z0, 0.0, C.W0_SIGMA)], h=250.0, dt=400.0,
                  seed=3, label="Gaussian $f_0$ (regression)"),
}


def load(tag, N=None):
    N = N_STARS if N is None else N
    """(z, w) observed at t = dt for dataset `tag`."""
    s = DATASETS[tag]
    if tag == "2w":
        from experiments import penalized_f0 as P
        return P.load_data(N)
    if tag == "gauss":
        return C.make_gaussian_data(N=N, seed_sample=3)
    rng = np.random.default_rng(s["seed"])
    a = np.array([c[0] for c in s["comps"]])
    k = rng.choice(len(a), size=N, p=a / a.sum())
    sz = np.array([c[1] for c in s["comps"]])[k]
    mw = np.array([c[2] for c in s["comps"]])[k]
    sw = np.array([c[3] for c in s["comps"]])[k]
    z0 = rng.normal(0.0, sz)
    w0 = rng.normal(mw, sw)
    Zs, Ws, _ = C.sim.evolve_record(z0, w0, t_obs_myr=s["dt"],
                                    n_steps=int(round(30 * s["dt"])), n_snaps=10,
                                    seed=0, use_satellite=False, h_pc_dm=s["h"])
    return Zs[-1].astype(np.float64), Ws[-1].astype(np.float64)


def true_logf0_t(tag, z0, w0):
    """log f0_true (torch, differentiable), per pc km/s."""
    terms = []
    for a, sz, mw, sw in DATASETS[tag]["comps"]:
        terms.append(math.log(a) - math.log(2 * math.pi * sz * sw)
                     - 0.5 * (z0 / sz) ** 2 - 0.5 * ((w0 - mw) / sw) ** 2)
    return torch.logsumexp(torch.stack(terms), 0)


def true_logf0(tag, z0, w0):
    return true_logf0_t(tag, torch.as_tensor(z0), torch.as_tensor(w0)).numpy()


def true_marginals(tag, zg, wg):
    pz = sum(a * C.gauss_pdf(zg, 0.0, sz) for a, sz, mw, sw in DATASETS[tag]["comps"])
    pw = sum(a * C.gauss_pdf(wg, mw, sw) for a, sz, mw, sw in DATASETS[tag]["comps"])
    return pz, pw


def backint_t(zt, wt, h, dt):
    from fitter import torch_back_integrate
    h = h if torch.is_tensor(h) else torch.tensor(float(h), dtype=torch.float64)
    return torch_back_integrate(zt, wt, dt, h, C.sim, dt_step_myr=DS)


def subsample(z, w, n):
    """Same 4000-star subsample basin hopping uses (rng 0)."""
    if n >= z.size:
        return z, w
    idx = np.random.default_rng(0).choice(z.size, size=n, replace=False)
    return z[idx], w[idx]


def fisher_excess(logp_fn, zb, wb):
    """E|grad_y log f|^2 - 2 in coordinates standardised by the CLOUD's sd
    (= C._joint_fisher's definition); 0 for a Gaussian."""
    zr = zb.detach().clone().requires_grad_(True)
    wr = wb.detach().clone().requires_grad_(True)
    gz, gw = torch.autograd.grad(logp_fn(zr, wr).sum(), (zr, wr))
    return float(((gz * zb.std()) ** 2 + (gw * wb.std()) ** 2).mean() - 2.0)


def _log(msg):
    with open(os.path.join(RESULTS, "MIR_progress.log"), "a") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


# ================================================================================
#   converged fits
# ================================================================================
def _lbfgs_converge(params, total, chunk=100, window=500, ftol=0.05,
                    max_iter=30000, verbose=False):
    """L-BFGS (strong Wolfe), fresh history every `chunk` iterations (a failed
    line search must not end the fit), until the objective improves by less
    than `ftol` nats over the last `window` iterations.

    WHY NOT A GRADIENT TOLERANCE: the coupling network creeps logarithmically
    (measured on 4000 stars: -2.5 nats over iterations 1000-3300, still
    -0.15 nats / 100 its) while max|grad| fluctuates between 3 and 50 -- a
    gradient test is never met.  So the stop is on the CREEP RATE, and the final
    gradient and creep are returned so every tooth's residual is visible.
    `max_iter` is a runaway guard; `capped` says whether it was hit."""
    n_it, hist = 0, []
    while True:
        lb = torch.optim.LBFGS(params, lr=1.0, max_iter=chunk, history_size=50,
                               tolerance_grad=0.0, tolerance_change=0.0,
                               line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            J = total()
            J.backward()
            return J
        try:
            lb.step(closure)
        except Exception:
            pass
        n_it += chunk
        Jf = float(total().detach())
        hist.append(Jf)
        if verbose:
            print(f"    it {n_it:5d}  J={Jf:.4f}", flush=True)
        k = window // chunk
        if len(hist) > k and hist[-k - 1] - Jf < ftol:
            break
        if n_it >= max_iter:
            break
    J = total()
    g = torch.autograd.grad(J, params, allow_unused=True)
    gmax = max(float(x.abs().max()) for x in g if x is not None)
    k = min(window // chunk, len(hist) - 1)
    creep = (hist[-k - 1] - hist[-1]) / (k * chunk) * 1000 if k > 0 else float("nan")
    return dict(J=float(J.detach()), gmax=gmax, n_iter=n_it, capped=n_it >= max_iter,
                creep_per_1000=creep, hist=hist)


def tooth_fit(z, w, h, dt0, flow=FLOW, seed=0, n_adam=1500, lr=1e-2,
              free_h=False, state=None, verbose=False):
    """Converged JOINT fit inside one comb tooth: Delta t and every flow
    parameter free (and h too if `free_h`), from (h, dt0).
      1. flow only, on the cloud at (h, dt0): Adam (cheap, no integration);
      2. joint L-BFGS over (N_steps[, h/10pc], all flow params) to convergence.
    Returns dict(h, dt, J [total nats], grad, iterations, flow state)."""
    t0 = time.time()
    zt, wt = torch.tensor(z), torch.tensor(w)
    jf = C.make_joint_flow(flow, seed)
    with torch.no_grad():
        zb, wb = backint_t(zt, wt, h, dt0)
    jf.set_ref(zb, wb)
    if state is not None:
        jf.load_state_dict({k: v for k, v in state.items()
                            if k not in ("mu_ref", "sig_ref", "m", "s")}, strict=False)
    if state is not None:          # warm start: no Adam (it would kick the
        n_adam = 0                 # flow out of the minimum it came from)
    opt = torch.optim.Adam(jf.parameters(), lr=lr)
    for _ in range(int(n_adam)):
        l = -jf.log_prob(zb, wb).mean()
        opt.zero_grad(); l.backward(); opt.step()
    # converge the flow on the FIXED cloud first: an iteration here costs ~1% of
    # a joint one (no integration), and the coupling network's slow creep
    # (~1 nat / 100 its) would otherwise be paid at the joint price.
    conv0 = _lbfgs_converge(list(jf.parameters()),
                            lambda: -jf.log_prob(zb, wb).sum(), verbose=verbose)
    J_flow_only = float(-jf.log_prob(zb, wb).sum().detach())

    nst = torch.nn.Parameter(torch.tensor(float(dt0) / DS, dtype=torch.float64))
    u = torch.nn.Parameter(torch.tensor(float(h) / C.H_SCALE, dtype=torch.float64))
    params = [nst] + ([u] if free_h else []) + list(jf.parameters())

    def total():
        hh = u * C.H_SCALE if free_h else torch.tensor(float(h), dtype=torch.float64)
        zb_, wb_ = backint_t(zt, wt, hh, nst * DS)
        return -jf.log_prob(zb_, wb_).sum()

    # joint stage: an iteration costs ~50x a flow-only one, and the flow's own
    # creep (~0.1 nats / 1000 its) was already taken out above -- stop once the
    # joint objective improves by < 0.25 nats per 500 its (creep reported).
    conv = _lbfgs_converge(params, total, ftol=0.25, verbose=verbose)
    _log(f"tooth_fit h0={h:.1f} dt0={dt0:.1f} {flow} seed={seed} N={z.size}: "
         f"dt={nst.item()*DS:.3f} h={u.item()*C.H_SCALE:.2f} J={conv['J']:.2f} "
         f"it={conv['n_iter']} creep={conv['creep_per_1000']:.3f} "
         f"{(time.time()-t0)/60:.1f} min")
    J = total()
    g = torch.autograd.grad(J, params)
    out = dict(h0=float(h), dt0=float(dt0), seed=seed, flow=flow,
               h=float(u.item() * C.H_SCALE) if free_h else float(h),
               dt=float(nst.item() * DS), J=float(J.detach()),
               J_flow_only=J_flow_only,
               grad_dt=float(g[0]) / DS,           # nats / Myr
               grad_h=float(g[1]) / C.H_SCALE if free_h else float("nan"),
               grad_flow_max=max(float(x.abs().max()) for x in g[(2 if free_h else 1):]),
               state={k: v.detach().clone() for k, v in jf.state_dict().items()},
               flow_only_iter=conv0["n_iter"],
               wall=time.time() - t0, N=int(z.size), **{f"conv_{k}": v
                                                         for k, v in conv.items()
                                                         if k != "hist"})
    return out


# ================================================================================
#   pool plumbing + cache
# ================================================================================
_CTX = {}


def _init(z, w):
    torch.set_num_threads(1)
    _CTX.update(z=z, w=w)


def _pool_map(fn, jobs, z, w, workers=10):
    with Pool(min(workers, len(jobs)), initializer=_init, initargs=(z, w)) as p:
        return p.map(fn, jobs, chunksize=1)


def _cache(name, build, force=False):
    path = os.path.join(RESULTS, f"MIR_{name}.pkl")
    if not force and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    out = build()
    out["wall_total"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


# ================================================================================
#   T0 -- the tooth test
# ================================================================================
T0_DTS = (400.0, 437.0, 366.0, 348.0, 446.0)
T0_HS = (200.0, 250.0, 300.0)


def _tooth_worker(job):
    h, dt, flow, seed = job
    return tooth_fit(_CTX["z"], _CTX["w"], h, dt, flow=flow, seed=seed)


def tooth_test(tag="2w", n=4000, hs=T0_HS, dts=None, flows=(FLOW, "zw114"),
               seeds=(0, 1), workers=10, force=False):
    """Converged fits (h fixed, Delta t + flow free) at every (h, tooth, flow,
    flow-init seed).  The teeth are given as offsets from the dataset's true
    Delta t (the T0 list is for dt_true = 400; for another truth the same
    offsets are used, so the test is identical in form)."""
    s = DATASETS[tag]
    dts = tuple(s["dt"] + (d - 400.0) for d in T0_DTS) if dts is None else dts

    def build():
        z, w = subsample(*load(tag), n)
        jobs = [(float(h), float(d), f, sd) for f in flows for h in hs
                for d in dts for sd in seeds]
        rows = _pool_map(_tooth_worker, jobs, z, w, workers)
        for r in rows:
            r.pop("state", None) if r["flow"] != FLOW else None
        return dict(tag=tag, n=n, rows=rows, hs=hs, dts=dts, flows=flows,
                    dt_true=s["dt"], h_true=s["h"])
    fl = "_".join(flows)
    return _cache(f"T0_{tag}_n{n}_{fl}", build, force)


def tooth_table(t0):
    """Best (over flow-init seeds) J per (flow, h, tooth), relative to the truth
    tooth at the same h and flow, + the convergence diagnostics."""
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "state"}
                       for r in t0["rows"]])
    best = df.loc[df.groupby(["flow", "h0", "dt0"])["J"].idxmin()].copy()
    spread = df.groupby(["flow", "h0", "dt0"])["J"].agg(lambda x: x.max() - x.min())
    best = best.set_index(["flow", "h0", "dt0"])
    best["seed_spread"] = spread
    ref = best.xs(t0["dt_true"], level="dt0")["J"]
    best["dJ_vs_truth"] = [best.loc[i, "J"] - ref.loc[(i[0], i[1])] for i in best.index]
    return best[["dt", "J", "dJ_vs_truth", "seed_spread", "grad_dt",
                 "grad_flow_max", "conv_n_iter", "conv_creep_per_1000", "conv_capped", "wall"]]


def tooth_verdict(t0, flow=FLOW):
    """Per h: margin = (best wrong tooth J) - (truth J); pass if > 1 nat... we
    require >> 1, report the number."""
    tab = tooth_table(t0).xs(flow, level="flow")
    out = {}
    for h in t0["hs"]:
        th = tab.xs(h, level="h0")
        wrong = th.drop(t0["dt_true"])["dJ_vs_truth"]
        out[h] = dict(margin=float(wrong.min()), worst_tooth=float(wrong.idxmin()),
                      truth_first=bool(wrong.min() > 0))
    return out


def plot_tooth_test(t0s, title=None):
    """One panel per (N): J - J(truth tooth) vs tooth, mirror vs unrestricted."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(t0s), figsize=(6.4 * len(t0s), 4.6), squeeze=False)
    for ax, t0 in zip(axs[0], t0s):
        tab = tooth_table(t0)
        for flow, ls, mk in ((FLOW, "-", "o"), ("zw114", ":", "x")):
            if flow not in tab.index.get_level_values(0):
                continue
            for h, col in zip(t0["hs"], ("C0", "C3", "C2")):
                th = tab.xs((flow, h), level=("flow", "h0")).sort_index()
                lab = (f"{'mirror f$_0$' if flow == FLOW else 'unrestricted'}"
                       f", h={h:.0f}")
                ax.plot(th.index, th["dJ_vs_truth"], ls=ls, marker=mk, color=col,
                        lw=2 if flow == FLOW else 1.2, label=lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(t0["dt_true"], color="0.5", ls="--", lw=0.8)
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("comb tooth $\\Delta t$ [Myr]  (converged fit inside it)")
        ax.set_ylabel("$J - J_{\\rm truth\\ tooth}$  [nats]")
        ax.set_title(f"T0 tooth test, {t0['n']} stars ({t0['tag']})")
        ax.legend(fontsize=7.5, ncol=2)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# ================================================================================
#   T2 -- joint basin hopping
# ================================================================================
HOP_STARTS = ((150, 250), (220, 300), (250, 400), (290, 520), (400, 220),
              (500, 380), (120, 430), (600, 60))


def hop_study(tag="2w", starts=HOP_STARTS, seeds=(0,), force=False, **kw):
    """Joint basin hopping (C.basin_hop_h_dt local='joint') with the mirror
    flow from every initial guess; cached results/PGTMIR{tag}_hop_*."""
    z, w = load(tag)
    opts = dict(starts=starts, seeds=seeds, flow=FLOW, local="joint",
                data_tag=f"MIR{tag}", dh=20.0, ddt=2.0, patience=15, n_iter=40,
                force=force)
    opts.update(kw)
    hop = C.run_hop_study(z, w, **opts)
    s = DATASETS[tag]
    hop.update(h_true=s["h"], dt_true=s["dt"], tag=tag)
    return hop


def _refine_worker(job):
    h, dt, state = job
    return tooth_fit(_CTX["z"], _CTX["w"], h, dt, free_h=True, state=state)


def refine(tag, hop, force=False):
    """Converge EVERY basin-hopping landing on the full sample: one joint
    L-BFGS over (h, N_steps, all flow params) from the landing and its flow,
    stopped by the creep rule (the hop's own full-sample stage is capped at
    400 iterations, which the coupling network's creep does not respect).
    Runs are refined independently, so their agreement is a measurement."""
    def build():
        z, w = load(tag)
        jobs = [(r["h_fit"], r["dt_fit"], r["flow_state"]) for r in hop["runs"]]
        fits = _pool_map(_refine_worker, jobs, z, w, workers=len(jobs))
        for f, r in zip(fits, hop["runs"]):
            f.update(h_init=r["h_init"], dt_init=r["dt_init"], h_hop=r["h_fit"],
                     dt_hop=r["dt_fit"], J_hop=r["J_full"], n_hops=r["n_hops"],
                     hop_wall=r["wall"])
        return dict(tag=tag, fits=fits)
    return _cache(f"refine_{tag}_N{N_STARS}", build, force)


def best_fit(ref):
    return min(ref["fits"], key=lambda r: r["J"])


def hop_errors(tag, ref, force=False):
    """Full Hessian (h, N_steps, 118 flow params) at the best refined fit: is
    it a minimum, and the (h, dt) errors marginalised over the flow."""
    b = best_fit(ref)

    def build():
        z, w = load(tag)
        return C.joint_hessian(z, w, b["h"], b["dt"], FLOW, b["state"])
    return _cache(f"hess_{tag}_N{N_STARS}", build, force)


def fit_table(ref, err=None):
    import pandas as pd
    Jb = best_fit(ref)["J"]
    s = DATASETS[ref["tag"]]
    rows = []
    for r in ref["fits"]:
        d = dict(h_init=r["h_init"], dt_init=r["dt_init"], h_hop=r["h_hop"],
                 dt_hop=r["dt_hop"], h_fit=r["h"], dt_fit=r["dt"], dJ=r["J"] - Jb,
                 grad_h=r["grad_h"], grad_dt=r["grad_dt"],
                 grad_flow_max=r["grad_flow_max"],
                 creep_per_1000=r["conv_creep_per_1000"], n_hops=r["n_hops"],
                 wall_min=(r["hop_wall"] + r["wall"]) / 60.0)
        if err is not None:
            d["pull_h"] = (r["h"] - s["h"]) / err["sig_h"]
            d["pull_dt"] = (r["dt"] - s["dt"]) / err["sig_dt"]
        rows.append(d)
    return pd.DataFrame(rows).sort_values("dJ")


def verdict(ref, err):
    """T2 + T3 numbers."""
    s = DATASETS[ref["tag"]]
    b = best_fit(ref)
    hf = np.array([r["h"] for r in ref["fits"]])
    df = np.array([r["dt"] for r in ref["fits"]])
    tooth = np.abs(df - s["dt"]) < 9.0               # teeth are ~18 Myr apart
    agree = (np.abs(hf - b["h"]) < err["sig_h"]) & (np.abs(df - b["dt"]) < err["sig_dt"])
    return dict(n=len(hf), n_tooth=int(tooth.sum()), n_agree_1sig=int(agree.sum()),
                spread_h=float(hf.std()), spread_dt=float(df.std()),
                h=b["h"], dt=b["dt"], sig_h=err["sig_h"], sig_dt=err["sig_dt"],
                corr=err["corr"], is_min=err["is_min"],
                pull_h=(b["h"] - s["h"]) / err["sig_h"],
                pull_dt=(b["dt"] - s["dt"]) / err["sig_dt"],
                T2=bool(tooth.all() and agree.all() and err["is_min"]),
                T3=bool(abs(b["h"] - s["h"]) < 2 * err["sig_h"] and
                        abs(b["dt"] - s["dt"]) < 2 * err["sig_dt"]))


# ================================================================================
#   T4 -- f0 recovery
# ================================================================================
def _fit_flow_fixed(zb, wb, seed=0, n_adam=1500, state=None):
    """Mirror flow fitted to a FIXED cloud (flow params only), converged."""
    jf = C.make_joint_flow(FLOW, seed)
    jf.set_ref(zb, wb)
    if state is not None:
        jf.load_state_dict({k: v for k, v in state.items()
                            if k not in ("mu_ref", "sig_ref", "m", "s")}, strict=False)
    opt = torch.optim.Adam(jf.parameters(), lr=1e-2)
    for _ in range(int(n_adam)):
        l = -jf.log_prob(zb, wb).mean()
        opt.zero_grad(); l.backward(); opt.step()
    params = list(jf.parameters())
    conv = _lbfgs_converge(params, lambda: -jf.log_prob(zb, wb).sum())
    return jf, conv


def f0_report(tag, ref, k=5, force=False):
    """At the best landing (h_hat, dt_hat):
      * held-out NLL: K-fold, the mirror flow re-fitted on (k-1)/k of the
        back-integrated stars and scored on the rest, minus the TRUE MODEL's
        NLL of the same stars (true f0 at the TRUE (h, dt)), per star;
      * the same in-sample for the jointly fitted flow;
      * excess Fisher information of the fitted f0 vs the true f0;
      * fitted vs true w-marginal in the wings (|w| = 60, 100, 150 km/s)."""
    b = best_fit(ref)
    s = DATASETS[tag]

    def build():
        z, w = load(tag)
        zt, wt = torch.tensor(z), torch.tensor(w)
        with torch.no_grad():
            zb, wb = backint_t(zt, wt, b["h"], b["dt"])
            ztr, wtr = backint_t(zt, wt, s["h"], s["dt"])
            lt = true_logf0_t(tag, ztr, wtr)                     # true model
        jf = C.make_joint_flow(FLOW)
        jf.load_state_dict(b["state"])
        with torch.no_grad():
            lj = jf.log_prob(zb, wb)
        in_sample = float((lt - lj).mean())      # >0: fitted beats truth in-sample
        perm = np.random.default_rng(0).permutation(z.size)
        held = np.empty(z.size)
        for idx in np.array_split(perm, k):
            tr = np.setdiff1d(np.arange(z.size), idx)
            f, _ = _fit_flow_fixed(zb[tr], wb[tr], state=b["state"])
            with torch.no_grad():
                held[idx] = f.log_prob(zb[idx], wb[idx]).numpy()
        excess = float((lt.numpy() - held).mean())   # nats/star, >0 = worse than truth
        R_fit = fisher_excess(jf.log_prob, zb, wb)
        R_true = fisher_excess(lambda a, c: true_logf0_t(tag, a, c), ztr, wtr)
        # wings: the fitted f0's w-marginal vs the truth
        zg = np.linspace(-6 * 600, 6 * 600, 1201)
        wq = np.array([0.0, 30.0, 60.0, 100.0, 150.0])
        ZG, WG = np.meshgrid(zg, wq)
        with torch.no_grad():
            pf = np.exp(jf.log_prob(torch.tensor(ZG.ravel()),
                                    torch.tensor(WG.ravel())).numpy()).reshape(ZG.shape)
        pw_fit = pf.sum(1) * (zg[1] - zg[0])
        pw_true = true_marginals(tag, zg, wq)[1]
        return dict(tag=tag, h=b["h"], dt=b["dt"], heldout_excess=excess,
                    insample_gain=in_sample, R_fit=R_fit, R_true=R_true,
                    w_q=wq, pw_fit=pw_fit, pw_true=pw_true, k=k)
    return _cache(f"f0_{tag}_N{N_STARS}", build, force)


# ================================================================================
#   figures
# ================================================================================
def run_colors(runs):
    import matplotlib.pyplot as plt
    order = sorted(range(len(runs)), key=lambda k: (runs[k]["h_init"],
                                                    runs[k]["dt_init"],
                                                    runs[k]["seed"]))
    cols = plt.cm.turbo(np.linspace(0.04, 0.96, len(order)))
    return {k: c for k, c in zip(order, cols)}


def _marg(logp_fn, zg, wg):
    ZG, WG = np.meshgrid(zg, wg)
    with torch.no_grad():
        p = np.exp(logp_fn(torch.tensor(ZG.ravel()), torch.tensor(WG.ravel())).numpy()
                   ).reshape(ZG.shape)
    return p.sum(0) * (wg[1] - wg[0]), p.sum(1) * (zg[1] - zg[0])


def plot_run_fits(tag, ref, title=None, ng=160):
    """T6 -- every run's fit, one colour per run.
    TOP: fit to the DATA at t_obs (observed stars, the true model, each run's
    f0 carried forward at its own (h, dt)).  BOTTOM: the recovered f0 at t=0 vs
    the true f0 (generating components dotted)."""
    import matplotlib.pyplot as plt
    s = DATASETS[tag]
    z, w = load(tag)
    runs = ref["fits"]
    cols = run_colors(runs)
    Jb = best_fit(ref)["J"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 9.5),
                            gridspec_kw=dict(width_ratios=[1, 1.35]))
    zq = np.percentile(np.abs(z), 99.7) * 1.15
    zg = np.linspace(-zq, zq, ng)
    wg = np.linspace(-220.0, 220.0, ng)

    def obs_lp(f0_lp, h, dt):
        def lp(zo, wo):
            zb, wb = backint_t(zo, wo, h, dt)
            return f0_lp(zb, wb)
        return lp

    axz, axw = axs[0]
    axz.hist(z, bins=90, range=(-zq, zq), density=True, color="0.82",
             label="observed stars")
    axw.hist(w, bins=120, range=(wg[0], wg[-1]), density=True, color="0.82",
             label="observed stars")
    mz, mwt = _marg(obs_lp(lambda a, c: true_logf0_t(tag, a, c), s["h"], s["dt"]),
                    zg, wg)
    axz.plot(zg, mz, "k-", lw=2.4, label="true model", zorder=5)
    axw.plot(wg, mwt, "k-", lw=2.4, label="true model", zorder=5)
    flows = {}
    for k, r in enumerate(runs):
        fl = C.make_joint_flow(FLOW); fl.load_state_dict(r["state"])
        flows[k] = fl
        mz, mw = _marg(obs_lp(fl.log_prob, r["h"], r["dt"]), zg, wg)
        lab = (f"start ({r['h_init']:.0f}, {r['dt_init']:.0f}) $\\to$ "
               f"({r['h']:.1f} pc, {r['dt']:.2f} Myr), "
               f"$\\Delta J$={r['J'] - Jb:.2f}")
        axz.plot(zg, mz, color=cols[k], lw=1.3)
        axw.plot(wg, mw, color=cols[k], lw=1.3, label=lab)
    axz.set_xlabel("z [pc]"); axw.set_xlabel("w [km/s]"); axz.set_ylabel("density")
    axz.set_title(f"fit to the DATA (t = {s['dt']:.0f} Myr): z marginal")
    axw.set_title(f"fit to the DATA (t = {s['dt']:.0f} Myr): w marginal (log)")
    axw.set_yscale("log"); axw.set_ylim(1e-6, 3.0 * float(mwt.max()))
    axw.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))

    zt_, wt_ = backint_t(torch.tensor(z), torch.tensor(w), s["h"], s["dt"])
    szmax = max(c[1] for c in s["comps"])
    z0g = np.linspace(-4 * szmax, 4 * szmax, ng)
    w0g = np.linspace(-220.0, 220.0, ng)
    bz, bw = axs[1]
    bz.hist(zt_.numpy(), bins=90, range=(z0g[0], z0g[-1]), density=True,
            color="0.82", label="stars back-integrated at the TRUE (h, $\\Delta t$)")
    bw.hist(wt_.numpy(), bins=120, range=(w0g[0], w0g[-1]), density=True,
            color="0.82", label="stars back-integrated at the TRUE (h, $\\Delta t$)")
    pz, pw = true_marginals(tag, z0g, w0g)
    bz.plot(z0g, pz, "k-", lw=2.4, label="true $f_0$")
    bw.plot(w0g, pw, "k-", lw=2.4, label="true $f_0$")
    for a, sz, mw_, sw in s["comps"]:
        if len(s["comps"]) > 1:
            bw.plot(w0g, a * C.gauss_pdf(w0g, mw_, sw), "k:", lw=1.0,
                    label=f"component {a:.2g} N({mw_:.0f},{sw:.0f})")
            bz.plot(z0g, a * C.gauss_pdf(z0g, 0.0, sz), "k:", lw=1.0)
    for k, r in enumerate(runs):
        mz, mw = _marg(flows[k].log_prob, z0g, w0g)
        bz.plot(z0g, mz, color=cols[k], lw=1.3)
        bw.plot(w0g, mw, color=cols[k], lw=1.3)
    bz.set_xlabel("z [pc]"); bw.set_xlabel("w [km/s]"); bz.set_ylabel("density")
    bz.set_title("recovered $f_0$ (t = 0): z marginal")
    bw.set_title("recovered $f_0$ (t = 0): w marginal (log)")
    bw.set_yscale("log"); bw.set_ylim(1e-6, None)
    bz.legend(fontsize=7.5)
    bw.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(title or s["label"], fontsize=13)
    fig.tight_layout()
    return fig


def plot_f0_2d(tag, ref, title=None):
    """T4 -- fitted vs true f0 in 2-D at t = 0, and the residual."""
    import matplotlib.pyplot as plt
    s = DATASETS[tag]
    b = best_fit(ref)
    jf = C.make_joint_flow(FLOW); jf.load_state_dict(b["state"])
    szmax = max(c[1] for c in s["comps"])
    ZG, WG = np.meshgrid(np.linspace(-3 * szmax, 3 * szmax, 180),
                         np.linspace(-160, 160, 180))
    with torch.no_grad():
        pf = np.exp(jf.log_prob(torch.tensor(ZG.ravel()),
                                torch.tensor(WG.ravel())).numpy()).reshape(ZG.shape)
    pt = np.exp(true_logf0(tag, ZG, WG))
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    vmax = pt.max()
    for ax, p, t in ((axs[0], pt, "true $f_0$"), (axs[1], pf, "fitted $f_0$")):
        im = ax.pcolormesh(ZG, WG, p, cmap="viridis", vmin=0, vmax=vmax, shading="auto")
        ax.set_title(t); fig.colorbar(im, ax=ax, fraction=0.046)
    r = pf - pt
    im = axs[2].pcolormesh(ZG, WG, r / vmax, cmap="RdBu_r", vmin=-0.1, vmax=0.1,
                           shading="auto")
    axs[2].set_title("(fitted $-$ true) / max(true)")
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    for ax in axs:
        ax.set_xlabel("z [pc]")
    axs[0].set_ylabel("w [km/s]")
    fig.suptitle(title or f"{s['label']}: $f_0$ at the best fit "
                 f"(h={b['h']:.1f} pc, $\\Delta t$={b['dt']:.2f} Myr)")
    fig.tight_layout()
    return fig
