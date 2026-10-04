"""Penalised-f0 fit of (h_dm, Delta t, f0) with the penalty strength chosen by
held-out likelihood.  Two-width (core+wings) data.

WHY.  By Liouville, for ANY (h, Delta t) there is an f0' = f_obs o T_{h,dt} that
reproduces the observed snapshot exactly, so no measure of fit-to-the-data
(likelihood, EMD, ...) can separate the comb teeth once the flow is flexible
enough: measured, a 114-param RealNVP prefers 437 or 348 Myr over the true 400,
and an EMD term leaves the truth last at every weight.  What distinguishes the
teeth is f0 ITSELF: at the truth it is the smooth core+wings blob, at a wrong
Delta t a partly-wound spiral with thin arms.  So the objective penalises the
COMPLEXITY of f0:

    J(h, dt; lam) = min_theta [ NLL(theta; h, dt) + lam * N * R(theta) ]

Two penalties R:
  * "fisher" -- EXCESS FISHER INFORMATION of f0 in standardised coordinates,
        R = E[ |grad_y log f0|^2 ] - 2,      y = (x - mean)/sd  of the cloud.
    For a fixed covariance the Gaussian MINIMISES Fisher information
    (Cramer-Rao), so R >= ~0, = 0 for a Gaussian, and it grows with every sharp
    feature -- thin spiral arms most of all.  Directly targets the failure.
  * "l2"     -- weight decay, R = sum theta^2.  Simple, only indirectly tied to
        how wiggly f0 is.  Kept as the comparison.

lam is NOT tuned by hand: it is chosen by K-fold HELD-OUT likelihood (fit the
flow on 4/5 of the stars, score it on the 1/5 it never saw, rotate).  Too small
a lam -> the flow memorises the training stars (spiral fits); too large -> it
cannot make the core+wings shape.  Held-out NLL measures exactly that balance,
and is also the reported measure of how well f0 is recovered.

PIPELINE (each stage cached in results/PF2w_*.pkl):
  0. candidate_teeth -- affine closed-form comb over (h, dt) on 4000 stars; the
     M deepest teeth.  PROPOSALS only: the affine model never decides.
  1. lambda_scan     -- 5-fold held-out NLL at every (tooth, penalty, lam).
                        Picks (penalty*, lam*) and shows whether the comb is back.
  2. final_fit       -- ONE JOINT minimisation of ALL parameters (h, N_steps and
                        every flow weight) from EVERY candidate tooth, in
                        parallel: flow warm-up, full-batch Adam, then a joint
                        L-BFGS finish.  Lowest J wins.  Reports the final
                        gradient so a stalled fit is visible.
  3. hessian_errors  -- full Hessian over all parameters at the optimum ->
                        (h, dt) errors marginalised over the flow weights.
                        `profile` (warm-started grid) is the fallback/check.
  4. f0_report       -- fitted vs true f0: marginals, 2-D residual, held-out
                        NLL/star minus the TRUE f0's NLL/star.
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

# ---- the data: same as the P2W notebook ----------------------------------------
COMPS = [(0.0, 15.0), (0.0, 60.0)]
WEIGHTS = [0.6, 0.4]
SEED = 3

# ---- the flow: ZWNormalizingFlow(c couplings, hidden) --------------------------
# 3 couplings (w|z, z|w, w|z) is the smallest stack that can make a NON-Gaussian
# w-marginal independent of z; ONE coupling is affine in w at fixed z and cannot
# (ZW(1,8) = 106 params gains only ~p/2 at the truth).  (3,4) -> 114 params.
FLOW_C, FLOW_HID = 3, 4

# ---- inner fit ------------------------------------------------------------------
N_ADAM = 1500
LR = 1e-2
N_LBFGS = 200
N_PEN = 4000            # points the Fisher penalty is estimated on (fixed subset)

# Calibrated 2026-10-01 on 16000 stars at h=250: unpenalised excess Fisher of the
# fitted f0 is 2.2 at dt=400 but 7.0 at 437 and 10.0 at 348; lam=0.01 halves it
# at the truth for 60 nats and puts the truth first by ~450 nats.  |theta|^2 is
# 80-100, and l2 lam=1e-3 already collapses the flow to the affine map (|theta|=0),
# hence the lower l2 ladder.
LAMS = {"fisher": (0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
        "l2": (1e-5, 3e-5, 1e-4, 3e-4)}
K_FOLDS = 5


# ================================================================================
#   pieces
# ================================================================================
def load_data(N=20000):
    return C.make_mixture_w_data(N=N, comps=COMPS, weights=WEIGHTS, seed=SEED)


def true_logf0(z0, w0):
    """log of the TRUE initial DF, per star (physical units, per pc km/s)."""
    lz = -0.5 * (z0 / C.Z0_SIGMA) ** 2 - math.log(math.sqrt(2 * math.pi) * C.Z0_SIGMA)
    fw = sum(a * C.gauss_pdf(w0, mu, s) for a, (mu, s) in zip(WEIGHTS, COMPS))
    return lz + np.log(fw)


def true_fisher_excess(zb, wb):
    """R of the TRUE f0 on the cloud, same definition as `fisher_excess` -- the
    reference a well-recovered f0 should approach (core+wings is not Gaussian,
    so it is > 0)."""
    dz = -zb / C.Z0_SIGMA ** 2
    comp = [a * C.gauss_pdf(wb, mu, s) for a, (mu, s) in zip(WEIGHTS, COMPS)]
    dw = sum(c * (-(wb - mu) / s ** 2) for c, (mu, s) in zip(comp, COMPS)) / sum(comp)
    return float(((dz * zb.std()) ** 2 + (dw * wb.std()) ** 2).mean() - 2.0)


def make_flow(zb, wb, c=FLOW_C, hid=FLOW_HID, seed=0):
    """Standardised on the cloud it is fitted to; zero-init last layers, so it
    STARTS at the diagonal-affine optimum.  Seed pinned -> deterministic."""
    from fitter import ZWNormalizingFlow
    torch.manual_seed(seed)
    f = ZWNormalizingFlow(int(c), int(hid)).double()
    f.set_standardization(zb, wb)
    return f


def fisher_excess(flow, zt, wt):
    """R = E|grad_y log f|^2 - 2 on the given points (y = standardised coords).
    Differentiable in the flow parameters (double backprop)."""
    zr = zt.detach().clone().requires_grad_(True)
    wr = wt.detach().clone().requires_grad_(True)
    lp = flow.log_prob(zr, wr)
    gz, gw = torch.autograd.grad(lp.sum(), (zr, wr), create_graph=True)
    return ((gz * flow.sig[0]) ** 2 + (gw * flow.sig[1]) ** 2).mean() - 2.0


def l2_norm(flow):
    return sum((p ** 2).sum() for p in flow.parameters())


def penalty(flow, kind, zp, wp):
    if kind == "fisher":
        return fisher_excess(flow, zp, wp)
    if kind == "l2":
        return l2_norm(flow)
    raise ValueError(kind)


def fit_flow(zb, wb, lam=0.0, kind="fisher", n_adam=N_ADAM, lr=LR,
             n_lbfgs=N_LBFGS, c=FLOW_C, hid=FLOW_HID, seed=0, warm=None):
    """min_theta  NLL/star + lam * R  on the cloud (zb, wb).
    `warm` = a parameter dict from a neighbouring fit to start from (the
    standardisation is still set from THIS cloud).
    Returns (flow, dict(nll=total nats, R=penalty value))."""
    f = make_flow(zb, wb, c, hid, seed)
    if warm is not None:
        f.load_state_dict({k: v for k, v in warm.items()
                           if k not in ("mu", "sig")}, strict=False)
    zt, wt = torch.tensor(zb), torch.tensor(wb)
    rng = np.random.default_rng(seed)
    ip = rng.choice(zb.size, size=min(N_PEN, zb.size), replace=False)
    zp, wp = zt[ip], wt[ip]
    use_pen = lam > 0

    def loss_fn():
        l = -f.log_prob(zt, wt).mean()
        if use_pen:
            l = l + lam * penalty(f, kind, zp, wp)
        return l

    opt = torch.optim.Adam(f.parameters(), lr=lr)
    for _ in range(int(n_adam)):
        l = loss_fn()
        opt.zero_grad(); l.backward(); opt.step()
    if n_lbfgs:
        lb = torch.optim.LBFGS(list(f.parameters()), max_iter=int(n_lbfgs),
                               history_size=25, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            l = loss_fn()
            l.backward()
            return l
        try:
            lb.step(closure)
        except Exception:                        # LBFGS can trip on flat dirs
            pass
    with torch.no_grad():
        nll = float(-f.log_prob(zt, wt).sum())
    R = float(penalty(f, kind, zp, wp).detach())
    return f, dict(nll=nll, R=R)


def objective(z, w, h, dt, lam, kind, **kw):
    """J(h, dt; lam) in nats: NLL + lam * N * R at the penalised optimum."""
    zb, wb = C.backint(z, w, h, t_obs=dt, dt_step=C.DT_STEP)
    _, info = fit_flow(zb, wb, lam, kind, **kw)
    return info["nll"] + lam * z.size * info["R"]


def folds(n, k=K_FOLDS, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    return np.array_split(perm, k)


def cv_score(z, w, h, dt, lam, kind, k=K_FOLDS, **kw):
    """K-fold held-out NLL (nats, summed over every star once) + the same for
    the TRUE f0, at (h, dt).  The training-set objective is returned too."""
    zb, wb = C.backint(z, w, h, t_obs=dt, dt_step=C.DT_STEP)
    held, train, R = 0.0, 0.0, []
    for idx in folds(z.size, k):
        tr = np.setdiff1d(np.arange(z.size), idx)
        f, info = fit_flow(zb[tr], wb[tr], lam, kind, **kw)
        with torch.no_grad():
            held += float(-f.log_prob(torch.tensor(zb[idx]),
                                      torch.tensor(wb[idx])).sum())
        train += info["nll"] / tr.size
        R.append(info["R"])
    # Reference = the TRUE MODEL of the observed stars: the true f0 on the cloud
    # back-integrated at the TRUE (h, Delta t).  (Evaluating the true f0 on the
    # cloud at the FITTED (h, dt) is meaningless when those are wrong -- it made
    # a wrong-tooth fit look 2.25 nats/star "better than the truth".)  Both are
    # densities of the same observed stars (Liouville: unit Jacobian).
    zt_, wt_ = C.backint(z, w, C.H_TRUE, t_obs=C.DT_TRUE, dt_step=C.DT_STEP)
    return dict(held=held, held_true=float(-true_logf0(zt_, wt_).sum()),
                train_per_star=train / k, R=float(np.mean(R)))


# ================================================================================
#   pool plumbing
# ================================================================================
_CTX = {}


def _init(z, w):
    torch.set_num_threads(1)
    _CTX.update(z=z, w=w)


def _pool_map(fn, jobs, z, w, workers):
    nw = (max((os.cpu_count() or 2) - 2, 1)) if workers is None else workers
    with Pool(min(nw, len(jobs)), initializer=_init, initargs=(z, w)) as p:
        return p.map(fn, jobs, chunksize=1)


def _cache(name, build, force):
    path = os.path.join(RESULTS, f"PF2w_{name}.pkl")
    if not force and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    out = build()
    out["wall"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


# ================================================================================
#   stage 0 -- candidate teeth (affine closed form; proposals only)
# ================================================================================
def _aff_worker(pt):
    return C._gauss_profile_nll(_CTX["z"], _CTX["w"], pt[0], pt[1])


def candidate_teeth(z, w, n_teeth=6, n_sub=4000, hgrid=None, dtgrid=None,
                    workers=None, force=False):
    hgrid = np.arange(50.0, 800.0 + 1e-9, 25.0) if hgrid is None else hgrid
    dtgrid = np.arange(0.0, 700.0 + 1e-9, 1.0) if dtgrid is None else dtgrid

    def build():
        idx = np.random.default_rng(0).choice(z.size, n_sub, replace=False)
        pts = [(float(h), float(d)) for d in dtgrid for h in hgrid]
        v = _pool_map(_aff_worker, pts, z[idx], w[idx], workers)
        Z = np.asarray(v).reshape(len(dtgrid), len(hgrid))
        Z = (Z - Z.min()) * (z.size / n_sub)
        teeth = C._comb_teeth(Z, half=3)
        depth = np.array([Z[i, j] for i, j in teeth])
        order = np.argsort(depth)[:n_teeth]
        T = [dict(h=float(hgrid[teeth[k][1]]), dt=float(dtgrid[teeth[k][0]]),
                  affine_dnll=float(depth[k])) for k in order]
        return dict(hgrid=hgrid, dtgrid=dtgrid, Z=Z, teeth=T, n_sub=n_sub)
    return _cache(f"teeth_N{z.size}", build, force)


# ================================================================================
#   stage 1 -- penalty strength by held-out likelihood
# ================================================================================
def _cv_worker(job):
    h, dt, lam, kind = job
    t0 = time.time()
    r = cv_score(_CTX["z"], _CTX["w"], h, dt, lam, kind)
    r.update(h=h, dt=dt, lam=lam, kind=kind, wall=time.time() - t0)
    return r


def lambda_scan(z, w, teeth, lams=LAMS, workers=None, force=False):
    """Held-out NLL at every (tooth, penalty, lam).  Selection rule:
    (kind*, lam*) = argmin over (kind, lam) of  min over teeth  held-out NLL --
    i.e. the penalised model that best predicts unseen stars at its own best
    tooth.  `comb_back` says whether the TRUE tooth is the held-out winner."""
    def build():
        # lam = 0 is the SAME unpenalised fit for every penalty: run it once
        # (under the first kind) and copy the rows to the others.
        kinds = list(lams)
        jobs = [(t["h"], t["dt"], float(l), k) for k in kinds for l in lams[k]
                for t in teeth if l > 0 or k == kinds[0]]
        rows = _pool_map(_cv_worker, jobs, z, w, workers)
        rows += [dict(r, kind=k) for k in kinds[1:] for r in list(rows)
                 if r["lam"] == 0 and r["kind"] == kinds[0]]
        best = min(rows, key=lambda r: r["held"])
        return dict(rows=rows, kind=best["kind"], lam=best["lam"],
                    tooth=(best["h"], best["dt"]))
    return _cache(f"lamscan_N{z.size}_c{FLOW_C}h{FLOW_HID}", build, force)


def lambda_table(scan):
    """DataFrame: held-out dNLL (nats, vs the best row) per tooth x (kind, lam)."""
    import pandas as pd
    df = pd.DataFrame(scan["rows"])
    df["held_dnll"] = df["held"] - df["held"].min()
    df["vs_true_f0_per_star"] = (df["held"] - df["held_true"]) / 20000.0
    return df.pivot_table(index="dt", columns=["kind", "lam"], values="held_dnll")


# ================================================================================
#   stage 2 -- JOINT fit of ALL parameters (h, Delta t, every flow weight)
# ================================================================================
# Hyper-parameters inherited from the tuned affine joint fitter
# (C.joint_fit_h_dt / experiments/hyperopt_time.py): full batch (the h gradient
# is below minibatch noise), Adam, lr_h 8 pc/epoch, lr_dt 4 Myr/epoch, cosine,
# grad clip 30, flow-only warm-up, low-lr polish.  The warm-up is longer here
# because a 114-param flow needs ~1500 steps where the affine map needed ~150.
JOINT = dict(warm=1500, epochs=600, polish=200, lr_flow=1e-2, lr_h=8.0,
             lr_dt=4.0, polish_lr=0.25, grad_clip=30.0, lbfgs=300)
# Adam alone does NOT reach the joint minimum (measured on 600 stars: after
# 1500+500 epochs the profile still sloped -0.2 nats/pc in h -- the minimum lies
# along a curved valley where h and the flow weights must move together, which
# Adam's per-parameter step sizes cannot follow; it stalls once the cosine
# schedule shrinks the steps).  So the fit ends with a JOINT L-BFGS over all
# parameters, which learns that coupling.  h enters in units of H_SCALE pc so
# all coordinates have O(1) curvature (sigma_h ~30 pc, sigma_N ~3 steps).
H_SCALE = 10.0


def _fisher_through(flow, zb, wb):
    """Excess Fisher information with the gradient kept THROUGH the points, so
    dR/dh and dR/d(Delta t) reach the Hamiltonian parameters (zb, wb are
    outputs of the integration, not leaves)."""
    lp = flow.log_prob(zb, wb)
    gz, gw = torch.autograd.grad(lp.sum(), (zb, wb), create_graph=True)
    return ((gz * flow.sig[0]) ** 2 + (gw * flow.sig[1]) ** 2).mean() - 2.0


def joint_fit(z, w, h0, dt0, lam, kind, seed=0, dt_min=0.0, dt_max=700.0,
              h_bounds=(50.0, 800.0), **over):
    """min over (theta, h, N_steps) of  NLL/star + lam * R  -- ONE optimisation
    of every parameter together.  Started inside a comb tooth; the 100-300-nat
    walls keep it there, so each start returns the best point OF ITS TOOTH and
    the caller compares teeth.  Returns dict(h, dt, J [nats], state, histories)."""
    p = dict(JOINT, **over)
    zt, wt = torch.tensor(z), torch.tensor(w)
    ds = C.DT_STEP
    h = torch.nn.Parameter(torch.tensor(float(h0), dtype=torch.float64))
    nst = torch.nn.Parameter(torch.tensor(float(dt0) / ds, dtype=torch.float64))
    ip = torch.tensor(np.random.default_rng(seed).choice(
        z.size, size=min(N_PEN, z.size), replace=False))

    def cloud():
        return torch_back_integrate_(zt, wt, nst * ds, h)

    def loss_fn(zb, wb):
        l = -flow.log_prob(zb, wb).mean()
        if lam > 0:
            l = l + lam * (_fisher_through(flow, zb[ip], wb[ip]) if kind == "fisher"
                           else l2_norm(flow))
        return l

    # 1. the flow alone at the starting (h, Delta t) -- same as a profile fit
    with torch.no_grad():
        zb0, wb0 = cloud()
    flow, _ = fit_flow(zb0.numpy(), wb0.numpy(), lam, kind, n_adam=p["warm"],
                       n_lbfgs=0, seed=seed)

    hist = dict(h=[], dt=[], loss=[])

    def descend(epochs, scale):
        opt = torch.optim.Adam([
            {"params": flow.parameters(), "lr": p["lr_flow"] * scale},
            {"params": [h], "lr": p["lr_h"] * scale},
            {"params": [nst], "lr": p["lr_dt"] * scale / ds}])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        for _ in range(int(epochs)):
            zb, wb = cloud()
            l = loss_fn(zb, wb)
            opt.zero_grad(); l.backward()
            torch.nn.utils.clip_grad_norm_([h], p["grad_clip"])
            torch.nn.utils.clip_grad_norm_([nst], p["grad_clip"] / ds)
            opt.step(); sch.step()
            with torch.no_grad():
                h.clamp_(*h_bounds); nst.clamp_(dt_min / ds, dt_max / ds)
            hist["h"].append(h.item()); hist["dt"].append(nst.item() * ds)
            hist["loss"].append(float(l.detach()))

    descend(p["epochs"], 1.0)                      # 2. everything together
    descend(p["polish"], p["polish_lr"])           # 3. low-lr polish

    if p["lbfgs"]:                                 # 4. joint quasi-Newton finish
        u = torch.nn.Parameter(torch.tensor(h.item() / H_SCALE, dtype=torch.float64))
        params = [u, nst] + list(flow.parameters())
        lb = torch.optim.LBFGS(params, max_iter=int(p["lbfgs"]), history_size=50,
                               tolerance_grad=1e-9, tolerance_change=1e-12,
                               line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad()
            zb_, wb_ = torch_back_integrate_(zt, wt, nst * ds, u * H_SCALE)
            l = loss_fn(zb_, wb_)
            l.backward()
            return l
        try:
            lb.step(closure)
        except Exception:
            pass
        with torch.no_grad():
            h.copy_(torch.clamp(u * H_SCALE, *h_bounds))
            nst.clamp_(dt_min / ds, dt_max / ds)
        hist["h"].append(h.item()); hist["dt"].append(nst.item() * ds)

    # convergence diagnostics: the gradient of J (nats) at the final point.  A
    # true joint minimum has all three ~0; the notebook reports them so a fit
    # that stalled is visible rather than silently quoted.
    zb, wb = cloud()
    l = loss_fn(zb, wb)
    g = torch.autograd.grad(l * z.size, [h, nst] + list(flow.parameters()))
    J = float(l.detach()) * z.size
    diag = dict(grad_h=float(g[0]), grad_dt=float(g[1]) / ds,
                grad_flow_max=max(float(x.abs().max()) for x in g[2:]))
    return dict(h0=float(h0), dt0=float(dt0), h=h.item(), dt=nst.item() * ds, J=J,
                **diag,
                state={k: v.detach().clone() for k, v in flow.state_dict().items()},
                **{f"{k}_hist": v for k, v in hist.items()})


def torch_back_integrate_(zt, wt, t, h):
    from fitter import torch_back_integrate
    return torch_back_integrate(zt, wt, t, h, C.sim, dt_step_myr=C.DT_STEP)


def _joint_worker(job):
    h0, dt0, lam, kind = job
    t0 = time.time()
    r = joint_fit(_CTX["z"], _CTX["w"], h0, dt0, lam, kind)
    r["wall"] = time.time() - t0
    return r


def final_fit(z, w, scan, teeth, workers=None, force=False):
    """One JOINT fit of all parameters from EVERY candidate tooth, in parallel;
    the lowest penalised objective J wins.  The affine model only supplied the
    starting points -- it never votes."""
    kind, lam = scan["kind"], scan["lam"]

    def build():
        jobs = [(t["h"], t["dt"], lam, kind) for t in teeth]
        runs = _pool_map(_joint_worker, jobs, z, w, workers)
        best = min(runs, key=lambda r: r["J"])
        return dict(runs=runs, h=best["h"], dt=best["dt"], J=best["J"],
                    state=best["state"], kind=kind, lam=lam)
    return _cache(f"joint_N{z.size}_c{FLOW_C}h{FLOW_HID}", build, force)


# ================================================================================
#   stage 3 -- error bars from the FULL Hessian at the joint optimum
# ================================================================================
def hessian_errors(z, w, fin, force=False):
    """Hessian of J (nats) over ALL parameters (h, N_steps, theta) at the joint
    optimum.  The (h, Delta t) uncertainty is its inverse restricted to those two
    -- i.e. marginalised over every flow weight (Schur complement), which is the
    local curvature of the profile likelihood, obtained without fixing anything.
    Flow-weight directions with ~zero curvature (network symmetries) are dropped
    by a pseudo-inverse."""
    def build():
        lam, kind = fin["lam"], fin["kind"]
        ds = C.DT_STEP
        zt, wt = torch.tensor(z), torch.tensor(w)
        h = torch.tensor(fin["h"], dtype=torch.float64, requires_grad=True)
        nst = torch.tensor(fin["dt"] / ds, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            zb0, wb0 = torch_back_integrate_(zt, wt, nst * ds, h)
        flow = make_flow(zb0.numpy(), wb0.numpy())
        flow.load_state_dict(fin["state"])
        ip = torch.tensor(np.random.default_rng(0).choice(
            z.size, size=min(N_PEN, z.size), replace=False))
        P = [h, nst] + list(flow.parameters())
        zb, wb = torch_back_integrate_(zt, wt, nst * ds, h)
        J = -flow.log_prob(zb, wb).sum()
        if lam > 0:
            J = J + lam * z.size * (_fisher_through(flow, zb[ip], wb[ip])
                                    if kind == "fisher" else l2_norm(flow))
        g = torch.cat([x.reshape(-1) for x in
                       torch.autograd.grad(J, P, create_graph=True)])
        n = g.numel()
        H = np.empty((n, n))
        for i in range(n):
            row = torch.autograd.grad(g[i], P, retain_graph=True, allow_unused=True)
            H[i] = torch.cat([(r if r is not None else torch.zeros_like(x)).reshape(-1)
                              for r, x in zip(row, P)]).numpy()
        H = 0.5 * (H + H.T)
        Hxx, Hxt, Htt = H[:2, :2], H[:2, 2:], H[2:, 2:]
        ev, V = np.linalg.eigh(Htt)
        keep = ev > 1e-8 * ev.max()
        Htt_pinv = (V[:, keep] / ev[keep]) @ V[:, keep].T
        S = Hxx - Hxt @ Htt_pinv @ Hxt.T        # curvature of the profile in (h, N)
        cov = np.linalg.inv(S)
        sig_h = math.sqrt(cov[0, 0]) if cov[0, 0] > 0 else float("nan")
        sig_dt = math.sqrt(cov[1, 1]) * ds if cov[1, 1] > 0 else float("nan")
        corr = cov[0, 1] / math.sqrt(abs(cov[0, 0] * cov[1, 1]))
        return dict(h=fin["h"], dt=fin["dt"], sig_h=sig_h, sig_dt=sig_dt, corr=corr,
                    S=S, H=H, n_dropped=int((~keep).sum()),
                    flow_min_eig=float(ev.min()),
                    is_min=bool(np.all(np.linalg.eigvalsh(S) > 0)))
    return _cache(f"hess_N{z.size}_c{FLOW_C}h{FLOW_HID}", build, force)


# ================================================================================
#   stage 3 -- profile of the penalised objective around the winner
# ================================================================================
WARM_ADAM = 300          # Adam steps for a warm-started neighbouring grid point


def _prof_row_worker(job):
    """One Delta-t row of the profile grid, swept OUTWARD in h from the centre,
    each point warm-started from its neighbour.  Cold re-fits leave ~5 nats of
    optimiser scatter (measured: 1500 vs 3000 Adam steps), which would swamp the
    h direction (Delta J = 0.5 at ~30 pc); continuation makes J smooth in h."""
    hgrid, dt, lam, kind = job
    z, w = _CTX["z"], _CTX["w"]
    out = np.empty(len(hgrid))
    k0 = len(hgrid) // 2

    def at(i, warm):
        zb, wb = C.backint(z, w, hgrid[i], t_obs=dt, dt_step=C.DT_STEP)
        kw = {} if warm is None else dict(warm=warm, n_adam=WARM_ADAM)
        f, info = fit_flow(zb, wb, lam, kind, **kw)
        out[i] = info["nll"] + lam * z.size * info["R"]
        return {k: v.clone() for k, v in f.state_dict().items()}

    centre = at(k0, None)
    for rng_ in (range(k0 + 1, len(hgrid)), range(k0 - 1, -1, -1)):
        warm = centre
        for i in rng_:
            warm = at(i, warm)
    return out


def profile(z, w, fin, h_half=100.0, n_h=17, dt_half=8.0, dt_fine=0.5,
            workers=None, force=False):
    def build():
        hgrid = np.linspace(fin["h"] - h_half, fin["h"] + h_half, int(n_h))
        dtgrid = np.arange(fin["dt"] - dt_half, fin["dt"] + dt_half + 1e-9, dt_fine)
        jobs = [(hgrid, float(d), fin["lam"], fin["kind"]) for d in dtgrid]
        J = np.asarray(_pool_map(_prof_row_worker, jobs, z, w, workers))
        J = J - J.min()
        prof_dt, prof_h = J.min(axis=1), J.min(axis=0)
        sig_dt, dt_best = C._parab_sigma(dtgrid, prof_dt)
        sig_h, h_best = C._parab_sigma(hgrid, prof_h)
        return dict(hgrid=hgrid, dtgrid=dtgrid, J=J, prof_h=prof_h,
                    prof_dt=prof_dt, sig_h=sig_h, sig_dt=sig_dt, h_best=h_best,
                    dt_best=dt_best, h_true=C.H_TRUE, dt_true=C.DT_TRUE)
    return _cache(f"profile_N{z.size}_c{FLOW_C}h{FLOW_HID}", build, force)


# ================================================================================
#   stage 2 (basin-hopping version) -- JOINT basin hopping from initial guesses
# ================================================================================
# 8 of the Gaussian notebook's 16 starting guesses, spread over the box.
HOP_STARTS_2W = ((150, 250), (220, 300), (250, 400), (290, 520), (400, 220),
                 (500, 380), (120, 430), (600, 60))


def hop_fit(z, w, scan, starts=HOP_STARTS_2W, force=False, **kw):
    """Basin hopping FROM each initial guess, every local minimisation joint over
    (h, Delta t, all flow parameters) with the penalty (scan's kind, lam*).
    Same algorithm as the Gaussian notebook's `local="joint"` runs; patience
    15 / at most 40 hops (each hop ~1 min here vs ~10 s for the affine map)."""
    opts = dict(starts=starts, seeds=(0,), flow="zw114", local="joint",
                lam=scan["lam"], pen=scan["kind"], data_tag="2w", dh=20.0,
                ddt=2.0, patience=15, n_iter=40, force=force)
    opts.update(kw)
    return C.run_hop_study(z, w, **opts)


def best_run(hop):
    """The run with the lowest FULL-SAMPLE penalised objective."""
    return min(hop["runs"], key=lambda r: r["J_full"])


def hop_errors(z, w, hop, force=False):
    """Full-Hessian errors at the best landing (all 120 parameters)."""
    b = best_run(hop)

    def build():
        return C.joint_hessian(z, w, b["h_fit"], b["dt_fit"], hop["flow"],
                               b["flow_state"], lam=hop["lam"], pen=hop["pen"])
    return _cache(f"hophess_N{z.size}", build, force)


def f0_report_hop(z, w, hop, force=False):
    """f0 recovery at the best basin-hopping landing: the JOINTLY fitted flow
    (penalised) vs an unpenalised flow fitted at the same (h, dt); each with
    5-fold held-out NLL/star minus the TRUE f0's on the same stars."""
    b = best_run(hop)

    def build():
        out = {}
        for tag, lam in (("penalised", hop["lam"]), ("unpenalised", 0.0)):
            cv = cv_score(z, w, b["h_fit"], b["dt_fit"], lam, hop["pen"])
            zb, wb = C.backint(z, w, b["h_fit"], t_obs=b["dt_fit"], dt_step=C.DT_STEP)
            if tag == "penalised":
                state, joint = b["flow_state"], True
            else:
                f, _ = fit_flow(zb, wb, 0.0, hop["pen"])
                state, joint = f.state_dict(), False
            out[tag] = dict(state=state, joint=joint, lam=lam,
                            excess_per_star=(cv["held"] - cv["held_true"]) / z.size,
                            train_per_star=cv["train_per_star"], R=cv["R"])
        zb, wb = C.backint(z, w, b["h_fit"], t_obs=b["dt_fit"], dt_step=C.DT_STEP)
        zt_, wt_ = C.backint(z, w, C.H_TRUE, t_obs=C.DT_TRUE, dt_step=C.DT_STEP)
        out.update(h=b["h_fit"], dt=b["dt_fit"], kind=hop["pen"],
                   R_true=true_fisher_excess(zb, wb),
                   R_true_on_true_cloud=true_fisher_excess(zt_, wt_), ref_fixed=True)
        return out
    return _cache(f"hopf0_N{z.size}", build, force)


def run_colors(hop):
    """Run -> colour, the SAME mapping as the right panel of
    C.plot_hdt_hop_paths (turbo over runs sorted by (h_init, dt_init, seed))."""
    import matplotlib.pyplot as plt
    runs = hop["runs"]
    order = sorted(range(len(runs)), key=lambda k: (runs[k]["h_init"],
                                                    runs[k]["dt_init"],
                                                    runs[k]["seed"]))
    cols = plt.cm.turbo(np.linspace(0.04, 0.96, len(order)))
    return {k: c for k, c in zip(order, cols)}


def _joint_flow(hop, run):
    f = C.make_joint_flow(hop["flow"])
    f.load_state_dict(run["flow_state"])
    return f


def _marginals_on_grid(logp_fn, zg, wg):
    """1-D marginals of a 2-D density given as log p(z, w) on a grid."""
    ZG, WG = np.meshgrid(zg, wg)                       # (len(wg), len(zg))
    p = np.exp(logp_fn(ZG.ravel(), WG.ravel())).reshape(ZG.shape)
    dz, dw = zg[1] - zg[0], wg[1] - wg[0]
    return p.sum(axis=0) * dw, p.sum(axis=1) * dz


def _obs_logp(f0_logp, h, dt):
    """Model density of the OBSERVED stars implied by an initial density f0 and
    (h, Delta t): p_obs(x) = f0(T^{-1} x) -- the back-integration preserves
    phase-space volume (Liouville), so there is no Jacobian."""
    def lp(zo, wo):
        zb, wb = C.backint(zo, wo, h, t_obs=dt, dt_step=C.DT_STEP)
        return f0_logp(zb, wb)
    return lp


def plot_run_fits(z, w, hop, title=None, ng=160):
    """The fit of every run's normalizing flow, one colour per run.

    TOP row -- the fit TO THE DATA (t = 400 Myr, observed coordinates): the
      observed stars (grey histogram), the TRUE model of them (black: true f0
      carried to t_obs at the true h, Delta t), and each run's model (its f0
      carried to t_obs at ITS OWN h, Delta t).  z left, w right.
    BOTTOM row -- the recovered f0 itself (t = 0): each run's f0 vs the true f0
      (black, with the two generating w-Gaussians dotted); grey = the stars
      back-integrated at the TRUE (h, Delta t)."""
    import matplotlib.pyplot as plt
    runs, cols = hop["runs"], run_colors(hop)
    Jb = best_run(hop)["J_full"]
    fig, axs = plt.subplots(2, 2, figsize=(15, 9.5),
                            gridspec_kw=dict(width_ratios=[1, 1.35]))

    def f0_lp(flow):
        def lp(z0, w0):
            with torch.no_grad():
                return flow.log_prob(torch.tensor(z0), torch.tensor(w0)).numpy()
        return lp

    # ---- top row: observed frame ------------------------------------------
    zq = np.percentile(np.abs(z), 99.7) * 1.15
    zg = np.linspace(-zq, zq, ng)
    wg = np.linspace(-220.0, 220.0, ng)
    axz, axw = axs[0]
    axz.hist(z, bins=90, range=(-zq, zq), density=True, color="0.82",
             label="observed stars (data)")
    axw.hist(w, bins=120, range=(wg[0], wg[-1]), density=True, color="0.82",
             label="observed stars (data)")
    mz, mw_true = _marginals_on_grid(_obs_logp(true_logf0, C.H_TRUE, C.DT_TRUE), zg, wg)
    axz.plot(zg, mz, "k-", lw=2.4, label="true model", zorder=5)
    axw.plot(wg, mw_true, "k-", lw=2.4, label="true model", zorder=5)
    for k, r in enumerate(runs):
        fl = _joint_flow(hop, r)
        mz, mw = _marginals_on_grid(_obs_logp(f0_lp(fl), r["h_fit"], r["dt_fit"]), zg, wg)
        lab = (f"start ({r['h_init']:.0f}, {r['dt_init']:.0f}) $\\to$ "
               f"({r['h_fit']:.0f} pc, {r['dt_fit']:.1f} Myr), "
               f"$\\Delta J$={r['J_full'] - Jb:.0f}")
        axz.plot(zg, mz, color=cols[k], lw=1.4)
        axw.plot(wg, mw, color=cols[k], lw=1.4, label=lab)
    axz.set_xlabel("z [pc]"); axw.set_xlabel("w [km/s]")
    axz.set_ylabel("density")
    axz.set_title("fit to the DATA (t = 400 Myr): z marginal")
    axw.set_title("fit to the DATA (t = 400 Myr): w marginal (log)")
    axw.set_yscale("log"); axw.set_ylim(1e-6, 3.0 * float(mw_true.max()))
    axw.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))

    # ---- bottom row: recovered f0 at t = 0 --------------------------------
    zt, wt = C.backint(z, w, C.H_TRUE, t_obs=C.DT_TRUE, dt_step=C.DT_STEP)
    z0g = np.linspace(-4 * C.Z0_SIGMA, 4 * C.Z0_SIGMA, ng)
    w0g = np.linspace(-220.0, 220.0, ng)
    bz, bw = axs[1]
    bz.hist(zt, bins=90, range=(z0g[0], z0g[-1]), density=True, color="0.82",
            label="stars back-integrated at the TRUE (h, $\\Delta t$)")
    bw.hist(wt, bins=120, range=(w0g[0], w0g[-1]), density=True, color="0.82",
            label="stars back-integrated at the TRUE (h, $\\Delta t$)")
    bz.plot(z0g, C.gauss_pdf(z0g, 0.0, C.Z0_SIGMA), "k-", lw=2.4, label="true $f_0$")
    bw.plot(w0g, C.mixture_pdf(COMPS, WEIGHTS)(w0g), "k-", lw=2.4, label="true $f_0$")
    for a, (mu, s) in zip(WEIGHTS, COMPS):
        bw.plot(w0g, a * C.gauss_pdf(w0g, mu, s), "k:", lw=1.0,
                label=f"component {a:.1f} N(0,{s:.0f})")
    for k, r in enumerate(runs):
        mz, mw = _marginals_on_grid(f0_lp(_joint_flow(hop, r)), z0g, w0g)
        bz.plot(z0g, mz, color=cols[k], lw=1.4)
        bw.plot(w0g, mw, color=cols[k], lw=1.4)
    bz.set_xlabel("z [pc]"); bw.set_xlabel("w [km/s]"); bz.set_ylabel("density")
    bz.set_title("recovered $f_0$ (t = 0): z marginal")
    bw.set_title("recovered $f_0$ (t = 0): w marginal (log)")
    bw.set_yscale("log"); bw.set_ylim(1e-6, None)
    bz.legend(fontsize=7.5); bw.legend(fontsize=7.5, loc="center left",
                                       bbox_to_anchor=(1.01, 0.5))
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def hop_table(hop):
    """One row per start: guess, landing, full-sample J relative to the best,
    search effort, and the final-fit flags."""
    import pandas as pd
    Jb = best_run(hop)["J_full"]
    rows = [dict(h_init=r["h_init"], dt_init=r["dt_init"], h_hop=r["h_hop"],
                 dt_hop=r["dt_hop"], h_fit=r["h_fit"], dt_fit=r["dt_fit"],
                 dJ_full=r["J_full"] - Jb, n_hops=r["n_hops"],
                 evals=r["n_eval_sub"] + r.get("n_eval_full", 0),
                 wall_min=r["wall"] / 60.0)
            for r in hop["runs"]]
    return pd.DataFrame(rows).sort_values("dJ_full")


# ================================================================================
#   stage 4 -- how well is f0 recovered?
# ================================================================================
def f0_report(z, w, fin, force=False):
    """Final flow at (h_hat, dt_hat, lam*) and the UNPENALISED flow at the same
    point, each with held-out NLL/star minus the TRUE f0's (0 = perfect)."""
    def build():
        out = {}
        for tag, lam in (("penalised", fin["lam"]), ("unpenalised", 0.0)):
            cv = cv_score(z, w, fin["h"], fin["dt"], lam, fin["kind"])
            zb, wb = C.backint(z, w, fin["h"], t_obs=fin["dt"], dt_step=C.DT_STEP)
            if tag == "penalised" and "state" in fin:    # THE jointly fitted f0
                f = make_flow(zb, wb); f.load_state_dict(fin["state"])
            else:                                        # same point, lam = 0
                f, _ = fit_flow(zb, wb, lam, fin["kind"])
            out[tag] = dict(state=f.state_dict(), lam=lam,
                            excess_per_star=(cv["held"] - cv["held_true"]) / z.size,
                            train_per_star=cv["train_per_star"], R=cv["R"])
        zb, wb = C.backint(z, w, fin["h"], t_obs=fin["dt"], dt_step=C.DT_STEP)
        out.update(h=fin["h"], dt=fin["dt"], kind=fin["kind"],
                   R_true=true_fisher_excess(zb, wb))
        return out
    return _cache(f"f0_N{z.size}_c{FLOW_C}h{FLOW_HID}", build, force)


def plot_f0(z, w, rep, title=None):
    """Top: z and w marginals of the fitted f0 (penalised, unpenalised) vs the
    truth, with the two generating w-Gaussians.  Bottom: 2-D residual
    (fitted - true) of the penalised f0 at t = 0."""
    import matplotlib.pyplot as plt
    zb, wb = C.backint(z, w, rep["h"], t_obs=rep["dt"], dt_step=C.DT_STEP)
    flows = {}
    for tag in ("penalised", "unpenalised"):
        if rep[tag].get("joint"):                   # the jointly fitted flow
            f = C.JointFlow(C._joint_layers("zw114")).double()
        else:
            f = make_flow(zb, wb)
        f.load_state_dict(rep[tag]["state"]); flows[tag] = f

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    axz, axw = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1:])
    zg = np.linspace(-4 * C.Z0_SIGMA, 4 * C.Z0_SIGMA, 300)
    wg = np.linspace(-200, 200, 400)
    axz.hist(zb, bins=80, density=True, color="0.85", label="back-integrated stars")
    axw.hist(wb, bins=120, density=True, color="0.85", label="back-integrated stars")
    axz.plot(zg, C.gauss_pdf(zg, 0.0, C.Z0_SIGMA), "k-", lw=2, label="true $f_0$")
    axw.plot(wg, C.mixture_pdf(COMPS, WEIGHTS)(wg), "k-", lw=2, label="true $f_0$")
    for a, (mu, s) in zip(WEIGHTS, COMPS):
        axw.plot(wg, a * C.gauss_pdf(wg, mu, s), "k:", lw=1,
                 label=f"component {a:.1f} N(0,{s:.0f})")
    cols = {"penalised": "C3", "unpenalised": "C0"}
    for tag, f in flows.items():
        zz, mz, ww, mw = C.fitted_marginals(f, zb, wb)
        r = rep[tag]
        lab = (f"{tag} ($\\lambda$={r['lam']:g}): held-out "
               f"{r['excess_per_star']:+.4f} nats/star vs true")
        axz.plot(zz, mz, color=cols[tag], lw=1.6, ls="--" if tag[0] == "u" else "-")
        axw.plot(ww, mw, color=cols[tag], lw=1.6, ls="--" if tag[0] == "u" else "-",
                 label=lab)
    axz.set_xlim(zg[0], zg[-1]); axw.set_xlim(wg[0], wg[-1])
    axz.set_xlabel("z [pc]"); axw.set_xlabel("w [km/s]")
    axw.set_yscale("log"); axw.set_ylim(1e-5, None)
    axw.legend(fontsize=8, loc="upper right"); axz.legend(fontsize=8)
    axz.set_title("z marginal of $f_0$"); axw.set_title("w marginal of $f_0$ (log)")

    ZG, WG = np.meshgrid(np.linspace(-3 * C.Z0_SIGMA, 3 * C.Z0_SIGMA, 160),
                         np.linspace(-150, 150, 160))
    pt = np.exp(true_logf0(ZG, WG))
    for k, tag in enumerate(("penalised", "unpenalised")):
        ax = fig.add_subplot(gs[1, k])
        with torch.no_grad():
            pf = np.exp(flows[tag].log_prob(torch.tensor(ZG.ravel()),
                                            torch.tensor(WG.ravel())).numpy()
                        ).reshape(ZG.shape)
        vmax = np.abs(pf - pt).max()
        im = ax.pcolormesh(ZG, WG, pf - pt, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           shading="auto")
        ax.set_title(f"{tag}: fitted $-$ true $f_0$ at $t=0$", fontsize=10)
        ax.set_xlabel("z [pc]"); ax.set_ylabel("w [km/s]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[1, 2])
    im = ax.pcolormesh(ZG, WG, pt, cmap="viridis", shading="auto")
    ax.set_title("true $f_0$", fontsize=10); ax.set_xlabel("z [pc]")
    fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title, fontsize=13)
    return fig


def plot_lambda_scan(scan, title=None):
    """Held-out NLL of every candidate tooth vs lam, one panel per penalty.
    The comb is 'back' where the dt=400 curve is the lowest."""
    import matplotlib.pyplot as plt
    rows = scan["rows"]
    kinds = sorted({r["kind"] for r in rows})
    ref = min(r["held"] for r in rows)
    fig, axs = plt.subplots(1, len(kinds), figsize=(6.2 * len(kinds), 4.4),
                            squeeze=False)
    for ax, k in zip(axs[0], kinds):
        rk = [r for r in rows if r["kind"] == k]
        lams = sorted({r["lam"] for r in rk})
        x = np.arange(len(lams))
        for dt in sorted({r["dt"] for r in rk}):
            y = [next(r["held"] for r in rk if r["dt"] == dt and r["lam"] == l) - ref
                 for l in lams]
            ax.plot(x, y, "o-", lw=2.4 if abs(dt - C.DT_TRUE) < 3 else 1.2,
                    label=f"$\\Delta t$={dt:.0f} Myr"
                          + (" (true tooth)" if abs(dt - C.DT_TRUE) < 3 else ""))
        ax.set_xticks(x); ax.set_xticklabels([f"{l:g}" for l in lams])
        ax.set_xlabel("$\\lambda$"); ax.set_ylabel("held-out NLL $-$ best [nats]")
        ax.set_title(f"penalty: {k}"); ax.legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
