"""Joint fit (h_dm, Omega) FIRST, then profile h_dm around the joint optimum.

This is the procedure the user asked for, and it is NOT a scan over a fixed
200-300 grid:

  1. gradient-descend h_dm together with every flow parameter -> h_hat (the
     joint MLE) and Omega_hat.  Several h_init values check it is a real
     optimum, not an init echo.
  2. put a LOCAL grid around h_hat (h_hat + offsets, +/-60 pc), and at each
     fixed h re-optimise ALL the other parameters, warm-started outward from
     Omega_hat -> NLL_p(h) = min_Omega NLL(h, Omega).
  3. Delta-NLL(h) = NLL_p(h) - min: its curvature gives sigma(h_dm) from the
     Wilks Delta = 0.5 crossings.

Refits are Adam + an LBFGS polish, because with N=20000 stars a residual
optimisation error of 1e-5 nats/star is already 0.2 nats on the profile -- the
same order as the Delta=0.5 threshold.  Without the polish the "uncertainty
curve" would be optimiser noise.
"""
import sys, os, time, math, copy, pickle, argparse
import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)
torch.set_num_threads(4)

from experiments import common as C
from experiments.common import TinyNVP
from fitter import ZWNormalizingFlow

OFFS = np.array([-60., -45., -32., -22., -15., -10., -6.5, -4., -2., -1.,
                 0., 1., 2., 4., 6.5, 10., 15., 22., 32., 45., 60.])


# ---------------------------------------------------------------- optimisers
def refit(flow, zt, wt, lr, adam_epochs=400, lbfgs_iter=120):
    """Converge Omega at a FIXED cloud (fixed h).  Adam then LBFGS polish."""
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    for _ in range(adam_epochs):
        opt.zero_grad()
        loss = -flow.log_prob(zt, wt).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
        opt.step()
    lb = torch.optim.LBFGS(list(flow.parameters()), max_iter=lbfgs_iter,
                           history_size=25, tolerance_grad=1e-12,
                           tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        lb.zero_grad()
        l = -flow.log_prob(zt, wt).mean()
        l.backward()
        return l
    try:
        lb.step(closure)
    except Exception as e:                       # LBFGS can trip on flat dirs
        print("    LBFGS skipped:", e, flush=True)
    with torch.no_grad():
        return float(-flow.log_prob(zt, wt).mean())


def cross(hs, d, level, side):
    k = int(np.argmin(d))
    rng_ = range(k, len(hs) - 1) if side > 0 else range(k, 0, -1)
    for i in rng_:
        j = i + 1 if side > 0 else i - 1
        if (d[i] - level) * (d[j] - level) <= 0 and d[j] != d[i]:
            f = (level - d[i]) / (d[j] - d[i])
            return hs[i] + f * (hs[j] - hs[i])
    return np.nan


def parab_min(hs, d):
    """Vertex of the parabola through the 3 points around the discrete min."""
    k = int(np.argmin(d))
    if k == 0 or k == len(d) - 1:
        return float(hs[k]), np.nan
    x = hs[k - 1:k + 2]; y = d[k - 1:k + 2]
    a, b, _ = np.polyfit(x, y, 2)
    if a <= 0:
        return float(hs[k]), np.nan
    return float(-b / (2 * a)), float(math.sqrt(0.5 / a))   # sigma from a*dh^2=0.5


# ---------------------------------------------------------------- one model
def run_model(label, make, z, w, lr, inits, seed=0,
              n_epochs=500, ns=1000, n_sub=3000, cloud_ns=2000):
    N = z.size
    joints = []
    for hi in inits:
        t0 = time.time()
        flow, hh, hist = C.joint_fit(make, z, w, hi, n_epochs=n_epochs, ns=ns,
                                     n_sub=n_sub, lr_flow=lr, seed=seed)
        zb, wb = C.backint(z, w, hh, ns=cloud_ns)
        with torch.no_grad():
            nll = N * float(-flow.log_prob(torch.tensor(zb), torch.tensor(wb)).mean())
        joints.append(dict(h_init=float(hi), h=float(hh), nll=nll,
                           hist=np.asarray(hist, dtype=np.float32),
                           state=copy.deepcopy(flow.state_dict())))
        print(f"  [{label}] joint h_init={hi:6.1f} -> h_hat={hh:7.2f}  "
              f"NLL={nll:12.3f}  ({time.time()-t0:.0f}s)", flush=True)

    best = min(joints, key=lambda d: d["nll"])
    h_hat = best["h"]
    hgrid = h_hat + OFFS
    k0 = int(np.argmin(np.abs(OFFS)))

    clouds = []
    for h in hgrid:
        zb, wb = C.backint(z, w, float(h), ns=cloud_ns)
        clouds.append((torch.tensor(zb), torch.tensor(wb)))

    nll = np.full(len(hgrid), np.nan)
    t0 = time.time()
    fu = make(); fu.load_state_dict(best["state"])
    nll[k0] = N * refit(fu, *clouds[k0], lr, adam_epochs=800)
    for i in range(k0 + 1, len(hgrid)):                       # outward, up
        nll[i] = N * refit(fu, *clouds[i], lr)
    fd = make(); fd.load_state_dict(best["state"])
    refit(fd, *clouds[k0], lr, adam_epochs=800)
    for i in range(k0 - 1, -1, -1):                           # outward, down
        nll[i] = N * refit(fd, *clouds[i], lr)
    dnll = nll - nll.min()
    h_min, sig_par = parab_min(hgrid, dnll)
    lo, hi_ = cross(hgrid, dnll, 0.5, -1), cross(hgrid, dnll, 0.5, +1)
    print(f"  [{label}] profile done ({time.time()-t0:.0f}s): h_min={h_min:.1f} "
          f"sigma_parab={sig_par:.1f}  1sig=[{lo:.1f},{hi_:.1f}]  "
          f"well(+/-60)={dnll.max():.1f} nats", flush=True)

    return dict(label=label, npar=sum(p.numel() for p in make().parameters()),
                lr=lr, h_hat=h_hat, h_inits=[d["h_init"] for d in joints],
                h_joints=[d["h"] for d in joints],
                nll_joints=[d["nll"] for d in joints],
                hist=[d["hist"] for d in joints],
                hgrid=hgrid, nll=nll, dnll=dnll, h_min=h_min,
                sigma=sig_par, lo=lo, hi=hi_)


# ---------------------------------------------------------------- the cases
def build(case):
    if case == "2w":
        z, w = C.make_mixture_w_data(N=20000, comps=[(0., 15.), (0., 60.)],
                                     weights=[0.6, 0.4], seed=3)
    else:
        z, w = C.make_gaussian_data(N=20000, seed_sample=3)
    zi, wi = C.backint(z, w, 300.0, ns=1000)

    def zw(c, hid):
        def make():
            f = ZWNormalizingFlow(c, hid).double(); f.set_standardization(zi, wi)
            return f
        return make

    def tiny(K):
        def make():
            f = TinyNVP(K).double(); f.set_standardization(zi, wi)
            return f
        return make

    def affine():
        from experiments.exp_A_linear_bijection import DiagonalAffineFlow
        return DiagonalAffineFlow(a_init=(C.Z0_SIGMA, C.W0_SIGMA)).double()

    if case == "2w":
        specs = [("affine 4p", affine, 3e-2, (250.,)),
                 ("204p", zw(3, 6), 1e-3, (250.,)),
                 ("272p", zw(4, 6), 1e-3, (200., 250., 320.)),
                 ("304p", zw(8, 4), 1e-3, (250.,))]
    else:
        specs = [("affine 4p", affine, 3e-2, (200., 250., 320.)),
                 ("8p", tiny(2), 5e-3, (250.,)),
                 ("16p", tiny(4), 5e-3, (250.,)),
                 ("272p", zw(4, 6), 1e-3, (250.,))]
    return z, w, specs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=["2w", "pg"])
    a = ap.parse_args()
    t00 = time.time()
    z, w, specs = build(a.case)
    print(f"=== case {a.case}: N={z.size} ===", flush=True)
    out = []
    for label, make, lr, inits in specs:
        out.append(run_model(label, make, z, w, lr, inits))
    res = dict(case=a.case, h_true=C.H_TRUE, models=out, z=z, w=w)
    if a.case == "pg":                      # exact closed-form reference curve
        hg = np.linspace(190., 310., 61)
        res["analytic"] = dict(hgrid=hg,
                               dnll=C._analytic_gauss_profile(z, w, hg, z.size))
    path = os.path.join(PROJ, "results", f"JP_{a.case}.pkl")
    with open(path, "wb") as f:
        pickle.dump(res, f)
    print(f"WROTE {path}  [{(time.time()-t00)/60:.1f} min]", flush=True)
