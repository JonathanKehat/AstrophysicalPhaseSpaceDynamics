"""Test C -- bijection-size sweep: does a bigger flow constrain h less?

Two reliable, complementary views on the SAME centred-Gaussian data:

LEFT panel -- the profile-likelihood FLOORS, computed in closed form (no flow
re-fitting, so no optimisation noise).  The best NLL a model family can reach at
a fixed h is  N*[H(cloud) + KL(cloud || best member)].  By Liouville the entropy
H(cloud(h)) is h-independent, so all h-dependence is the model misfit.  We show
three families that bracket every flow:
  * diagonal Gaussian (like the 4-param affine, NO z-w correlation): a SHARP
    well -- at the wrong h the back-integrated cloud shears and grows a z-w
    correlation, which a diagonal model can only absorb by inflating its
    marginal variances -> NLL rises -> h is pinned.
  * full Gaussian (adds ONE nuisance, the correlation rho): the well nearly
    FLATTENS -- rho soaks up the shear (det Cov is conserved by the symplectic
    flow), so the extra parameter already removes most of the h-constraint.
  * flexible limit: EXACTLY FLAT.  A universal density reaches the entropy floor
    H(cloud) at every h, and that entropy is conserved by the symplectic flow
    (H(T_h x) = H(x) since det J = 1), so the floor is h-independent.  We draw
    this as an exact flat line rather than estimating it (a kNN entropy estimate
    is itself badly biased for the sheared clouds at wrong h -- unreliable).

RIGHT panel -- the MEASURED spread of the recovered h over three inits
{150,250,350} for five real bijections (4 -> 35344 params).  This uses joint-fit
endpoints only (not the noisy per-flow profile) and rises monotonically, the
empirical confirmation that more nuisance parameters loosen h.
"""
import numpy as np
import matplotlib.pyplot as plt

from experiments import common as C
from experiments.exp_A_linear_bijection import DiagonalAffineFlow
from experiments.exp_B_tiny_nn import TinyNNFlow
from fitter import ZWNormalizingFlow


def _realnvp(z, w, n_c, n_h):
    def mk():
        f = ZWNormalizingFlow(n_couplings=n_c, hidden=n_h).double()
        zi, wi = C.backint(z, w, 300.0, ns=1000)
        f.set_standardization(zi, wi)
        return f
    n_params = sum(p.numel() for p in ZWNormalizingFlow(n_c, n_h).parameters())
    return mk, n_params


def run(cfg):
    N = cfg.get("N", 2000)
    z, w = C.make_gaussian_data(N=N)
    inits = (150.0, 250.0, 350.0)

    def mk_tiny():
        f = TinyNNFlow().double()
        zb, wb = C.backint(z, w, 300.0, ns=1000)
        f.set_std(zb, wb)
        return f

    specs = [
        ("affine", lambda: DiagonalAffineFlow((C.Z0_SIGMA * 1.5, C.W0_SIGMA * 1.5)).double(), 4, 5e-2),
        ("tiny NN", mk_tiny, 8, 5e-3),
    ]
    for n_c, n_h in [(4, 6), (6, 12), (8, 64)]:
        mk, npar = _realnvp(z, w, n_c, n_h)
        specs.append((f"RealNVP c{n_c}h{n_h}", mk, npar, 1e-3))

    # ---- measured constraint: spread of recovered h over inits (joint fits) ----
    sweep = []
    for name, mk, npar, lr in specs:
        hs = [C.joint_fit(mk, z, w, hi, lr_flow=lr)[1] for hi in inits]
        sweep.append(dict(name=name, n=npar, hs=hs, spread=float(max(hs) - min(hs))))

    # ---- exact closed-form profile floors (diagonal & full Gaussian) ----
    #   flexible limit is EXACTLY flat (Liouville), drawn as a reference in plot.
    hgrid = np.linspace(160.0, 340.0, 25)
    diag = np.empty(len(hgrid)); full = np.empty(len(hgrid))
    for i, h in enumerate(hgrid):
        zb, wb = C.backint(z, w, h)
        diag[i] = N * (np.log(2 * np.pi) + np.log(zb.std()) + np.log(wb.std()) + 1.0)
        cov = np.cov(np.stack([zb, wb]))
        full[i] = N * 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(cov))
    diag -= diag.min(); full -= full.min()
    sig_diag = 0.5 * (C.cross(hgrid, diag, 0.5, +1) - C.cross(hgrid, diag, 0.5, -1))
    sig_full = 0.5 * (C.cross(hgrid, full, 0.5, +1) - C.cross(hgrid, full, 0.5, -1))

    return dict(id="C", sweep=sweep, inits=list(inits), h_true=C.H_TRUE,
                hgrid=hgrid, diag=diag, full=full,
                sig_diag=float(sig_diag), sig_full=float(sig_full))


def plot(res):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))

    ax[0].plot(res["hgrid"], res["diag"], "-", color="C2", lw=2.2,
               label=fr'diagonal Gaussian (4p): sharp well, $\sigma_h\approx${res["sig_diag"]:.0f} pc')
    ax[0].plot(res["hgrid"], res["full"], "-", color="C0", lw=2.0,
               label=fr'full Gaussian (+corr., 5p): $\sigma_h\approx${res["sig_full"]:.0f} pc')
    ax[0].axhline(0.0, color="C3", lw=2.0, ls="-",
                  label="flexible limit: flat (entropy conserved, Liouville)")
    ax[0].axhline(0.5, color="k", ls=":", label=r"$\Delta$NLL=0.5 (1$\sigma$)")
    ax[0].axvline(res["h_true"], color="k", ls="--", label=f'truth {res["h_true"]:.0f}')
    ax[0].set_ylim(0, 6); ax[0].set_xlabel(r"$h_{\rm dm}$ [pc]")
    ax[0].set_ylabel(r"profile floor $\Delta$NLL$_{\rm total}$ [nats]")
    ax[0].set_title("exact profile floors: rigid = sharp well, flexible = flat\n"
                    "(one nuisance -- the z-w correlation -- already flattens it)")
    ax[0].legend(fontsize=8)

    ns = [s["n"] for s in res["sweep"]]
    ax[1].plot(ns, [s["spread"] for s in res["sweep"]], "s-", color="C0", ms=7,
               label="measured (joint fits, 3 inits)")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of bijection parameters")
    ax[1].set_ylabel(r"recovered-$h$ spread [pc]")
    ax[1].set_title("h gets LESS constrained with more nuisance parameters")
    ax[1].legend(fontsize=9)
    for s in res["sweep"]:
        ax[1].annotate(s["name"].replace("RealNVP ", ""), (s["n"], s["spread"]),
                       fontsize=7, textcoords="offset points", xytext=(0, 6))

    fig.suptitle("Test C -- bijection-size sweep on centred-Gaussian data", fontsize=13)
    fig.tight_layout()
    return fig
