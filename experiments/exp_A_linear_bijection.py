"""Test A -- linear bijection F(u) = a*u + b (4 parameters).

The true f0 is a single centred 2-D Gaussian, so the base->data map only has to
rescale/shift each axis: a diagonal affine flow with exactly 4 numbers
(a_z, a_w, b_z, b_w).  Fitted jointly with h_dm.  Checks that a -> the true
sigmas, b -> 0, and h -> 250, and that the profile likelihood is a sharp,
resolved well (unlike a full RealNVP).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from experiments import common as C

N_PARAMS = 4


class DiagonalAffineFlow(nn.Module):
    """Base u ~ N(0, I); data x = a*u + b (elementwise). Exactly 4 params."""
    def __init__(self, a_init=(1.0, 1.0), b_init=(0.0, 0.0)):
        super().__init__()
        self.log_a = nn.Parameter(torch.log(torch.tensor(a_init, dtype=torch.float64)))
        self.b = nn.Parameter(torch.tensor(b_init, dtype=torch.float64))

    def log_prob(self, z, w):
        x = torch.stack([z, w], dim=1)
        u = (x - self.b) / torch.exp(self.log_a)
        return -0.5 * (u ** 2).sum(1) - math.log(2 * math.pi) - self.log_a.sum()


def run(cfg):
    N = cfg.get("N", 2000)
    inits = cfg.get("inits", [180, 350])
    z_obs, w_obs = C.make_gaussian_data(N=N)

    # deliberately mis-initialise a (1.5x too wide) so the fit has to move it
    mk = lambda: DiagonalAffineFlow(a_init=(C.Z0_SIGMA * 1.5, C.W0_SIGMA * 1.5)).double()

    fits = {}
    flows = {}
    for hi in inits:
        flow, hf, hist = C.joint_fit(mk, z_obs, w_obs, hi)
        fits[hi] = dict(h=hf, hist=hist,
                        a=torch.exp(flow.log_a).detach().numpy().tolist(),
                        b=flow.b.detach().numpy().tolist())
        flows[hi] = flow
    h_est = float(np.mean([fits[hi]["h"] for hi in inits]))

    # ---- profile likelihood (closed form: diagonal-Gaussian MLE = entropy) ----
    hgrid = np.linspace(h_est - 90, h_est + 90, 41)

    def _profile(h):
        zb, wb = C.backint(z_obs, w_obs, h)
        return N * (math.log(2 * math.pi) + math.log(zb.std()) + math.log(wb.std()) + 1.0)

    nll = np.array([_profile(h) for h in hgrid])
    dnll = nll - nll.min()
    h_min = float(hgrid[dnll.argmin()])
    lo, hi_ = C.cross(hgrid, dnll, 0.5, -1), C.cross(hgrid, dnll, 0.5, +1)
    sig = 0.5 * (hi_ - lo)

    # ---- marginals of the fitted f0 at the first init's fit ----
    flow0 = flows[inits[0]]
    zb, wb = C.backint(z_obs, w_obs, fits[inits[0]]["h"])
    zg, mz, wg, mw = C.fitted_marginals(flow0, zb, wb)

    return dict(
        id="A", n_params=N_PARAMS, inits=inits, fits=fits, h_est=h_est,
        h_true=C.H_TRUE, z0_sigma=C.Z0_SIGMA, w0_sigma=C.W0_SIGMA,
        prof_h=hgrid, prof_dnll=dnll, h_min=h_min, sig=sig, lo=lo, hi=hi_,
        z0_samples=zb, w0_samples=wb, zg=zg, mz=mz, wg=wg, mw=mw,
    )


def _plot_marginals(axz, axw, res):
    for ax, samp, grid, marg, s_true, lab, unit in [
            (axz, res["z0_samples"], res["zg"], res["mz"], res["z0_sigma"], "z_0", "pc"),
            (axw, res["w0_samples"], res["wg"], res["mw"], res["w0_sigma"], "w_0", "km/s")]:
        ax.hist(samp, bins=45, density=True, alpha=0.35, color="C0",
                label="back-integrated samples")
        ax.plot(grid, C.gauss_pdf(grid, 0.0, s_true), "k--", lw=2,
                label=fr"true $f_0$: $\mathcal{{N}}(0,{s_true:.1f})$")
        ax.plot(grid, marg, color="C3", lw=2, label="fitted $f_0$ marginal")
        ax.set_xlabel(fr"${lab}$ [{unit}]"); ax.set_ylabel("density")
        ax.legend(fontsize=8)


def plot(res):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for hi in res["inits"]:
        f = res["fits"][hi]
        ax[0, 0].plot(f["hist"], label=f'init h={hi} -> {f["h"]:.1f}')
    ax[0, 0].axhline(res["h_true"], color="k", ls="--", label=f'truth {res["h_true"]:.0f}')
    ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel(r"$h_{\rm dm}$ [pc]")
    ax[0, 0].set_title("joint (linear $f_0$)+$h$ converges from both starts")
    ax[0, 0].legend()

    ax[0, 1].plot(res["prof_h"], res["prof_dnll"], "o-", color="C3", ms=4,
                  label="profile $\\Delta$NLL(h)")
    ax[0, 1].axhline(0.5, color="k", ls=":", label=r"$\Delta$NLL=0.5 (1$\sigma$)")
    ax[0, 1].axvspan(res["lo"], res["hi"], color="C2", alpha=0.15,
                     label=fr'{res["h_min"]:.0f}$\pm${res["sig"]:.0f} pc')
    ax[0, 1].axvline(res["h_true"], color="k", ls="--", label=f'truth {res["h_true"]:.0f}')
    ax[0, 1].set_ylim(0, 6); ax[0, 1].set_xlabel(r"$h_{\rm dm}$ [pc]")
    ax[0, 1].set_ylabel(r"$\Delta$NLL$_{\rm total}$ [nats]")
    ax[0, 1].set_title("profile likelihood: sharp well -> $h$ constrained")
    ax[0, 1].legend(fontsize=8)

    _plot_marginals(ax[1, 0], ax[1, 1], res)
    ax[1, 0].set_title(r"$z_0$ marginal"); ax[1, 1].set_title(r"$w_0$ marginal")
    fig.suptitle(f"Test A -- linear bijection ({res['n_params']} params):  "
                 f"h = {res['h_min']:.0f} $\\pm$ {res['sig']:.0f} pc", fontsize=13)
    fig.tight_layout()
    return fig
