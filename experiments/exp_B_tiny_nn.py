"""Test B -- tiny neural-network bijection with only 8 parameters.

The smallest genuine RealNVP-style flow: two affine coupling layers, each with a
tanh non-linearity + a single Linear(1,2) head (4 params each = 8 total), versus
the ~35k parameters of the full RealNVP.  Fitted jointly with h_dm.  Checks that
h is still recovered with a resolved profile-likelihood well.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from experiments import common as C
from experiments.exp_A_linear_bijection import _plot_marginals

N_PARAMS = 8


class TinyCoupling(nn.Module):
    """One RealNVP coupling; non-linear (tanh) head Linear(1,2)->(log-scale,shift).
    `flip` swaps which coordinate is transformed. 4 params each."""
    def __init__(self, flip):
        super().__init__()
        self.flip = bool(flip)
        self.lin = nn.Linear(1, 2)
        nn.init.zeros_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def _ss(self, a):
        st = self.lin(torch.tanh(a))
        return 1.5 * torch.tanh(st[:, 0:1] / 1.5), st[:, 1:2]

    def _split(self, xy):
        return (xy[:, 1:2], xy[:, 0:1]) if self.flip else (xy[:, 0:1], xy[:, 1:2])

    def _join(self, a, b):
        return torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)

    def forward(self, xy):
        a, b = self._split(xy); s, t = self._ss(a)
        return self._join(a, (b - t) * torch.exp(-s)), -s.squeeze(-1)


class TinyNNFlow(nn.Module):
    """Two tiny couplings + fixed standardisation. 8 trainable parameters."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyCoupling(flip=(i % 2 == 1)) for i in range(2)])
        self.register_buffer("mu", torch.zeros(2, dtype=torch.float64))
        self.register_buffer("sig", torch.ones(2, dtype=torch.float64))

    def set_std(self, z, w):
        self.mu = torch.tensor([float(np.mean(z)), float(np.mean(w))], dtype=torch.float64)
        self.sig = torch.tensor([float(np.std(z)), float(np.std(w))], dtype=torch.float64)

    def log_prob(self, z, w):
        xy = (torch.stack([z, w], 1) - self.mu) / self.sig
        ld = xy.new_zeros(xy.shape[0])
        for L in self.layers:
            xy, d = L(xy); ld = ld + d
        lb = -0.5 * (xy ** 2).sum(1) - math.log(2 * math.pi)
        return lb + ld - torch.log(self.sig).sum()


def run(cfg):
    N = cfg.get("N", 2000)
    inits = cfg.get("inits", [180, 350])
    z_obs, w_obs = C.make_gaussian_data(N=N)

    def mk():
        f = TinyNNFlow().double()
        zb, wb = C.backint(z_obs, w_obs, 300.0, ns=1000)   # fixed standardisation frame
        f.set_std(zb, wb)
        return f

    fits, flows = {}, {}
    for hi in inits:
        flow, hf, hist = C.joint_fit(mk, z_obs, w_obs, hi, lr_flow=5e-3)
        fits[hi] = dict(h=hf, hist=hist)
        flows[hi] = flow
    h_est = float(np.mean([fits[hi]["h"] for hi in inits]))

    # ---- profile likelihood: RE-FIT the 8 NN params at each h (warm sweeps) ----
    def _refit(flow, zb, wb, epochs, lr=5e-3):
        opt = torch.optim.Adam(flow.parameters(), lr=lr)
        zt, wt = torch.tensor(zb), torch.tensor(wb)
        for _ in range(epochs):
            opt.zero_grad(); loss = -flow.log_prob(zt, wt).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0); opt.step()
        with torch.no_grad():
            return float((-flow.log_prob(zt, wt)).mean())

    hgrid = np.linspace(h_est - 90, h_est + 90, 13)
    clouds = [C.backint(z_obs, w_obs, h) for h in hgrid]
    fu = mk(); _refit(fu, *clouds[0], 1500)
    up = N * np.array([_refit(fu, *clouds[i], 400) for i in range(len(hgrid))])
    fd = mk(); _refit(fd, *clouds[-1], 1500)
    dn = N * np.array([_refit(fd, *clouds[i], 400)
                       for i in range(len(hgrid) - 1, -1, -1)])[::-1]
    nll = np.minimum(up, dn); dnll = nll - nll.min()
    noise = float(np.abs(up - dn).max())
    h_min = float(hgrid[dnll.argmin()])
    lo, hi_ = C.cross(hgrid, dnll, 0.5, -1), C.cross(hgrid, dnll, 0.5, +1)
    sig = 0.5 * (hi_ - lo)

    flow0 = flows[inits[0]]
    zb, wb = C.backint(z_obs, w_obs, fits[inits[0]]["h"])
    zg, mz, wg, mw = C.fitted_marginals(flow0, zb, wb)

    return dict(
        id="B", n_params=N_PARAMS, inits=inits, fits=fits, h_est=h_est,
        h_true=C.H_TRUE, z0_sigma=C.Z0_SIGMA, w0_sigma=C.W0_SIGMA,
        prof_h=hgrid, prof_dnll=dnll, noise=noise, h_min=h_min, sig=sig, lo=lo, hi=hi_,
        z0_samples=zb, w0_samples=wb, zg=zg, mz=mz, wg=wg, mw=mw,
    )


def plot(res):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for hi in res["inits"]:
        f = res["fits"][hi]
        ax[0, 0].plot(f["hist"], label=f'init h={hi} -> {f["h"]:.1f}')
    ax[0, 0].axhline(res["h_true"], color="k", ls="--", label=f'truth {res["h_true"]:.0f}')
    ax[0, 0].set_xlabel("epoch"); ax[0, 0].set_ylabel(r"$h_{\rm dm}$ [pc]")
    ax[0, 0].set_title(f"joint ({res['n_params']}-param NN $f_0$)+$h$ recovers ~250")
    ax[0, 0].legend()

    ax[0, 1].plot(res["prof_h"], res["prof_dnll"], "o-", color="C3", ms=4,
                  label="profile $\\Delta$NLL(h)")
    ax[0, 1].axhline(0.5, color="k", ls=":", label=r"$\Delta$NLL=0.5 (1$\sigma$)")
    ax[0, 1].axvspan(res["lo"], res["hi"], color="C2", alpha=0.15,
                     label=fr'{res["h_min"]:.0f}$\pm${res["sig"]:.0f} pc')
    ax[0, 1].axvline(res["h_true"], color="k", ls="--", label=f'truth {res["h_true"]:.0f}')
    ax[0, 1].set_ylim(0, 6); ax[0, 1].set_xlabel(r"$h_{\rm dm}$ [pc]")
    ax[0, 1].set_ylabel(r"$\Delta$NLL$_{\rm total}$ [nats]")
    ax[0, 1].set_title("profile likelihood (re-fit NN at each $h$)")
    ax[0, 1].legend(fontsize=8)

    _plot_marginals(ax[1, 0], ax[1, 1], res)
    ax[1, 0].set_title(r"$z_0$ marginal"); ax[1, 1].set_title(r"$w_0$ marginal")
    fig.suptitle(f"Test B -- tiny NN bijection ({res['n_params']} params):  "
                 f"h = {res['h_min']:.0f} $\\pm$ {res['sig']:.0f} pc", fontsize=13)
    fig.tight_layout()
    return fig
