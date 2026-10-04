"""Test 9 -- flow AS the initial DF f0, optimised jointly with h_dm.

Part A: a full RealNVP flow and h are optimised together by maximum likelihood,
        L(phi,h) = sum_i log f0_flow(T_h(x_i)).  It recovers ~250 but does not
        beat the rigid-Gaussian MLE.
Part B: the profile-likelihood well depth in h for a Gaussian f0 (sharp) vs a
        flexible f0 (flat) -- entropy floors via covariance vs kNN
        (Kozachenko-Leonenko).  Explains why a flexible flow barely constrains h.
"""
import numpy as np
import torch
from math import lgamma
from scipy.special import digamma
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from experiments import common as C
from fitter import ZWNormalizingFlow


def _knnH(X, k=5):
    N, d = X.shape
    dist, _ = cKDTree(X).query(X, k=k + 1)
    c_d = np.pi ** (d / 2) / np.exp(lgamma(d / 2 + 1))
    return digamma(N) - digamma(k) + np.log(c_d) + (d / N) * np.sum(np.log(dist[:, k] + 1e-12))


def run(cfg):
    N = cfg.get("N", 2000)
    inits = cfg.get("inits", [180, 350])
    z_obs, w_obs = C.make_gaussian_data(N=N)

    # ---- Part A: joint (flow=f0) + h maximum likelihood ----
    # Small RealNVP (few couplings, tiny hidden) -- NOT the ~35k-param hidden=64
    # flow -- to match the "small bijection" philosophy of Tests A/B.
    n_couplings, hidden = 4, 6
    n_params = sum(p.numel() for p in ZWNormalizingFlow(n_couplings, hidden).parameters())
    jointA = {}
    for hi in inits:
        def mk(hi=hi):
            f = ZWNormalizingFlow(n_couplings=n_couplings, hidden=hidden).double()
            zi, wi = C.backint(z_obs, w_obs, hi, ns=1000)
            f.set_standardization(zi, wi)
            return f
        _, hf, hist = C.joint_fit(mk, z_obs, w_obs, hi, n_epochs=250, ns=1000,
                                  lr_flow=1e-3, lr_h=2.0)
        jointA[hi] = dict(h=hf, hist=hist)

    # ---- Part B: profile-likelihood well, Gaussian f0 vs flexible f0 ----
    zc, wc = C.backint(z_obs, w_obs, C.H_TRUE)
    SZ, SW = zc.std(), wc.std()
    hgridP = np.array([150, 175, 200, 225, 240, 250, 260, 275, 300, 325, 350, 400], float)
    flex, gau = [], []
    for h in hgridP:
        zb, wb = C.backint(z_obs, w_obs, h)
        X = np.stack([zb / SZ, wb / SW], axis=1)
        flex.append(_knnH(X))
        cov = np.cov(X.T)
        gau.append(0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(cov)))
    flex = np.array(flex) - np.min(flex)
    gau = np.array(gau) - np.min(gau)

    return dict(id="9", inits=inits, jointA=jointA, h_true=C.H_TRUE,
                hgridP=hgridP, flex=flex, gau=gau,
                gau_depth=float(gau.max()), flex_depth=float(flex.max()))


def plot(res):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for hi in res["inits"]:
        ax.plot(res["jointA"][hi]["hist"], label=f'init {hi} -> {res["jointA"][hi]["h"]:.1f}')
    ax.axhline(res["h_true"], color="k", ls="--", label=f'truth {res["h_true"]:.0f}')
    ax.set_xlabel("epoch"); ax.set_ylabel(r"$h_{\rm dm}$ [pc]")
    ax.set_title("Part A: joint (flow=$f_0$) + h recovers ~250\n(ties the Gaussian MLE)")
    ax.legend()

    ax = axes[1]
    ax.plot(res["hgridP"], res["gau"], "o-", color="C0",
            label=f'Gaussian $f_0$  (well {res["gau_depth"]:.2f} nats)')
    ax.plot(res["hgridP"], res["flex"], "s-", color="C3",
            label=f'flexible $f_0$ / flow  (well {res["flex_depth"]:.2f} nats)')
    ax.axvline(res["h_true"], color="k", ls="--")
    ax.set_xlabel(r"$h_{\rm dm}$ [pc]"); ax.set_ylabel(r"profile NLL $-$ min [nats]")
    ax.set_title("Part B: flexible $f_0$ likelihood is ~flat in h\n"
                 "(Liouville: symplectic flow conserves phase-space entropy)")
    ax.legend()
    fig.suptitle("Test 9 -- flow as $f_0$, jointly with h", fontsize=12)
    fig.tight_layout()
    return fig
