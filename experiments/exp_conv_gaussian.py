"""Why is fitted h positively correlated with the initial guess?  (Gaussian data)

Hypothesis: it is UNDER-CONVERGENCE -- the joint optimiser is stopped after a
fixed number of steps before it reaches the minimum, so h stays near where it
started.  Test: sweep the number of optimisation steps and watch the final-vs-init
slope.  If it is under-convergence the slope collapses to 0 as steps grow (fits
reach the true minimum from every init); if it were a genuine flat-likelihood
degeneracy the slope would stay ~1 forever.

Panel A: fitted h vs initial guess, one curve per #epochs (flexible NVP 272p),
         with the rigid analytic-linear flow as a reference (flat at all steps).
Panel B: the h optimisation trajectories (NVP 272p) -- every init converges to
         ~truth once enough steps are taken.
"""
import numpy as np
import matplotlib.pyplot as plt

from experiments import common as C
from fitter import ZWNormalizingFlow
from experiments.exp_A_linear_bijection import DiagonalAffineFlow

INITS = [150.0, 220.0, 290.0, 350.0]
EPOCHS = [100, 400, 1200, 3000]


def _slope(inits, finals):
    return float(np.polyfit(inits, finals, 1)[0])


def run(cfg):
    z, w = C.make_gaussian_data(N=cfg.get("N", 20000), seed_sample=cfg.get("seed", 3))
    zi, wi = C.backint(z, w, 300.0, ns=1000)
    mk_nvp = lambda: (lambda f: (f.set_standardization(zi, wi), f)[1])(ZWNormalizingFlow(4, 6).double())
    mk_aff = lambda: DiagonalAffineFlow((C.Z0_SIGMA * 1.5, C.W0_SIGMA * 1.5)).double()

    # flexible NVP: sweep the step budget
    nvp_finals, traj = {}, {}
    for ep in EPOCHS:
        fn = []
        for hi in INITS:
            _, hf, hist = C.joint_fit(mk_nvp, z, w, hi, n_epochs=ep, ns=1000, lr_flow=1e-3, lr_h=2.0)
            fn.append(hf)
            if ep == EPOCHS[-1]:
                traj[hi] = np.array(hist)
        nvp_finals[ep] = fn

    # rigid analytic: reference at the max step budget only (it converges instantly)
    aff = [C.joint_fit(mk_aff, z, w, hi, n_epochs=EPOCHS[-1], ns=1000, lr_flow=5e-2, lr_h=2.0)[1]
           for hi in INITS]

    return dict(id="CV", inits=INITS, epochs=EPOCHS, h_true=C.H_TRUE,
                nvp_finals=nvp_finals, aff_finals={EPOCHS[-1]: aff},
                nvp_slope={ep: _slope(INITS, nvp_finals[ep]) for ep in EPOCHS},
                aff_slope={EPOCHS[-1]: _slope(INITS, aff)},
                traj={int(k): v for k, v in traj.items()})


def plot(res):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    inits = res["inits"]
    cols = plt.cm.viridis(np.linspace(0.0, 0.82, len(res["epochs"])))

    lo, hi = min(inits), max(inits)
    ax[0].plot([lo, hi], [lo, hi], ":", color="0.6", label="final = init (slope 1)")
    for ep, c in zip(res["epochs"], cols):
        ax[0].plot(inits, res["nvp_finals"][ep], "o-", color=c, lw=1.8,
                   label=f'NVP 272p, {ep} ep (slope {res["nvp_slope"][ep]:+.2f})')
    ax[0].plot(inits, res["aff_finals"][res["epochs"][-1]], "s--", color="C3", lw=1.6,
               label=f'analytic 4p, {res["epochs"][-1]} ep (slope {res["aff_slope"][res["epochs"][-1]]:+.2f})')
    ax[0].axhline(res["h_true"], color="k", ls="-", lw=0.8, label=f'truth {res["h_true"]:.0f}')
    ax[0].set_xlabel("initial guess $h_{\\rm init}$ [pc]")
    ax[0].set_ylabel("fitted $h_{\\rm dm}$ [pc]")
    ax[0].set_title("final-vs-init flattens as #steps grows\n=> the correlation is UNDER-CONVERGENCE")
    ax[0].legend(fontsize=8)

    for hi in sorted(res["traj"]):
        tr = res["traj"][hi]
        ax[1].plot(np.arange(len(tr)), tr, lw=1.3, label=f'init {hi}')
    ax[1].axhline(res["h_true"], color="k", ls="--", lw=1.2, label=f'truth {res["h_true"]:.0f}')
    ax[1].set_xlabel("optimisation step (epoch)")
    ax[1].set_ylabel("$h_{\\rm dm}$ [pc]")
    ax[1].set_title("NVP 272p trajectories: every init converges to ~truth\n"
                    "(given enough steps)")
    ax[1].legend(fontsize=8)

    fig.suptitle("Why fitted $h$ tracks the initial guess: optimiser step budget", fontsize=13)
    fig.tight_layout()
    return fig
