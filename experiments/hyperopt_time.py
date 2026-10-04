"""Hyper-parameter search for the joint (h_dm, Delta t) fit.

WHY.  In the time-parameter notebook the *profile* likelihood put the minimum at
the truth (250 pc, 400 Myr) but the *joint gradient fit* landed wherever it
started: (220,300) -> (243,291), (290,520) -> (326,503), (240,250) -> (277,233).
That is not slow convergence, it is the shape of the objective:

  * In Delta t the exact profiled NLL is a COMB.  Back-integrating by the wrong
    total time leaves the cloud partly wound up, and the leftover width breathes
    with the ~41 Myr half-period of the non-equilibrium blob -> ~16 local minima
    between 100 and 700 Myr separated by 100-300 nat barriers (N = 20000), with
    a single razor-sharp global tooth at 400 Myr (1-sigma half width ~0.3 Myr,
    smooth only within ~+-10 Myr).  No learning rate lets a local step cross a
    200-nat barrier.
  * In h_dm the same objective is a very SHALLOW parabola (~3 nats at 70 pc from
    the truth), so h needs many small, well-scaled steps -- the classic
    under-convergence that makes the fitted h track its initial guess.

So the fit needs a global stage for Delta t and a well-tuned local stage for h.
This module benchmarks candidate fitters on a fixed spread of starting points
and reports which hyper-parameters actually buy convergence.

Score (no knowledge of the truth): the mean, over starts, of the FULL-SAMPLE
closed-form profiled NLL at the landing point -- i.e. how well the optimiser did
the job it was asked to do.  Recovery vs the truth is reported alongside as the
validation, never as the thing being optimised.

Run:
    python -m experiments.hyperopt_time stage1      # local-only Adam knobs
    python -m experiments.hyperopt_time stage2      # global machinery
    python -m experiments.hyperopt_time stage3      # refine the winner
    python -m experiments.hyperopt_time stage4      # 1-D vs 2-D de-aliasing sweep
    python -m experiments.hyperopt_time final       # validate on 16 starts x 3 seeds
"""
import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), os.pardir, "results")

# starting points (h_dm [pc], Delta t [Myr]) spanning the prior box.  The truth
# is (250, 400); every other start is deliberately in the wrong comb tooth.
INITS = [(150, 250), (200, 300), (250, 400), (300, 500),
         (350, 600), (400, 220), (180, 550), (320, 350)]

INITS_BIG = INITS + [(250, 210), (250, 690), (120, 430), (500, 380),
                     (220, 460), (280, 340), (160, 620), (380, 240)]

# cheap 4-point subset used while sweeping (two far starts, one at the truth,
# one far in h but near in t) -- the full 16 are kept for the final validation.
INITS_SWEEP = [(150, 250), (350, 600), (250, 400), (400, 220)]

# the settings the notebook used before this study (fixed step COUNT, no global
# stage, no polish) -- the baseline every candidate has to beat.
BASELINE = dict(n_epochs=300, n_sub=2000, lr_flow=2e-2, lr_h=2.0, lr_dt=10.0,
                warm=150, sched="cosine", scan=False, jitter0=0.0, polish=0,
                legacy_fixed_count=True)


# --------------------------------------------------------------------------
def _one_fit(args):
    """Run one fit from one start; return its landing + scores.  Top level so it
    is picklable for the process pool."""
    cfg, (h_init, dt_init), seed, N = args
    import torch
    torch.set_num_threads(1)
    from experiments import common as C

    z, w = C.make_gaussian_data(N=N, seed_sample=3)
    kw = dict(cfg)
    legacy = kw.pop("legacy_fixed_count", False)
    label = kw.pop("label", "")
    t0 = time.time()
    if legacy:
        # reproduce the OLD fitter exactly: fixed step count, dt = Delta t / ns
        flow, hf, dtf, hh, dd, ll = _legacy_fit(C, z, w, h_init, dt_init,
                                                seed=seed, **kw)
    else:
        flow, hf, dtf, hh, dd, ll = C.joint_fit_h_dt(
            C._diag_affine_maker, z, w, h_init, dt_init, seed=seed, **kw)
    wall = time.time() - t0
    # honest score: full-sample closed-form profiled NLL at the landing point
    nll = C._gauss_profile_nll(z, w, hf, dtf)
    nll0 = C._gauss_profile_nll(z, w, C.H_TRUE, C.DT_TRUE)
    return dict(label=label, h_init=h_init, dt_init=dt_init, seed=seed,
                h_fit=hf, dt_fit=dtf, dnll=nll - nll0, wall=wall,
                dh=hf - C.H_TRUE, ddt=dtf - C.DT_TRUE)


def _legacy_fit(C, z, w, h_init, dt_init, n_epochs=300, n_sub=2000, ns=800,
                lr_flow=2e-2, lr_h=2.0, lr_dt=10.0, warm=150, sched="cosine",
                seed=0, **_ignored):
    """The pre-study fitter: fixed step COUNT ns, free timestep dt = Delta t/ns."""
    import torch
    import torch.nn as nn
    from fitter import torch_back_integrate
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    flow = C._diag_affine_maker()
    h = nn.Parameter(torch.tensor(float(h_init), dtype=torch.float64))
    dt = nn.Parameter(torch.tensor(float(dt_init), dtype=torch.float64))
    zt, wt = torch.tensor(z), torch.tensor(w)

    def _cloud(idx):
        return torch_back_integrate(zt[idx], wt[idx], dt, h, C.sim, n_steps=ns)

    optf = torch.optim.Adam(flow.parameters(), lr=lr_flow)
    for _ in range(warm):
        idx = rng.choice(z.size, size=min(n_sub, z.size), replace=False)
        with torch.no_grad():
            zb, wb = _cloud(idx)
        optf.zero_grad(); (-flow.log_prob(zb, wb).mean()).backward(); optf.step()

    opt = torch.optim.Adam([{'params': flow.parameters(), 'lr': lr_flow},
                            {'params': [h], 'lr': lr_h},
                            {'params': [dt], 'lr': lr_dt}])
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
           if sched == "cosine" else None)
    hh, dd, ll = [], [], []
    for _ in range(n_epochs):
        idx = rng.choice(z.size, size=min(n_sub, z.size), replace=False)
        zb, wb = _cloud(idx)
        loss = -flow.log_prob(zb, wb).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([h], 30.0)
        torch.nn.utils.clip_grad_norm_([dt], 30.0)
        opt.step()
        if sch is not None:
            sch.step()
        with torch.no_grad():
            h.clamp_(50.0, 800.0); dt.clamp_(C.DT_MIN, C.DT_MAX)
        hh.append(h.item()); dd.append(dt.item()); ll.append(loss.item())
    return flow, h.item(), dt.item(), hh, dd, ll


# --------------------------------------------------------------------------
def evaluate(cfgs, inits=None, seeds=(0,), N=20000, workers=10, verbose=True):
    """Run every (cfg, init, seed) and aggregate per-cfg scores."""
    inits = INITS_SWEEP if inits is None else inits
    jobs = [(cfg, ini, sd, N) for cfg in cfgs for ini in inits for sd in seeds]
    t0 = time.time()
    with Pool(workers) as p:
        out = p.map(_one_fit, jobs, chunksize=1)
    rows = {}
    for r in out:
        rows.setdefault(r["label"], []).append(r)
    table = []
    for cfg in cfgs:
        rs = rows[cfg["label"]]
        dn = np.array([r["dnll"] for r in rs])
        dh = np.array([r["dh"] for r in rs])
        dd = np.array([r["ddt"] for r in rs])
        ok = (np.abs(dh) < 10.0) & (np.abs(dd) < 2.0)
        table.append(dict(
            label=cfg["label"], mean_dnll=float(dn.mean()), max_dnll=float(dn.max()),
            med_abs_dh=float(np.median(np.abs(dh))), max_abs_dh=float(np.abs(dh).max()),
            med_abs_ddt=float(np.median(np.abs(dd))), max_abs_ddt=float(np.abs(dd).max()),
            spread_h=float(np.std([r["h_fit"] for r in rs])),
            spread_dt=float(np.std([r["dt_fit"] for r in rs])),
            frac_ok=float(ok.mean()), wall=float(np.mean([r["wall"] for r in rs])),
            runs=rs))
    table.sort(key=lambda t: t["mean_dnll"])
    if verbose:
        print_table(table)
        print(f"[{len(jobs)} fits in {time.time()-t0:.0f} s wall]")
    return table


def print_table(table):
    hdr = (f"{'config':<34}{'mean dNLL':>11}{'max dNLL':>10}{'|dh| med':>10}"
           f"{'|dh| max':>10}{'|ddt| med':>11}{'|ddt| max':>10}"
           f"{'sd(h)':>8}{'sd(dt)':>8}{'ok':>7}{'s/fit':>7}")
    print(hdr); print("-" * len(hdr))
    for t in table:
        print(f"{t['label']:<34}{t['mean_dnll']:>11.1f}{t['max_dnll']:>10.1f}"
              f"{t['med_abs_dh']:>10.1f}{t['max_abs_dh']:>10.1f}"
              f"{t['med_abs_ddt']:>11.2f}{t['max_abs_ddt']:>10.2f}"
              f"{t['spread_h']:>8.1f}{t['spread_dt']:>8.1f}"
              f"{t['frac_ok']*100:>6.0f}%{t['wall']:>7.1f}")


def _save(name, table):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"hyperopt_{name}.json")
    slim = [{k: v for k, v in t.items() if k != "runs"} for t in table]
    runs = {t["label"]: [{k: v for k, v in r.items() if k != "label"}
                         for r in t["runs"]] for t in table}
    with open(path, "w") as f:
        json.dump(dict(table=slim, runs=runs), f, indent=1)
    print("saved", path)


# --------------------------------------------------------------------------
def stage1(workers=10):
    """Local-only Adam: can ANY (lr, #epochs, batch, schedule) reach the truth
    without a global stage?  Includes the legacy fixed-step-count baseline."""
    cfgs = [dict(BASELINE, label="LEGACY fixed-count (old notebook)")]
    for lr_h, lr_dt in itertools.product((1.0, 3.0, 10.0), (2.0, 10.0, 30.0)):
        cfgs.append(dict(n_epochs=400, n_sub=2000, lr_flow=2e-2, lr_h=lr_h,
                         lr_dt=lr_dt, warm=150, sched="cosine", scan=False,
                         jitter0=0.0, polish=0,
                         label=f"local lr_h={lr_h} lr_dt={lr_dt}"))
    # longer / no-schedule / bigger-batch variants of the mid setting
    cfgs += [
        dict(n_epochs=1500, n_sub=2000, lr_flow=2e-2, lr_h=3.0, lr_dt=10.0,
             warm=150, sched="cosine", scan=False, jitter0=0.0, polish=0,
             label="local 1500 epochs"),
        dict(n_epochs=400, n_sub=2000, lr_flow=2e-2, lr_h=3.0, lr_dt=10.0,
             warm=150, sched="none", scan=False, jitter0=0.0, polish=0,
             label="local no schedule"),
        dict(n_epochs=400, n_sub=20000, lr_flow=2e-2, lr_h=3.0, lr_dt=10.0,
             warm=150, sched="cosine", scan=False, jitter0=0.0, polish=0,
             label="local full batch"),
    ]
    t = evaluate(cfgs, workers=workers); _save("stage1", t); return t


def stage2(workers=10):
    """Global machinery: randomised-smoothing annealing vs the de-aliasing scan."""
    cfgs = []
    for j0 in (20.0, 40.0, 80.0):
        for nj in (2, 4):
            cfgs.append(dict(n_epochs=600, n_sub=2000, lr_flow=2e-2, lr_h=3.0,
                             lr_dt=10.0, warm=150, sched="cosine", scan=False,
                             jitter0=j0, jitter_end=0.5, n_jit=nj, polish=150,
                             label=f"anneal jitter {j0:.0f}->0.5 x{nj}"))
    for st in (1.0, 2.0, 4.0, 8.0, 16.0):
        cfgs.append(dict(n_epochs=400, n_sub=2000, lr_flow=2e-2, lr_h=3.0,
                         lr_dt=4.0, warm=150, sched="cosine", scan=True,
                         scan_step=st, scan_sub=4000, jitter0=0.0, polish=150,
                         label=f"scan step={st:.0f} Myr"))
    cfgs.append(dict(n_epochs=400, n_sub=2000, lr_flow=2e-2, lr_h=3.0, lr_dt=4.0,
                     warm=150, sched="cosine", scan=True, scan_step=2.0,
                     scan_sub=1000, jitter0=0.0, polish=150,
                     label="scan step=2 sub=1000"))
    cfgs.append(dict(n_epochs=400, n_sub=2000, lr_flow=2e-2, lr_h=3.0, lr_dt=4.0,
                     warm=150, sched="cosine", scan=True, scan_step=2.0,
                     scan_sub=20000, jitter0=0.0, polish=150,
                     label="scan step=2 sub=all"))
    cfgs.append(dict(n_epochs=600, n_sub=2000, lr_flow=2e-2, lr_h=3.0, lr_dt=10.0,
                     warm=150, sched="cosine", scan=True, scan_step=2.0,
                     jitter0=40.0, jitter_end=0.5, n_jit=2, polish=150,
                     label="scan + anneal jitter"))
    t = evaluate(cfgs, workers=workers); _save("stage2", t); return t


def stage3(workers=10):
    """With Delta t solved by the sweep, fix the OTHER half: h_dm.  Its gradient
    is ~0.05 nats/pc for the whole 20000-star sample (2.6e-6 per star), far below
    the fresh-minibatch gradient noise, so h random-walks instead of descending.
    The candidates are all ways of making the objective deterministic (full batch
    or a frozen minibatch) or of giving h more travel (lr_h, epochs, polish)."""
    base = dict(lr_flow=2e-2, warm=150, sched="cosine", scan=True,
                scan_step=2.0, scan_sub=4000, jitter0=0.0, lr_dt=4.0)
    cfgs = []
    # (a) batch regime -- the suspected root cause
    for sub, rs in ((2000, True), (2000, False), (8000, True), (20000, True)):
        cfgs.append(dict(base, n_epochs=400, n_sub=sub, resample=rs, lr_h=3.0,
                         polish=150,
                         label=f"n_sub={sub}{'' if rs else ' frozen batch'}"))
    # (b) how far h is allowed to travel
    for lr_h in (1.0, 3.0, 8.0, 20.0):
        cfgs.append(dict(base, n_epochs=400, n_sub=20000, lr_h=lr_h, polish=150,
                         label=f"full batch lr_h={lr_h}"))
    for ep in (200, 800):
        cfgs.append(dict(base, n_epochs=ep, n_sub=20000, lr_h=3.0, polish=150,
                         label=f"full batch epochs={ep}"))
    # (c) the polish stage (full batch by construction)
    for pol, plr in itertools.product((0, 150, 400), (0.25, 1.0)):
        cfgs.append(dict(base, n_epochs=400, n_sub=2000, lr_h=3.0, polish=pol,
                         polish_lr=plr, label=f"minibatch + polish={pol} x{plr}"))
    t = evaluate(cfgs, workers=workers); _save("stage3", t); return t


def stage4(workers=10):
    """The residual failure the first `final` run exposed: 2 of 48 tuned runs (both
    from h_init >= 400) landed on the WRONG comb tooth and ran h to its clamp.
    Cause: which tooth is deepest depends on h, so a sweep of Delta t at the
    caller's h_init alone can pick the wrong one when h_init is far off -- at
    h = 500 pc even the FULL-sample sweep prefers 442 Myr over 400.  Fix: sweep the
    coarse (h ladder) x (Delta t) grid instead.  Scored on the starts that failed
    plus controls, over 3 seeds."""
    hard = [(400, 220), (500, 380), (120, 430), (250, 400)]
    cfgs = [dict(BEST, label="2-D sweep (h ladder x dt)"),
            dict(BEST, scan_h=None, label="1-D sweep (dt at h_init)")]
    t = evaluate(cfgs, inits=hard, seeds=(0, 1, 2), workers=workers)
    _save("stage4", t)
    for row in t:
        print(f"\n--- {row['label']} ---")
        for r in sorted(row["runs"], key=lambda r: (r["h_init"], r["seed"])):
            print(f"  init (h={r['h_init']:>3}, dt={r['dt_init']:>3}) seed {r['seed']} "
                  f"-> h={r['h_fit']:7.2f}  dt={r['dt_fit']:7.2f}   dNLL={r['dnll']:+9.2f}")
    return t


# Adopted defaults (now the defaults of C.joint_fit_h_dt too).  lr_h = 20 was
# statistically indistinguishable (same minimum to 4 decimals in NLL, spread
# 0.09 vs 0.41 pc); 8.0 is kept as the more conservative step -- still an order of
# magnitude below the width of the h well.
BEST = dict(n_epochs=400, n_sub=20000, lr_flow=2e-2, lr_h=8.0, lr_dt=4.0,
            warm=150, sched="cosine", scan=True, scan_step=4.0, scan_sub=4000,
            jitter0=0.0, polish=150)


def final(workers=10):
    """Validate the adopted defaults on 16 starts x 3 seeds, against the legacy
    fitter on the same starts."""
    cfgs = [dict(BEST, label="TUNED (adopted defaults)"),
            dict(BASELINE, label="LEGACY fixed-count (old notebook)")]
    t = evaluate(cfgs, inits=INITS_BIG, seeds=(0, 1, 2), workers=workers)
    _save("final", t)
    for row in t:
        print(f"\n--- {row['label']} : every landing ---")
        for r in sorted(row["runs"], key=lambda r: (r["h_init"], r["dt_init"])):
            print(f"  init (h={r['h_init']:>3}, dt={r['dt_init']:>3}) seed {r['seed']} "
                  f"-> h={r['h_fit']:7.2f}  dt={r['dt_fit']:7.2f}   dNLL={r['dnll']:+9.2f}")
    return t


def load(name):
    """Load a saved stage's table (list of dicts) + per-run landings."""
    with open(os.path.join(RESULTS, f"hyperopt_{name}.json")) as f:
        d = json.load(f)
    return d["table"], d["runs"]


def show(name="stage1"):
    """Print a saved stage's ranking table."""
    table, _ = load(name)
    print_table(table)
    return table


def plot_final(name="final", title=None):
    """The 'regardless of the initial guess' figure: where every start LANDS,
    tuned fitter vs the old local-only one, in both parameters."""
    import matplotlib.pyplot as plt
    _, runs = load(name)
    keys = [k for k in runs if k.startswith("TUNED")] + \
           [k for k in runs if k.startswith("LEGACY")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    cols = {keys[0]: "C2", keys[1]: "C3"}
    for k in keys:
        rs = runs[k]
        for j, (par, truth, lab) in enumerate(
                [("h_fit", 250.0, r"$h_{\rm dm}$ [pc]"),
                 ("dt_fit", 400.0, r"$\Delta t$ [Myr]")]):
            x = [r["h_init"] if j == 0 else r["dt_init"] for r in rs]
            y = [r[par] for r in rs]
            ax[j].plot(x, y, "o", color=cols[k], ms=6, alpha=0.75,
                       label=k.split(" (")[0])
            ax[j].axhline(truth, color="k", ls="--", lw=1.0)
            ax[j].set_xlabel("initial guess " + lab)
            ax[j].set_ylabel("fitted " + lab)
    lo, hi = ax[0].get_xlim()
    ax[0].plot([lo, hi], [lo, hi], color="0.7", lw=0.8, ls=":")
    lo, hi = ax[1].get_xlim()
    ax[1].plot([lo, hi], [lo, hi], color="0.7", lw=0.8, ls=":",
               label="fit = initial guess")
    for a in ax:
        h_, l_ = a.get_legend_handles_labels()
        seen = dict(zip(l_, h_))
        a.legend(seen.values(), seen.keys(), fontsize=8)
        a.grid(alpha=0.25)
    ax[0].set_title(r"$h_{\rm dm}$: fitted vs initial guess")
    ax[1].set_title(r"$\Delta t$: fitted vs initial guess")
    fig.suptitle(title or "landing vs starting point, 16 starts x 3 seeds "
                          "(dashed = truth, dotted = 'fit just tracks the init')",
                 fontsize=12)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "stage1"
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    {"stage1": stage1, "stage2": stage2, "stage3": stage3,
     "stage4": stage4, "final": final}[which](nw)
