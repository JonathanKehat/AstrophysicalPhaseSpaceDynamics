"""Cluster-scale version of the (h_dm, Delta t) hyper-parameter study.

`experiments/hyperopt_time.py` is the LAPTOP study: four hand-built stages, a
4-point start subset, one seed, ~10 workers, everything held in one process and
written at the end.  It answered the qualitative question (a global de-aliasing
stage is necessary and sufficient; h needs a deterministic full-batch objective)
and produced `hyperopt_time.BEST`.

This module is the same experiment sized for a real machine:

  * a RANDOM SEARCH over the full fitter hyper-parameter space instead of four
    hand-picked axes, so the adopted defaults are checked against configurations
    nobody thought to try;
  * every (config, start, seed, N, dt_step) combination is an independent unit of
    work with a stable content-addressed id, written to its own small JSON the
    moment it finishes -- so the study is RESUMABLE and SHARDABLE.  A job already
    on disk is never recomputed, whatever killed the previous attempt;
  * shards are assigned round-robin from a job list that every task rebuilds
    identically from a fixed seed, so a SLURM array needs no coordination, no
    shared scheduler and no head node -- only a shared filesystem;
  * the expensive axes the laptop cannot touch are first-class: sample size N up
    to 5e5 stars, several data realisations, and the pinned timestep dt_step
    itself (the one thing the new parametrisation ASSUMES is small enough).

Plans
-----
  smoke     ~5 min, 1 core   -- correctness check, tiny N, tiny epochs.  Run this
                               before submitting anything.
  search    the random search over fitter knobs (the actual tuning).
  validate  the adopted config over many starts x seeds x N (the "lands on the
            minimum regardless of the initial guess" evidence, at scale).
  robust    does the answer move with the pinned timestep or with N?  This is the
            check that dt_step = 0.5 Myr is small enough and that the recovered
            (h, Delta t) is a property of the likelihood, not of the integrator.

Usage
-----
    # always first, on the login node or a laptop
    python -m experiments.hyperopt_cluster smoke

    # how big is it?
    python -m experiments.hyperopt_cluster plan search
    python -m experiments.hyperopt_cluster plan validate

    # one shard (this is what the array task runs)
    python -m experiments.hyperopt_cluster run search --shard 3 --n-shards 64 \
        --workers 8

    # everything on one fat node instead of an array
    python -m experiments.hyperopt_cluster run search --workers 64

    # collect + rank (safe to run while jobs are still landing)
    python -m experiments.hyperopt_cluster merge search
    python -m experiments.hyperopt_cluster report search

SLURM template: `experiments/slurm/hyperopt_array.sbatch`.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
# One directory of per-job JSONs per plan.  Override with $HYPEROPT_SCRATCH when
# the cluster wants the churn on fast scratch rather than on the project share.
SCRATCH = os.environ.get("HYPEROPT_SCRATCH", os.path.join(RESULTS, "hyperopt_jobs"))

from experiments.hyperopt_time import BASELINE, BEST, INITS_BIG, INITS_SWEEP


# ===========================================================================
#  The search space
# ===========================================================================
# Every entry is a knob of `common.joint_fit_h_dt`, grouped by the stage it
# controls.  The laptop study varied the marked ones one axis at a time; the
# random search varies all of them together, which is the point -- e.g. a small
# scan_sub may be fine at a fine scan_step and fatal at a coarse one, and that
# interaction is invisible to a one-axis-at-a-time sweep.
SPACE = {
    # ---- stage 0: the global de-aliasing sweep ---------------------------
    # The comb teeth are ~20 Myr apart, so the sweep grid has to resolve them;
    # `scan_sub` is how many stars each grid point is scored on, and `scan_h`
    # how many h values the (h x Delta t) grid carries (which tooth is deepest
    # drifts by ~-0.05 Myr/pc, so a 1-D sweep at a wrong h can pick wrongly).
    "scan":        [True, False],
    "scan_step":   [1.0, 2.0, 4.0, 8.0, 16.0],          # Myr
    "scan_sub":    [1000, 2000, 4000, 8000, 20000],      # stars
    "scan_refine": [1, 4, 8],
    "scan_h":      ["none", "coarse5", "mid9", "fine13"],

    # ---- stage 1: the joint Adam descent ---------------------------------
    "n_epochs":    [200, 400, 800, 1500],
    "n_sub":       [2000, 8000, "full"],  # "full" = the whole sample, whatever N
    "resample":    [True, False],         # fresh minibatch each epoch, or frozen
    "lr_h":        [1.0, 3.0, 8.0, 20.0, 50.0],    # pc / epoch
    "lr_dt":       [1.0, 4.0, 10.0, 30.0],         # Myr / epoch
    "lr_flow":     [5e-3, 2e-2, 5e-2],
    "sched":       ["cosine", "none"],
    "warm":        [50, 150, 400],        # flow-only epochs before (h, N) unfreeze
    "grad_clip":   [10.0, 30.0, 100.0],

    # ---- stage 2: the full-batch polish ----------------------------------
    "polish":      [0, 150, 400, 1000],
    "polish_lr":   [0.1, 0.25, 1.0],      # multiplier on the stage-1 rates
}

# `scan_h` is a name in the job spec (JSON-serialisable and readable in the
# tables) and a ladder of h values in the fitter.
H_LADDERS = {
    "none":    None,                                   # 1-D sweep at h_init
    "coarse5": (120.0, 220.0, 320.0, 420.0, 520.0),    # what BEST uses
    "mid9":    tuple(np.linspace(100.0, 580.0, 9)),
    "fine13":  tuple(np.linspace(100.0, 640.0, 13)),
}

SEARCH_SEED = 20260922      # fixes the sampled configs -> every shard agrees


def _sample_cfgs(n, rng):
    """`n` random points in SPACE, de-duplicated, with the dead branches pruned
    (there is no point sampling scan_step/scan_sub when scan is off -- that would
    spend most of the budget re-running the same no-sweep fit under different
    labels)."""
    out, seen = [], set()
    while len(out) < n:
        cfg = {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}
        if not cfg["scan"]:
            for dead in ("scan_step", "scan_sub", "scan_refine", "scan_h"):
                cfg[dead] = SPACE[dead][0]
        if cfg["polish"] == 0:
            cfg["polish_lr"] = SPACE["polish_lr"][0]
        key = json.dumps(cfg, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        cfg["label"] = f"rs{len(out):04d}"
        out.append(cfg)
    return out


def _anchors():
    """Reference points carried through every plan so the random search is
    scored against something meaningful: the adopted defaults, the pre-study
    fitter, and the two single-change ablations that matter."""
    return [
        dict(BEST, label="BEST (adopted)"),
        dict(BASELINE, label="LEGACY fixed-count"),
        dict(BEST, scan=False, label="BEST minus sweep"),
        dict(BEST, n_sub=2000, resample=True, label="BEST minus full batch"),
        dict(BEST, polish=0, label="BEST minus polish"),
        dict(BEST, scan_h="none", label="BEST with 1-D sweep"),
    ]


# ===========================================================================
#  Plans -- a plan is just a job list
# ===========================================================================
def build_jobs(plan):
    """Deterministic job list for `plan`.  Called independently by every array
    task; it must return exactly the same list everywhere, so nothing in here may
    depend on the environment, the wall clock or the filesystem."""
    rng = np.random.default_rng(SEARCH_SEED)

    if plan == "smoke":
        cfgs = [dict(BEST, n_epochs=40, warm=20, polish=20, n_sub=4000,
                     scan_step=4.0, scan_sub=2000, label="BEST (short)"),
                dict(BEST, n_epochs=40, warm=20, polish=20, n_sub=4000,
                     scan=False, label="no sweep (short)")]
        return _expand(cfgs, inits=[(150, 250), (350, 600)], seeds=(0,),
                       Ns=(4000,), data_seeds=(3,), dt_steps=(0.5,))

    if plan == "search":
        # The tuning run.  160 configs x 4 starts x 2 seeds = 1280 fits at
        # N = 20000.  The 4-start subset is deliberate: a config that cannot
        # handle (400, 220) and (350, 600) is already disqualified, and the
        # budget is better spent on more configs than on more starts.
        cfgs = _anchors() + _sample_cfgs(160, rng)
        return _expand(cfgs, inits=INITS_SWEEP, seeds=(0, 1), Ns=(20000,),
                       data_seeds=(3,), dt_steps=(0.5,))

    if plan == "validate":
        # The evidence figure, at scale: the adopted config (plus the ablations,
        # so the claim "each stage is load-bearing" is made on the same starts)
        # over the full 16-start spread, 5 optimiser seeds, 3 data realisations
        # and two sample sizes.
        # 3 optimiser seeds is enough to show determinism (the tuned config's
        # seed-to-seed spread is 0.06 pc); the budget goes into the 3 DATA
        # realisations instead, which is the axis that actually carries a
        # scientific claim.
        cfgs = _anchors()
        return _expand(cfgs, inits=INITS_BIG, seeds=(0, 1, 2),
                       Ns=(20000, 100000), data_seeds=(3, 11, 29),
                       dt_steps=(0.5,))

    if plan == "robust":
        # Two independent questions, deliberately NOT crossed.
        #   (a) is the answer a property of the likelihood or of the integrator?
        #       dt_step is the one thing the new parametrisation ASSUMES is small
        #       enough, so it has to be shown not to matter: 1.0 Myr is
        #       DT_STEP_MAX, 0.125 Myr is 4x finer than the adopted 0.5.
        #   (b) how do the recovered values and their errors scale with N?
        # Crossing them would be wasteful and unbalanced: cost and memory go as
        # N / dt_step, so (N = 5e5, dt_step = 0.125) alone is a 68 GB, ~17 h fit
        # that answers neither question better than its two cheap projections.
        cfgs = [dict(BEST, label="BEST (adopted)")]
        jobs = _expand(cfgs, inits=INITS_SWEEP, seeds=(0, 1, 2), Ns=(20000,),
                       data_seeds=(3, 11), dt_steps=(1.0, 0.5, 0.25, 0.125))
        jobs += _expand(cfgs, inits=INITS_SWEEP, seeds=(0, 1, 2),
                        Ns=(20000, 100000, 500000), data_seeds=(3, 11),
                        dt_steps=(0.5,))
        seen, out = set(), []          # the two blocks share (N=2e4, dt=0.5)
        for j in jobs:
            if j["jid"] not in seen:
                seen.add(j["jid"]); out.append(j)
        return out

    raise SystemExit(f"unknown plan {plan!r} "
                     f"(smoke | search | validate | robust)")


def _expand(cfgs, inits, seeds, Ns, data_seeds, dt_steps):
    """Cartesian product -> job list.  Ordered largest-N-first so that on a
    partially-filled array the expensive jobs start earliest and the cheap ones
    backfill; the round-robin shard assignment then spreads cost evenly."""
    jobs = []
    for N in sorted(Ns, reverse=True):
        for cfg in cfgs:
            for ini in inits:
                for sd in seeds:
                    for ds in data_seeds:
                        for dts in dt_steps:
                            jobs.append(dict(cfg=cfg, h_init=ini[0], dt_init=ini[1],
                                             seed=int(sd), N=int(N),
                                             data_seed=int(ds), dt_step=float(dts)))
    for j in jobs:
        j["jid"] = _job_id(j)
    return jobs


def _job_id(job):
    """Content hash of everything that affects the answer.  Identity is the
    CONTENT, not the position in the list, so adding configs to a plan later does
    not invalidate (or silently reuse) any result already on disk."""
    payload = {k: job[k] for k in ("cfg", "h_init", "dt_init", "seed", "N",
                                   "data_seed", "dt_step")}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


# ===========================================================================
#  Running
# ===========================================================================
def _run_one(job):
    """One fit, in a worker process.  Returns the landing + its honest score.

    Never raises: a configuration that diverges or blows up is a legitimate
    (bad) outcome of a hyper-parameter search, and on a 1000-job array it must
    not take the shard down with it."""
    import torch
    torch.set_num_threads(int(os.environ.get("HYPEROPT_TORCH_THREADS", "1")))
    from experiments import common as C
    from experiments.hyperopt_time import _legacy_fit

    cfg = dict(job["cfg"])
    label = cfg.pop("label", "")
    legacy = cfg.pop("legacy_fixed_count", False)
    if legacy:
        cfg.pop("scan_h", None)          # _legacy_fit has no sweep stage
    elif "scan_h" in cfg:
        cfg["scan_h"] = H_LADDERS[cfg["scan_h"]]   # name -> ladder of h values
    # FULL BATCH IS THE POINT, so it has to follow N.  The laptop study ran only
    # at N = 20000, where `n_sub = 20000` meant "the whole sample"; carried
    # literally to N = 100000 that same number is a 20% minibatch, which
    # reintroduces exactly the gradient noise that full batch was adopted to
    # remove -- the winning config would quietly stop being the winning config.
    if cfg.get("n_sub") in ("full", 20000):
        cfg["n_sub"] = job["N"]
    dt_step = job["dt_step"]

    t0 = time.time()
    rec = dict(jid=job["jid"], label=label, h_init=job["h_init"],
               dt_init=job["dt_init"], seed=job["seed"], N=job["N"],
               data_seed=job["data_seed"], dt_step=dt_step)
    try:
        z, w = C.make_gaussian_data(N=job["N"], seed_sample=job["data_seed"])
        if legacy:
            _, hf, dtf, *_ = _legacy_fit(C, z, w, job["h_init"], job["dt_init"],
                                         seed=job["seed"], **cfg)
        else:
            _, hf, dtf, *_ = C.joint_fit_h_dt(
                C._diag_affine_maker, z, w, job["h_init"], job["dt_init"],
                seed=job["seed"], dt_step=dt_step, **cfg)
        # The score, with no knowledge of the truth: the full-sample closed-form
        # profiled NLL at the landing point, relative to its value at the truth.
        # It measures how well the optimiser did the job it was given.  dh/ddt
        # are recorded alongside as validation only -- never optimised against.
        nll = C._gauss_profile_nll(z, w, hf, dtf, dt_step=dt_step)
        nll0 = C._gauss_profile_nll(z, w, C.H_TRUE, C.DT_TRUE, dt_step=dt_step)
        rec.update(h_fit=float(hf), dt_fit=float(dtf), dnll=float(nll - nll0),
                   dh=float(hf - C.H_TRUE), ddt=float(dtf - C.DT_TRUE), ok=True)
    except Exception as exc:                       # noqa: BLE001 -- see docstring
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    rec["wall"] = time.time() - t0
    return rec


def _write(out_dir, rec):
    """Atomic per-job write: a torn JSON from a job killed mid-write would be
    indistinguishable from a real result at merge time."""
    tmp = os.path.join(out_dir, f".{rec['jid']}.tmp")
    with open(tmp, "w") as f:
        json.dump(rec, f)
    os.replace(tmp, os.path.join(out_dir, f"{rec['jid']}.json"))


def run(plan, shard=0, n_shards=1, workers=1, out_dir=None, limit=None,
        redo=False):
    """Run this task's slice of `plan`.  Jobs already on disk are skipped, so a
    requeued / preempted / extended array simply picks up where it stopped."""
    out_dir = out_dir or os.path.join(SCRATCH, plan)
    os.makedirs(out_dir, exist_ok=True)
    jobs = build_jobs(plan)
    mine = [j for i, j in enumerate(jobs) if i % n_shards == shard]
    if not redo:
        mine = [j for j in mine
                if not os.path.exists(os.path.join(out_dir, f"{j['jid']}.json"))]
    if limit:
        mine = mine[:limit]
    print(f"[{plan}] shard {shard}/{n_shards}: {len(mine)} jobs to run "
          f"({len(jobs)} in the plan), {workers} workers -> {out_dir}",
          flush=True)
    if not mine:
        return []

    t0 = time.time()
    done = []
    if workers > 1:
        with Pool(workers) as p:
            for k, rec in enumerate(p.imap_unordered(_run_one, mine, chunksize=1), 1):
                _write(out_dir, rec)
                done.append(rec)
                print(f"  [{k}/{len(mine)}] {rec['label']:<22} "
                      f"init=({rec['h_init']},{rec['dt_init']}) "
                      + (f"-> h={rec['h_fit']:7.2f} dt={rec['dt_fit']:7.2f} "
                         f"dNLL={rec['dnll']:+10.2f}" if rec["ok"]
                         else f"FAILED {rec['error'][:60]}")
                      + f"  [{rec['wall']:.0f}s]", flush=True)
    else:
        for k, job in enumerate(mine, 1):
            rec = _run_one(job)
            _write(out_dir, rec)
            done.append(rec)
            print(f"  [{k}/{len(mine)}] {rec['label']} "
                  + (f"-> h={rec['h_fit']:.2f} dt={rec['dt_fit']:.2f} "
                     f"dNLL={rec['dnll']:+.2f}" if rec["ok"] else "FAILED"),
                  flush=True)
    print(f"[{plan}] shard {shard} finished {len(done)} jobs in "
          f"{time.time()-t0:.0f}s", flush=True)
    return done


# ===========================================================================
#  Merging + reporting
# ===========================================================================
def merge(plan, out_dir=None, save=True):
    """Aggregate every per-job JSON into one ranked table.

    Safe to run while the array is still going -- it reports coverage, so a
    partial table is labelled as partial rather than quietly ranked as if it were
    complete."""
    out_dir = out_dir or os.path.join(SCRATCH, plan)
    jobs = build_jobs(plan)
    recs = []
    for fn in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if fn.endswith(".json") and not fn.startswith("."):
            with open(os.path.join(out_dir, fn)) as f:
                recs.append(json.load(f))
    print(f"[{plan}] {len(recs)}/{len(jobs)} jobs present "
          f"({100*len(recs)/max(len(jobs),1):.0f}%)")

    by = {}
    for r in recs:
        by.setdefault(r["label"], []).append(r)
    table = []
    for label, rs in by.items():
        good = [r for r in rs if r.get("ok")]
        if not good:
            table.append(dict(label=label, n=len(rs), n_failed=len(rs),
                              mean_dnll=float("inf")))
            continue
        dn = np.array([r["dnll"] for r in good])
        dh = np.array([r["dh"] for r in good])
        dd = np.array([r["ddt"] for r in good])
        ok = (np.abs(dh) < 10.0) & (np.abs(dd) < 2.0)
        table.append(dict(
            label=label, n=len(rs), n_failed=len(rs) - len(good),
            mean_dnll=float(dn.mean()), med_dnll=float(np.median(dn)),
            max_dnll=float(dn.max()),
            med_abs_dh=float(np.median(np.abs(dh))),
            max_abs_dh=float(np.abs(dh).max()),
            med_abs_ddt=float(np.median(np.abs(dd))),
            max_abs_ddt=float(np.abs(dd).max()),
            spread_h=float(np.std([r["h_fit"] for r in good])),
            spread_dt=float(np.std([r["dt_fit"] for r in good])),
            frac_ok=float(ok.mean()),
            wall=float(np.mean([r["wall"] for r in good]))))
    table.sort(key=lambda t: t["mean_dnll"])

    cfgs = {c["label"]: c for c in
            ([j["cfg"] for j in jobs] if jobs else [])}
    if save:
        os.makedirs(RESULTS, exist_ok=True)
        path = os.path.join(RESULTS, f"hyperopt_cluster_{plan}.json")
        with open(path, "w") as f:
            json.dump(dict(plan=plan, n_jobs=len(jobs), n_done=len(recs),
                           table=table, cfgs=cfgs, runs=recs), f, indent=1)
        print("saved", path)
    return table


def report(plan, top=25):
    """Print the ranking of a merged plan, best first."""
    path = os.path.join(RESULTS, f"hyperopt_cluster_{plan}.json")
    if not os.path.exists(path):
        return merge(plan)
    with open(path) as f:
        d = json.load(f)
    table, cfgs = d["table"], d.get("cfgs", {})
    print(f"[{plan}] {d['n_done']}/{d['n_jobs']} jobs"
          + ("  (PARTIAL)" if d["n_done"] < d["n_jobs"] else ""))
    hdr = (f"{'config':<24}{'n':>5}{'fail':>5}{'mean dNLL':>11}{'max dNLL':>11}"
           f"{'|dh| med':>10}{'|ddt| med':>11}{'sd(h)':>8}{'sd(dt)':>8}"
           f"{'ok':>7}{'s/fit':>8}")
    print(hdr); print("-" * len(hdr))
    for t in table[:top]:
        if t["mean_dnll"] == float("inf"):
            print(f"{t['label']:<24}{t['n']:>5}{t['n_failed']:>5}   all failed")
            continue
        print(f"{t['label']:<24}{t['n']:>5}{t['n_failed']:>5}"
              f"{t['mean_dnll']:>11.2f}{t['max_dnll']:>11.2f}"
              f"{t['med_abs_dh']:>10.2f}{t['med_abs_ddt']:>11.3f}"
              f"{t['spread_h']:>8.2f}{t['spread_dt']:>8.3f}"
              f"{t['frac_ok']*100:>6.0f}%{t['wall']:>8.0f}")
    best = table[0]
    print(f"\nbest: {best['label']}")
    if best["label"] in cfgs:
        print(json.dumps(cfgs[best["label"]], indent=1, sort_keys=True))
    return table


# Measured on one core (torch 2.8, float64) for the adopted config at the adopted
# dt_step = 0.5 Myr (= 800 leapfrog steps over the 400 Myr flow):
#   time   ~600 s per fit at N = 20000;
#   memory 0.67 GB at N = 5000 and 1.16 GB at N = 20000, i.e. a ~0.5 GB fixed
#          floor plus ~33 kB per star.
# BOTH the per-star memory and the time are really per star PER STEP: d(NLL)/dh
# needs the whole leapfrog trajectory retained for the backward pass, so halving
# dt_step doubles the step count and doubles both.  Cost therefore scales as
# N / dt_step, which is why the `robust` plan does not cross those two axes.
SEC_PER_FIT_20K = 600.0            # at N = 20000, dt_step = 0.5
MEM_FLOOR_GB = 0.5
MEM_PER_STAR_GB = 33e-6            # at dt_step = 0.5
REF_DT_STEP = 0.5


def _cost(job):
    """(seconds, GB) for one job on one core."""
    scale = (job["N"] / 20000.0) * (REF_DT_STEP / job["dt_step"])
    return (SEC_PER_FIT_20K * scale,
            MEM_FLOOR_GB + MEM_PER_STAR_GB * job["N"]
            * (REF_DT_STEP / job["dt_step"]))


def describe(plan):
    """Cost of a plan -- wall time AND memory per worker -- before anything is
    submitted.  Memory is the constraint people get wrong here: a shard running
    8 concurrent fits at N = 5e5 wants ~140 GB, not the 16 GB a default sbatch
    asks for."""
    jobs = build_jobs(plan)
    out_dir = os.path.join(SCRATCH, plan)
    have = (len([f for f in os.listdir(out_dir) if f.endswith(".json")])
            if os.path.isdir(out_dir) else 0)
    labels = sorted({j["cfg"]["label"] for j in jobs})
    Ns = sorted({j["N"] for j in jobs})
    costs = [_cost(j) for j in jobs]
    est = sum(c[0] for c in costs) / 3600.0
    slowest = max(c[0] for c in costs) / 3600.0
    mem = max(c[1] for c in costs)
    print(f"plan {plan!r}")
    print(f"  jobs            : {len(jobs)}  ({have} already on disk)")
    print(f"  configs         : {len(labels)}")
    print(f"  starts x seeds  : {len({(j['h_init'], j['dt_init']) for j in jobs})}"
          f" x {len({j['seed'] for j in jobs})}")
    print(f"  N               : {Ns}")
    print(f"  dt_step         : {sorted({j['dt_step'] for j in jobs})}")
    print(f"  data seeds      : {sorted({j['data_seed'] for j in jobs})}")
    print(f"  est. core-hours : {est:.0f}   "
          f"(~{est/64:.1f} h wall on 64 cores, ~{est/256:.1f} h on 256)")
    print(f"  longest fit     : {slowest:.1f} h on one core")
    print(f"  mem per worker  : {mem:.1f} GB for the heaviest job  ->  submit "
          f"with --mem-per-cpu={int(np.ceil(mem))}G")
    print(f"  scratch         : {out_dir}")
    return jobs


# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(prog="experiments.hyperopt_cluster",
                                 description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="cost of a plan, nothing runs")
    p.add_argument("plan")

    p = sub.add_parser("run", help="run this task's shard of a plan")
    p.add_argument("plan")
    p.add_argument("--shard", type=int, default=int(
        os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    p.add_argument("--n-shards", type=int, default=int(
        os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    p.add_argument("--workers", type=int, default=int(
        os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--out-dir", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="run at most this many jobs (for a timed trial)")
    p.add_argument("--redo", action="store_true",
                   help="recompute jobs already on disk")

    p = sub.add_parser("merge", help="aggregate per-job JSONs into one table")
    p.add_argument("plan")
    p.add_argument("--out-dir", default=None)

    p = sub.add_parser("report", help="print the ranking of a merged plan")
    p.add_argument("plan")
    p.add_argument("--top", type=int, default=25)

    p = sub.add_parser("smoke", help="alias for: run smoke --workers 2")
    p.add_argument("--workers", type=int, default=2)

    a = ap.parse_args(argv)
    if a.cmd == "plan":
        describe(a.plan)
    elif a.cmd == "run":
        run(a.plan, shard=a.shard, n_shards=a.n_shards, workers=a.workers,
            out_dir=a.out_dir, limit=a.limit, redo=a.redo)
    elif a.cmd == "merge":
        merge(a.plan, out_dir=a.out_dir)
    elif a.cmd == "report":
        report(a.plan, top=a.top)
    elif a.cmd == "smoke":
        run("smoke", workers=a.workers)
        merge("smoke")
        report("smoke")


if __name__ == "__main__":
    sys.exit(main())
