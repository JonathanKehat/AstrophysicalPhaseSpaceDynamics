"""Experiment registry + thin run/plot API with a results cache.

Usage (notebook or REPL):
    import experiments as X
    res = X.run("A")        # computes (or loads cached) result dict
    X.plot(res)             # draws the figure   (or X.plot("A"))
    X.show("A")             # run + plot in one call
    X.info()               # print the registry table

CLI:
    python -m experiments list
    python -m experiments run A          # force recompute + cache
    python -m experiments index          # (re)write INDEX.md
"""
import importlib
import os
import pickle

from .registry import REGISTRY, NOTEBOOKS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "results")


def _module(tid):
    if tid not in REGISTRY:
        raise KeyError(f"unknown test id {tid!r}; known: {sorted(REGISTRY)}")
    return importlib.import_module("experiments." + REGISTRY[tid]["module"])


# deterministic cfg knobs that get their own cache file (so runs cache side by
# side and switching a knob never silently reuses the wrong result).
_CACHE_KNOBS = ("N", "dt_min")


def _cache_path(tid, cfg=None):
    """Cache file for a test.  Every deterministic knob in `_CACHE_KNOBS` the cfg
    sets is appended to the filename (results/PG_N10000.pkl,
    results/PGT_N20000_dtmin0.pkl), so different knob values stay cached in
    parallel and switching one never silently reuses the wrong result."""
    suffix = ""
    if cfg:
        for k in _CACHE_KNOBS:
            if k in cfg:
                suffix += f"_{k.replace('_', '')}{int(cfg[k])}"
    return os.path.join(RESULTS_DIR, f"{tid}{suffix}.pkl")


def run(tid, force=False, cfg=None):
    """Run test `tid` and cache the result dict.  Cacheable when cfg is None or
    only sets deterministic knobs (`_CACHE_KNOBS`); other cfg keys bypass the
    cache."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = _cache_path(tid, cfg)
    cacheable = cfg is None or set(cfg) <= set(_CACHE_KNOBS)
    if not force and cacheable and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    res = _module(tid).run(cfg or {})
    res.setdefault("id", tid)
    if cacheable:
        with open(path, "wb") as f:
            pickle.dump(res, f)
    return res


def plot(tid_or_res):
    """Draw the figure for a test.  Accepts a test id (loads/uses cache) or a
    result dict returned by run()."""
    res = run(tid_or_res) if isinstance(tid_or_res, str) else tid_or_res
    return _module(res["id"]).plot(res)


def show(tid, force=False, cfg=None):
    """Convenience: run (or load cache) then plot."""
    return plot(run(tid, force=force, cfg=cfg))


def info():
    """Print the registry grouped by notebook."""
    for nb, desc in NOTEBOOKS.items():
        print(f"\n== {nb} :: {desc} ==")
        for tid, e in REGISTRY.items():
            if e["notebook"] != nb:
                continue
            flag = "OK " if e["status"] == "migrated" else "..."
            print(f"  [{flag}] {tid:4s} {e['title']}")


def index_md():
    """Return an INDEX.md string built from the registry."""
    lines = ["# Test index", "",
             "Auto-generated from `experiments/registry.py` "
             "(`python -m experiments index`).", ""]
    for nb, desc in NOTEBOOKS.items():
        lines += [f"## `{nb}.ipynb` — {desc}", "",
                  "| Test | Studies | True $f_0$ | Code | Status |",
                  "|------|---------|-----------|------|--------|"]
        for tid, e in REGISTRY.items():
            if e["notebook"] != nb:
                continue
            lines.append(f"| {tid} | {e['title']} | {e['true_df']} | "
                         f"`experiments/{e['module']}.py` | {e['status']} |")
        lines.append("")
    return "\n".join(lines)
