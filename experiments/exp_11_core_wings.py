"""Test 11 -- non-Gaussian f0: sharp core + heavy wings.

f0 = N(0, sigma_z) in z  x  [0.6 N(0,30) + 0.4 N(0,40)] in w: two Gaussians with
the SAME mean (0) but different widths -> a peaked, heavy-tailed (leptokurtic)
velocity DF.

Sweep of normalizing flows ONLY, at increasing parameter counts, to show how the
number of nuisance parameters trades off shape-fidelity (how well the flow mimics
the true f0 / data) against the constraint on h.
"""
from experiments import common as C

COMPS = [(0.0, 30.0), (0.0, 40.0)]
WEIGHTS = [0.6, 0.4]


def run(cfg):
    N = cfg.get("N", 2000)
    z, w = C.make_mixture_w_data(N=N, comps=COMPS, weights=WEIGHTS, seed=11)
    return C.run_nf_sweep_test(
        "11", z, w,
        true_pdf_z=lambda g: C.gauss_pdf(g, 0.0, C.Z0_SIGMA),
        true_pdf_w=C.mixture_pdf(COMPS, WEIGHTS),
        sizes=cfg.get("sizes"))


def plot(res):
    return C.plot_nf_sweep(
        res, title="Test 11 -- core+wings $f_0$: normalizing flows of increasing "
                   "capacity (shape fidelity vs $h$ constraint)")
