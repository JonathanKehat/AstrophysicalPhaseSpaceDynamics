"""Test 12 -- bimodal f0: two Gaussians in w with different means, same width.

f0 = N(0, sigma_z) in z  x  [0.5 N(-35,20) + 0.5 N(+35,20)] in w.

Sweep of normalizing flows ONLY, at increasing parameter counts, to show how the
number of nuisance parameters trades off shape-fidelity (how well the flow mimics
the bimodal true f0 / data) against the constraint on h.
"""
from experiments import common as C

COMPS = [(-35.0, 20.0), (35.0, 20.0)]
WEIGHTS = [0.5, 0.5]


def run(cfg):
    N = cfg.get("N", 2000)
    z, w = C.make_mixture_w_data(N=N, comps=COMPS, weights=WEIGHTS, seed=12)
    return C.run_nf_sweep_test(
        "12", z, w,
        true_pdf_z=lambda g: C.gauss_pdf(g, 0.0, C.Z0_SIGMA),
        true_pdf_w=C.mixture_pdf(COMPS, WEIGHTS),
        sizes=cfg.get("sizes"))


def plot(res):
    return C.plot_nf_sweep(
        res, title="Test 12 -- bimodal $f_0$: normalizing flows of increasing "
                   "capacity (shape fidelity vs $h$ constraint)")
