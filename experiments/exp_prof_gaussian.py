"""Regular-Gaussian notebook: profile-likelihood comparison across NF kinds.

True f0 = centred 2-D Gaussian.  Compares an analytic linear bijection (ax+b,
4 params, closed-form profile) against two NF families -- NVP/coupling and
residual -- each swept over parameter count (a few -> ~1000).  Two profile-
likelihood plots (one per kind) show how differently each constrains h_dm, and
an init-guess -> fitted-h map replaces the old init-spread metric.
"""
from experiments import common as C


def run(cfg):
    z, w = C.make_gaussian_data(N=cfg.get("N", 20000), seed_sample=cfg.get("seed", 3))
    return C.run_profile_comparison(
        "PG", z, w,
        true_pdf_z=lambda g: C.gauss_pdf(g, 0.0, C.Z0_SIGMA),
        true_pdf_w=lambda g: C.gauss_pdf(g, 0.0, C.W0_SIGMA),
        include_analytic=True)


def plot(res):
    return C.plot_profile_comparison(
        res, title="Regular Gaussian $f_0$ — profile likelihood by NF kind & #parameters")
