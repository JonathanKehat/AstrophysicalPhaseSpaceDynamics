"""Shifted-mean notebook: profile-likelihood comparison across NF kinds.

True f0 = N(0, sigma_z) in z  x  N(mu_w=30, 20) in w (bulk vertical streaming).
Same layout as the two-widths notebook (no analytic-linear reference; a
distribution panel with the generating gaussian on top of true / data / fits).
"""
from experiments import common as C

MU_W, SW = 30.0, 20.0


def run(cfg):
    z, w = C.make_gaussian_data(N=cfg.get("N", 20000), w_sigma=SW, mu_w=MU_W,
                                seed_sample=cfg.get("seed", 3))
    return C.run_profile_comparison(
        "PSM", z, w,
        true_pdf_z=lambda g: C.gauss_pdf(g, 0.0, C.Z0_SIGMA),
        true_pdf_w=lambda g: C.gauss_pdf(g, MU_W, SW),
        include_analytic=False,
        w_components=[(1.0, MU_W, SW)])


def plot(res):
    return C.plot_profile_comparison(
        res, show_dist=True,
        title="Shifted-mean $f_0$ ($\\mu_w=30$) — profile likelihood by NF kind")
