"""Two-widths notebook: profile-likelihood comparison across NF kinds.

True f0 = N(0, sigma_z) in z  x  [0.6 N(0,15) + 0.4 N(0,60)] in w -- a sharp core
plus broad wings with a LARGE width contrast (15 vs 60).  No analytic-linear
reference (a linear map cannot make a non-Gaussian shape).  Adds a distribution
panel showing the two generating gaussians on top of true / data / best fits.
"""
from experiments import common as C

COMPS = [(0.0, 15.0), (0.0, 60.0)]
WEIGHTS = [0.6, 0.4]

# extended flexible ladder (ZWNormalizingFlow (c,h) -> params): densely samples the
# ORDER-100 range -- 76, 106, 152, 204, 304 -- to find where a flow first fits the
# core+wings shape AND recovers h; then 424 / 1014 and two large 2028, 4188 to show
# how it holds up (or over-fits) as capacity grows.
FLEX_CONFIGS = [(2, 4), (1, 8), (4, 4), (3, 6), (8, 4),
                (4, 8), (3, 16), (6, 16), (6, 24)]  # 76,106,152,204,304,424,1014,2028,4188


def run(cfg):
    z, w = C.make_mixture_w_data(N=cfg.get("N", 20000), comps=COMPS, weights=WEIGHTS,
                                 seed=cfg.get("seed", 3))
    return C.run_profile_comparison(
        "P2W", z, w,
        true_pdf_z=lambda g: C.gauss_pdf(g, 0.0, C.Z0_SIGMA),
        true_pdf_w=C.mixture_pdf(COMPS, WEIGHTS),
        include_analytic=False,
        w_components=[(WEIGHTS[0], 0.0, 15.0), (WEIGHTS[1], 0.0, 60.0)],
        flex_configs=FLEX_CONFIGS,
        n_feat=99)   # show EVERY fit in the 2-D density panel too (eval is free here)


def plot(res):
    return C.plot_profile_comparison(
        res, show_dist=True,
        title="Two-width $f_0$ (core 15 + wings 60) — profile likelihood by NF kind")
