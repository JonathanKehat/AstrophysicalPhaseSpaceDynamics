"""Three-width notebook: profile-likelihood comparison across NF kinds.

True f0 = N(0, sigma_z) in z  x  [0.5 N(0,15) + 0.35 N(0,40) + 0.15 N(0,90)] in w
-- a THREE-component velocity DF: a sharp core, an intermediate shoulder and a
broad wing (an extra Gaussian on top of the two-width case).  No analytic-linear
reference (a linear map cannot make a non-Gaussian shape).  A distribution panel
shows the three generating gaussians on top of true / data / best fits.
"""
from experiments import common as C

COMPS = [(0.0, 15.0), (0.0, 40.0), (0.0, 90.0)]
WEIGHTS = [0.5, 0.35, 0.15]

# extended flexible ladder (ZWNormalizingFlow (c,h) -> params): densely samples the
# ORDER-100 range -- 76, 106, 152, 204, 304 -- to find where a flow first fits the
# three-component shape AND recovers h; then 424 / 1014 / 2028 / 4188 for growth.
FLEX_CONFIGS = [(2, 4), (1, 8), (4, 4), (3, 6), (8, 4),
                (4, 8), (3, 16), (6, 16), (6, 24)]  # 76,106,152,204,304,424,1014,2028,4188


def run(cfg):
    z, w = C.make_mixture_w_data(N=cfg.get("N", 20000), comps=COMPS, weights=WEIGHTS,
                                 seed=cfg.get("seed", 3))
    return C.run_profile_comparison(
        "P3W", z, w,
        true_pdf_z=lambda g: C.gauss_pdf(g, 0.0, C.Z0_SIGMA),
        true_pdf_w=C.mixture_pdf(COMPS, WEIGHTS),
        include_analytic=False,
        w_components=[(wt, mu, s) for wt, (mu, s) in zip(WEIGHTS, COMPS)],
        flex_configs=FLEX_CONFIGS,
        n_feat=99)   # show EVERY fit in the 2-D density panel too (eval is free here)


def plot(res):
    return C.plot_profile_comparison(
        res, show_dist=True,
        title="Three-width $f_0$ (core 15 + shoulder 40 + wing 90) — "
              "profile likelihood by NF kind")
