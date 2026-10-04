"""Test 13 -- shifted-mean Gaussian f0 (bulk vertical streaming).

f0 = N(0, sigma_z) in z  x  N(mu_w=30, 20) in w: a single Gaussian whose w
component has a NONZERO mean (net <w> != 0 at t=0).  A linear bijection
F(u)=a*u+b handles this exactly with 4 parameters (b_w learns the 30 km/s
offset) -- no big NN needed.
"""
from experiments import common as C
from experiments.exp_A_linear_bijection import DiagonalAffineFlow

MU_W = 30.0
SW = 20.0
N_PARAMS = 4


def run(cfg):
    N = cfg.get("N", 2000)
    z, w = C.make_gaussian_data(N=N, w_sigma=SW, mu_w=MU_W, seed_sample=13)
    mk = lambda: DiagonalAffineFlow(a_init=(C.Z0_SIGMA * 1.3, SW * 1.3),
                                    b_init=(0.0, 15.0)).double()
    return C.run_density_test(
        "13", z, w, mk,
        true_pdf_z=lambda g: C.gauss_pdf(g, 0.0, C.Z0_SIGMA),
        true_pdf_w=lambda g: C.gauss_pdf(g, MU_W, SW),
        n_params=N_PARAMS, profile_mode="gauss")


def plot(res):
    return C.plot_density_test(
        res, title=f"Test 13 -- shifted-mean Gaussian, linear bijection "
                   f"({res['n_params']} params):  h = {res['h_min']:.0f} "
                   f"$\\pm$ {res['sig']:.0f} pc")
