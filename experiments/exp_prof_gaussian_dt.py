"""Time-parameter extension of the regular-Gaussian notebook.

Same simulated data as `exp_prof_gaussian` (a single centred 2-D Gaussian f0
evolved to T_OBS = 400 Myr), but the total dynamical time of the Hamiltonian
flow

    Delta t = N_steps * dt

is now a FREE fit parameter.  PARAMETRISATION (2026-09-17): the leapfrog
timestep dt is what accuracy constrains, so it is PINNED IN ADVANCE at
C.DT_STEP = 0.5 Myr (<= C.DT_STEP_MAX) and the free, real-valued parameter is
the STEP COUNT N_steps -- hence the total time.  It is optimised jointly with
h_dm and the NF bijection parameters, subject to Delta t >= dt_min.

The fitter's hyper-parameters are the ones selected by the sweep in
`experiments/hyperopt_time.py`; the headline of that study is that NO purely
local setting works, because NLL(Delta t) is an alias comb, so the fit opens
with a global de-aliasing sweep and closes with a full-batch polish.

Builds the alias comb, a fine 2-D profile likelihood zoomed on the joint
minimum, its 1-D profiles, and the joint-optimisation trajectories.
"""
from experiments import common as C


def run(cfg):
    dt_min = cfg.get("dt_min", C.DT_MIN)
    z, w = C.make_gaussian_data(N=cfg.get("N", 20000), seed_sample=cfg.get("seed", 3))
    # Once the bound is dropped to zero, two further starts become reachable:
    # BELOW the well, at Delta t = 100 and 0 Myr.  They are run with exactly the
    # same tuned fitter as every other start -- the point is that they too land
    # on the global minimum, so the Delta t >= 200 prior is not what makes the
    # fit work.  (They were previously run with the de-aliasing sweep disabled,
    # as a demonstration of what a purely local optimiser does; that made the
    # figure look as if some starts fail, which is the opposite of the result.)
    extra = ((250, 100), (250, 0)) if dt_min < 100 else ()
    return C.run_time_parameter("PGT", z, w, dt_min=dt_min, extra_inits=extra,
                                extra_tag="tuned")


def plot(res):
    print(C.hdt_summary(res))
    C.plot_hdt_landscape(res)
    C.plot_hdt_profiles(
        res, title=r"Regular Gaussian $f_0$ — 1-D profiles of $\Delta t$ and $h_{\rm dm}$")
    return C.plot_hdt_trajectories(
        res, title=r"joint optimisation of $\Delta t$ and $h_{\rm dm}$")
