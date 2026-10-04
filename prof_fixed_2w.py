"""Drift-free profile scan for the two-width (core 15 + wings 60) case.

Differences from experiments.common._profile_curve:
  * NO chained warm-starts.  Every grid point is fitted INDEPENDENTLY from the
    identical (identity) init with the identical epoch budget, so the point at
    distance j from the centre no longer receives warm + j*250 cumulative
    epochs.  That accumulation made NLL_p(h) drift downward with |h - h_hat|
    and slid the minimum to the grid edges.
  * No joint_fit anchor -> the curve is not centred on h_hat by construction.
  * Wider grid (160-400) so a displaced minimum is not censored at the edge.
  * Uniform budget (no 800-vs-250 asymmetry).  A convergence probe showed the
    old 800-epoch budget leaves these flows 41-52 nats above their converged
    value -- comparable to the whole well depth -- so 3000 is used here.

Writes results/fixed2w_<key>.pkl.   Run as:  prof_fixed_2w.py <model_key>
"""
import os, sys, time, pickle, numpy as np, torch
from experiments import common as C
from fitter import ZWNormalizingFlow
from experiments.exp_A_linear_bijection import DiagonalAffineFlow

torch.set_num_threads(3)

COMPS, WEIGHTS = [(0., 15.), (0., 60.)], [0.6, 0.4]
HGRID = np.arange(160.0, 401.0, 15.0)          # 17 points, 250 lands on-grid
EPOCHS = 3000
N = 20000
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

MODELS = {                                      # key: (config, lr, label)
    "affine4": (None,     3e-2, "affine 4p"),
    "f304":    ((8,  4),  1e-3, "304p"),
    "f1014":   ((3, 16),  1e-3, "1014p"),
    "f4188":   ((6, 24),  1e-3, "4188p"),
}


def make(key, zb, wb):
    cfg, _, _ = MODELS[key]
    if cfg is None:
        return DiagonalAffineFlow(a_init=(C.Z0_SIGMA, C.W0_SIGMA)).double()
    f = ZWNormalizingFlow(*cfg).double()
    f.set_standardization(zb, wb)
    return f


def main(key):
    _, lr, label = MODELS[key]
    z, w = C.make_mixture_w_data(N=N, comps=COMPS, weights=WEIGHTS, seed=3)
    clouds = [C.backint(z, w, h, ns=2000) for h in HGRID]

    nll, tail = np.zeros(len(HGRID)), np.zeros(len(HGRID))
    npar = None
    t0 = time.time()
    for i, h in enumerate(HGRID):
        zb, wb = clouds[i]
        torch.manual_seed(0)                    # identical init at EVERY h
        f = make(key, zb, wb)
        npar = sum(p.numel() for p in f.parameters())
        opt = torch.optim.Adam(f.parameters(), lr=lr)
        zt, wt = torch.tensor(zb), torch.tensor(wb)
        hist = []
        for _ in range(EPOCHS):
            opt.zero_grad(); loss = -f.log_prob(zt, wt).mean(); loss.backward()
            torch.nn.utils.clip_grad_norm_(f.parameters(), 10.0); opt.step()
            hist.append(loss.item())
        nll[i] = N * hist[-1]
        tail[i] = (hist[-200] - hist[-1]) * N / 2.0   # nats still falling /100ep
        print(f"  [{label}] h={h:6.1f}  NLL={nll[i]:14.3f}  "
              f"tail={tail[i]:6.2f} nats/100ep  ({time.time()-t0:.0f}s)", flush=True)

    d = nll - nll.min()
    k = int(d.argmin())
    if 0 < k < len(HGRID) - 1:                  # parabolic sub-grid refinement
        y0, y1, y2 = d[k-1], d[k], d[k+1]
        den = y0 - 2*y1 + y2
        h_par = HGRID[k] + (0.5*(y0-y2)/den if den else 0.0) * (HGRID[1]-HGRID[0])
    else:
        h_par = float(HGRID[k])
    out = dict(key=key, label=label, npar=npar, hgrid=HGRID, nll=nll, dnll=d,
               tail=tail, h_grid_min=float(HGRID[k]), h_parab=float(h_par),
               epochs=EPOCHS, lr=lr, edge=(k in (0, len(HGRID)-1)))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"fixed2w_{key}.pkl"), "wb") as fh:
        pickle.dump(out, fh)
    print(f"DONE {label}: grid-min h={HGRID[k]:.1f}  parab h={h_par:.1f}  "
          f"edge={out['edge']}  [{(time.time()-t0)/60:.1f} min]", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
