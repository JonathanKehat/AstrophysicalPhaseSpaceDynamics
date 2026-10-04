"""Two-width f0: penalised-f0 fit with JOINT basin hopping.  Stages cached in
results/PF2w_* and results/PGT2w_hop_*; the notebook
NF_hdm_fit__two_widths__time_parameter.ipynb reads the same caches."""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import common as C, penalized_f0 as P

if __name__ == "__main__":
    t0 = time.time()
    z, w = P.load_data(20000)
    T = P.candidate_teeth(z, w)
    print("teeth:", [(t["h"], t["dt"], round(t["affine_dnll"], 1)) for t in T["teeth"]],
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)
    sc = P.lambda_scan(z, w, T["teeth"])
    print(f"lambda scan: kind={sc['kind']} lam*={sc['lam']:g} best tooth={sc['tooth']} "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)
    print(P.lambda_table(sc).round(1).to_string(), flush=True)
    hop = P.hop_fit(z, w, sc)
    print(f"hop done [{(time.time()-t0)/60:.1f} min]", flush=True)
    print(P.hop_table(hop).round(3).to_string(), flush=True)
    he = P.hop_errors(z, w, hop)
    print(f"hessian: is_min={he['is_min']} sig_h={he['sig_h']:.2f} sig_dt={he['sig_dt']:.3f} "
          f"corr={he['corr']:+.3f} grad_max={he['grad_max']:.2e} "
          f"flow eig [{he['flow_min_eig']:.2e},{he['flow_max_eig']:.2e}] "
          f"[{(time.time()-t0)/60:.1f} min]", flush=True)
    rep = P.f0_report_hop(z, w, hop)
    for k in ("penalised", "unpenalised"):
        r = rep[k]
        print(f"f0 {k}: lam={r['lam']:g} held-out excess vs true = {r['excess_per_star']:+.4f} "
              f"nats/star, R={r['R']:.3f}", flush=True)
    print(f"R_true={rep['R_true']:.3f}  ALL DONE [{(time.time()-t0)/60:.1f} min]", flush=True)
