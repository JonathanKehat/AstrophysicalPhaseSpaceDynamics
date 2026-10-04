"""T2-T4 for one dataset with the mirror-symmetric f0 (experiments/mirror_f0.py):
joint basin hopping from every initial guess -> converged full-sample refine of
every landing -> full Hessian at the best -> f0 report.  All cached.

usage: python3 experiments/run_mirror_fit.py <tag> [n_starts]
  n_starts = 8 (default, mirror_f0.HOP_STARTS) or 16 (common.HOP_STARTS)."""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import common as C, mirror_f0 as M

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "2w"
    n_starts = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    starts = M.HOP_STARTS if n_starts == 8 else C.HOP_STARTS
    t0 = time.time()
    hop = M.hop_study(tag, starts=starts)
    print(f"[{tag}] hopping done {(time.time()-t0)/60:.1f} min", flush=True)
    for r in hop["runs"]:
        print(f"   start ({r['h_init']}, {r['dt_init']}) -> hop ({r['h_fit']:.2f}, "
              f"{r['dt_fit']:.3f})  J_full={r['J_full']:.2f}  hops={r['n_hops']}", flush=True)
    ref = M.refine(tag, hop)
    print(f"[{tag}] refine done {(time.time()-t0)/60:.1f} min", flush=True)
    err = M.hop_errors(tag, ref)
    print(f"[{tag}] hessian: is_min={err['is_min']} sig_h={err['sig_h']:.2f} "
          f"sig_dt={err['sig_dt']:.3f} corr={err['corr']:+.3f} "
          f"flow eig [{err['flow_min_eig']:.2e}, {err['flow_max_eig']:.2e}] "
          f"grad_max={err['grad_max']:.2e}", flush=True)
    print(M.fit_table(ref, err).round(3).to_string(), flush=True)
    print(M.verdict(ref, err), flush=True)
    rep = M.f0_report(tag, ref)
    print(f"[{tag}] f0: held-out excess vs true model = {rep['heldout_excess']:+.5f} "
          f"nats/star; in-sample gain {rep['insample_gain']:+.5f}; "
          f"R_fit={rep['R_fit']:.3f} R_true={rep['R_true']:.3f}", flush=True)
    print("   w-marginal fit/true at |w| =", rep["w_q"], ":",
          (rep["pw_fit"] / rep["pw_true"]).round(3), flush=True)
    print(f"[{tag}] ALL DONE {(time.time()-t0)/60:.1f} min", flush=True)
