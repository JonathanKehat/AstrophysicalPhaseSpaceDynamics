"""Re-run the Gaussian time-parameter basin-hopping studies with JOINT local fits
(h, Delta t and every flow parameter minimised together after each jump).
Writes results/PGT_hop_N20000_dtmin0{,_nvp}_joint.pkl and the matching
PGT_hopprof_*_joint.pkl.  Same data/starts/seeds as the notebook's figures."""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import common as C

if __name__ == "__main__":
    z, w = C.make_gaussian_data(N=20000, seed_sample=3)
    for flow, kw in (("affine", {}), ("nvp", dict(seeds=(0,), dh=20.0, ddt=2.0))):
        t0 = time.time()
        hop = C.run_hop_study(z, w, flow=flow, local="joint", **kw)
        print(f"[{flow}] hop study done in {(time.time()-t0)/60:.1f} min", flush=True)
        print(C.hop_summary(hop), flush=True)
        prof = C.hop_profile(hop, z, w)
        print(f"[{flow}] profile: h = {prof['h_best']:.2f} +/- {prof['sig_h']:.2f} pc, "
              f"dt = {prof['dt_best']:.3f} +/- {prof['sig_dt']:.3f} Myr  "
              f"[{(time.time()-t0)/60:.1f} min total]", flush=True)
