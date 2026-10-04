"""Re-run the Gaussian joint basin-hopping studies (force=True) after the L-BFGS
restart fix in `_joint_local`; same configuration as run_gauss_joint."""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import common as C

if __name__ == "__main__":
    z, w = C.make_gaussian_data(N=20000, seed_sample=3)
    for flow, kw in (("affine", {}), ("nvp", dict(seeds=(0,), dh=20.0, ddt=2.0))):
        t0 = time.time()
        hop = C.run_hop_study(z, w, flow=flow, local="joint", force=True, **kw)
        print(f"[{flow}] RERUN hop study done in {(time.time()-t0)/60:.1f} min", flush=True)
        print(C.hop_summary(hop), flush=True)
        prof = C.hop_profile(hop, z, w, force=True)
        print(f"[{flow}] RERUN profile: h = {prof['h_best']:.2f} +/- {prof['sig_h']:.2f} pc, "
              f"dt = {prof['dt_best']:.3f} +/- {prof['sig_dt']:.3f} Myr", flush=True)
