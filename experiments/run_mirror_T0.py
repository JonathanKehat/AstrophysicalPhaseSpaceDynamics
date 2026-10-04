"""T0 tooth test for the mirror-symmetric f0 (experiments/mirror_f0.py).
Converged joint (Delta t + flow) fits, h fixed at 200/250/300, five teeth."""
import sys, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import mirror_f0 as M

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "2w"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    ctrl = (len(sys.argv) > 3 and sys.argv[3] == "ctrl")
    t0 = time.time()
    r = M.tooth_test(tag, n=n, flows=(M.FLOW,), seeds=(0, 1))
    print(f"[{tag} n={n}] mirror done {(time.time()-t0)/60:.1f} min", flush=True)
    print(M.tooth_table(r).round(3).to_string(), flush=True)
    print(M.tooth_verdict(r), flush=True)
    if ctrl:
        c = M.tooth_test(tag, n=n, flows=("zw114",), seeds=(0,))
        print(f"control done {(time.time()-t0)/60:.1f} min", flush=True)
        print(M.tooth_table(c).round(3).to_string(), flush=True)
        print(M.tooth_verdict(c, flow="zw114"), flush=True)
