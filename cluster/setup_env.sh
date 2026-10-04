#!/bin/bash
# ---------------------------------------------------------------------------
# One-time environment setup on the cluster.  Run from the repository root on a
# LOGIN node (it needs internet access):
#
#   git clone git@github.com:<user>/<repo>.git && cd <repo>
#   bash cluster/setup_env.sh
#
# Creates ./.venv, installs CPU-only torch + requirements.txt, then runs an
# import check.  experiments/slurm/*.sbatch activate ./.venv automatically.
#
# If the cluster needs a module for a recent python, load it first, e.g.
#   module load python/3.11      (check names with `module avail python`)
# or point PYTHON at a specific interpreter:  PYTHON=python3.11 bash cluster/setup_env.sh
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

echo "python: $($PYTHON --version)  ($(command -v $PYTHON))"
"$PYTHON" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# CPU-only torch wheel (much smaller; the jobs never use a GPU).
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

mkdir -p logs results

python - <<'EOF'
import torch, numpy, scipy, matplotlib, pandas, sklearn
import fitter, experiments.common, experiments.hyperopt_cluster
print("torch", torch.__version__, "| numpy", numpy.__version__,
      "| threads", torch.get_num_threads())
print("environment OK")
EOF

cat <<'EOF'

Next steps:
  source .venv/bin/activate
  python -m experiments.hyperopt_cluster smoke        # ~5 min correctness check
  python -m experiments.hyperopt_cluster plan search  # prints cost / --mem-per-cpu
  sbatch --array=0-63 experiments/slurm/hyperopt_array.sbatch search
EOF
