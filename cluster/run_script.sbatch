#!/bin/bash
# ---------------------------------------------------------------------------
# Run any single project script / module as one SLURM job.  Submit from the
# repository root; everything after the script name is passed through:
#
#   sbatch cluster/run_script.sbatch experiments/run_mirror_fit.py 2w 16
#   sbatch cluster/run_script.sbatch -m experiments.joint_profile_fit
#   sbatch --cpus-per-task=4 --mem=16G --time=48:00:00 cluster/run_script.sbatch ...
#
# Memory guide (one fit, float64): ~0.5 GB + 33 kB per star (autograd graph of
# the leapfrog), i.e. ~1.2 GB at N=2e4, ~3.8 GB at N=1e5.
# ---------------------------------------------------------------------------
#SBATCH --job-name=phasespace
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
mkdir -p logs results

# module load python/3.11
if [ -f "$PROJECT/.venv/bin/activate" ]; then
    source "$PROJECT/.venv/bin/activate"
fi
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# Match torch's thread pool to the allocation (default would grab the whole node).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg   # no display on compute nodes

echo "host=$(hostname)  job=${SLURM_JOB_ID:-local}  cpus=${SLURM_CPUS_PER_TASK:-?}  commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "cmd: python -u $*"
python -u "$@"
