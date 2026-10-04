#!/bin/bash
# ---------------------------------------------------------------------------
# SLURM array for the cluster-scale (h_dm, Delta t) hyper-parameter study.
#
#   sbatch --array=0-63 experiments/slurm/hyperopt_array.sbatch search
#   sbatch --array=0-63 experiments/slurm/hyperopt_array.sbatch validate
#
# MEMORY, NOT CORES, IS THE BINDING CONSTRAINT.  dNLL/dh needs the whole ~800-step
# leapfrog autograd graph retained, so one fit costs ~0.5 GB + 33 kB per star:
# 1.2 GB at N = 2e4, 3.8 GB at N = 1e5, 17 GB at N = 5e5 -- times the number of
# concurrent workers in the task.  The defaults below (8 cpus, 4 GB each) suit
# `search` and `validate`.  The `robust` plan reaches N = 5e5, so give it fewer,
# fatter workers and more time (one 5e5 fit is ~4 h on one core):
#
#   sbatch --array=0-31 --cpus-per-task=2 --mem-per-cpu=20G --time=24:00:00 \
#          experiments/slurm/hyperopt_array.sbatch robust
#
# `python -m experiments.hyperopt_cluster plan <plan>` prints both numbers.
# If a task does run out of time or memory, just resubmit the same line: the
# finished jobs are on disk and are skipped.
#
# Each task rebuilds the full job list from a fixed seed and takes the slice
# `index % n_shards == SLURM_ARRAY_TASK_ID`, so the tasks need no coordination --
# only a shared filesystem for the per-job JSONs.  A job whose JSON already
# exists is skipped, so this script is safe to resubmit verbatim after a
# timeout, a preemption or a partial failure: it resumes, it does not restart.
#
# BEFORE SUBMITTING
#   python -m experiments.hyperopt_cluster smoke        # ~5 min, correctness
#   python -m experiments.hyperopt_cluster plan search  # cost, nothing runs
#
# AFTERWARDS
#   python -m experiments.hyperopt_cluster merge  search
#   python -m experiments.hyperopt_cluster report search
# ---------------------------------------------------------------------------
#SBATCH --job-name=hdt-hyperopt
#SBATCH --output=logs/hyperopt_%A_%a.out
#SBATCH --error=logs/hyperopt_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G          # one concurrent fit per cpu; see the note above
# Array size is given on the command line (--array=0-63) so one script serves
# every plan.  SLURM_ARRAY_TASK_COUNT is then set automatically and is what the
# runner uses as the shard count.

set -euo pipefail

PLAN="${1:-search}"

# --- project root ----------------------------------------------------------
# SLURM_SUBMIT_DIR is where sbatch was called from; submit from the project root.
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"
mkdir -p logs

# --- environment -----------------------------------------------------------
# cluster/setup_env.sh creates ./.venv; it is activated here if present.
# If the cluster needs a module for python, uncomment and adapt:
# module load python/3.11
if [ -f "$PROJECT/.venv/bin/activate" ]; then
    source "$PROJECT/.venv/bin/activate"
fi
PY="${PY:-python3}"

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"

# One torch thread per worker: the runner already uses every allocated core by
# running $SLURM_CPUS_PER_TASK fits concurrently, and letting torch thread on top
# of that oversubscribes the node and makes everything slower.
export HYPEROPT_TORCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Per-job JSONs churn a lot of small files.  Point this at node-local or parallel
# scratch if the project share is a network filesystem, then merge from there.
# export HYPEROPT_SCRATCH=/scratch/$USER/hdt_hyperopt

echo "host=$(hostname)  plan=$PLAN  task=${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}  cpus=${SLURM_CPUS_PER_TASK}"

srun "$PY" -m experiments.hyperopt_cluster run "$PLAN" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --workers "${SLURM_CPUS_PER_TASK}"

# The last task to finish merges.  Harmless if several do -- merge only reads the
# per-job JSONs and rewrites the summary atomically enough for this purpose, and
# it reports coverage so a partial table is labelled as partial.
if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    "$PY" -m experiments.hyperopt_cluster merge  "$PLAN" || true
    "$PY" -m experiments.hyperopt_cluster report "$PLAN" || true
fi
