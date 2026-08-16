#!/bin/bash --login
# One actual Procgen trainer per Jupyter-backed allocation.  The notebook is
# localhost-only and exists to provide a durable interactive control surface;
# this job never keeps a GPU after its trainer exits.
#SBATCH --job-name=pg-j2b-jupyter
#SBATCH --partition=gpuA
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=24-27%4
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.err

set -euo pipefail
ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
RUN_ROOT="$ROOT/formal_1m_seed0_jupyter_v2"
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ENV_NAMES=(bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10)
ENV_NAME=${ENV_NAMES[$((TASK_ID % 4))]}
RUN="$RUN_ROOT/full_clean_conservative/$ENV_NAME/seed0"
WORKER_DIR="$ROOT/jupyter_workers/${SLURM_JOB_ID}_${TASK_ID}"
mkdir -p "$WORKER_DIR/jupyter_runtime"

export JUPYTER_RUNTIME_DIR="$WORKER_DIR/jupyter_runtime"
export JUPYTER_CONFIG_DIR="$WORKER_DIR/jupyter_config"
mkdir -p "$JUPYTER_CONFIG_DIR"
PORT=$((18000 + TASK_ID))
# The module brings an Anaconda libcrypto that conflicts with the Procgen
# Python 3.9 venv.  Keep it inside the notebook child only; the trainer below
# inherits the original SLURM environment exactly as in the working batch run.
(
  module load apps/binapps/jupyter-notebook/any
  exec jupyter notebook --no-browser --ip=127.0.0.1 --port="$PORT" --NotebookApp.port_retries=0
) >"$WORKER_DIR/jupyter.log" 2>&1 &
JUPYTER_PID=$!
printf 'node=%s\nport=%s\npid=%s\nrun=%s\n' "$(hostname)" "$PORT" "$JUPYTER_PID" "$RUN" > "$WORKER_DIR/jupyter_worker.txt"

set +e
PROCGEN_RUN_ROOT="$RUN_ROOT" bash "$ROOT/run_task.sh" "$TASK_ID"
RC=$?
set -e
kill "$JUPYTER_PID" 2>/dev/null || true
wait "$JUPYTER_PID" 2>/dev/null || true
exit "$RC"
