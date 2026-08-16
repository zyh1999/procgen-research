#!/bin/bash --login
# Identical strict head-only causal control to the gpuA submission, scheduled
# on a single L40S per environment.  The same DMLP1024 strict-2B shape has
# previously fit on 24 GiB 3090s; L40S provides 48 GiB.
#SBATCH --job-name=pg-j2b-head-L
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=12-15%4
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.err

set -euo pipefail
ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
RUN_ROOT="$ROOT/formal_1m_seed0_jupyter_full_head_matched_gpul_v1"
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ENV_NAMES=(bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10)
ENV_NAME=${ENV_NAMES[$((TASK_ID - 12))]}
RUN="$RUN_ROOT/full_head/$ENV_NAME/seed0"
WORKER_DIR="$ROOT/jupyter_workers/${SLURM_JOB_ID}_${TASK_ID}"
mkdir -p "$WORKER_DIR/jupyter_runtime" "$WORKER_DIR/jupyter_config"
export JUPYTER_RUNTIME_DIR="$WORKER_DIR/jupyter_runtime"
export JUPYTER_CONFIG_DIR="$WORKER_DIR/jupyter_config"
PORT=$((18600 + TASK_ID))
(
  module load apps/binapps/jupyter-notebook/any
  exec jupyter notebook --no-browser --ip=127.0.0.1 --port="$PORT" --NotebookApp.port_retries=0
) >"$WORKER_DIR/jupyter.log" 2>&1 &
JUPYTER_PID=$!
printf 'node=%s\nport=%s\npid=%s\nrun=%s\n' "$(hostname)" "$PORT" "$JUPYTER_PID" "$RUN" > "$WORKER_DIR/jupyter_worker.txt"
set +e
PROCGEN_RUN_ROOT="$RUN_ROOT" \
PROCGEN_ADAPTIVE_LR_MAX=0.05 \
PROCGEN_ADAPTIVE_KL_UPPER=0.02 \
bash "$ROOT/run_task.sh" "$TASK_ID"
RC=$?
set -e
kill "$JUPYTER_PID" 2>/dev/null || true
wait "$JUPYTER_PID" 2>/dev/null || true
exit "$RC"
