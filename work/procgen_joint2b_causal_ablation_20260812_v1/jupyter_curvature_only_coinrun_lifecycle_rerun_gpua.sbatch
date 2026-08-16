#!/bin/bash --login
# Lifecycle-only rerun of the invalidated CoinRun curvature-only seed-0 trace.
# Algorithm/config are identical to the original 2B control.  The worker takes
# an immutable snapshot of its task launcher before Python starts.
#SBATCH --job-name=pg-j2b-coinrerun
#SBATCH --partition=gpuA
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%j.err

set -euo pipefail
ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
RUN_ROOT="$ROOT/formal_1m_seed0_jupyter_curvature_only_coinrun_lifecycle_rerun_v1"
TASK_ID=7
WORKER_DIR="$ROOT/jupyter_workers/${SLURM_JOB_ID}_${TASK_ID}"
mkdir -p "$WORKER_DIR/jupyter_runtime" "$WORKER_DIR/jupyter_config"
export JUPYTER_RUNTIME_DIR="$WORKER_DIR/jupyter_runtime"
export JUPYTER_CONFIG_DIR="$WORKER_DIR/jupyter_config"
PORT=18607
(
  module load apps/binapps/jupyter-notebook/any
  exec jupyter notebook --no-browser --ip=127.0.0.1 --port="$PORT" --NotebookApp.port_retries=0
) >"$WORKER_DIR/jupyter.log" 2>&1 &
JUPYTER_PID=$!
IMMUTABLE_TASK="$WORKER_DIR/run_task.immutable.sh"
cp "$ROOT/run_task.sh" "$IMMUTABLE_TASK"
chmod 700 "$IMMUTABLE_TASK"
sha256sum "$IMMUTABLE_TASK" > "$WORKER_DIR/run_task.immutable.sha256"
printf 'node=%s\nport=%s\npid=%s\nrun_root=%s\ntask_launcher_sha256=%s\n' \
  "$(hostname)" "$PORT" "$JUPYTER_PID" "$RUN_ROOT" "$(awk '{print $1}' "$WORKER_DIR/run_task.immutable.sha256")" \
  > "$WORKER_DIR/jupyter_worker.txt"
set +e
PROCGEN_RUN_ROOT="$RUN_ROOT" \
PROCGEN_ADAPTIVE_LR_MAX=0.05 \
PROCGEN_ADAPTIVE_KL_UPPER=0.02 \
bash "$IMMUTABLE_TASK" "$TASK_ID"
RC=$?
set -e
kill "$JUPYTER_PID" 2>/dev/null || true
wait "$JUPYTER_PID" 2>/dev/null || true
exit "$RC"
