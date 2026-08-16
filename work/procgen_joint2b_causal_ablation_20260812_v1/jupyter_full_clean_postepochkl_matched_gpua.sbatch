#!/bin/bash --login
# Deferred causal control: identical full-clean strict 2B system, but the
# adaptive controller observes KL over the full fixed rollout after all four
# epochs rather than only the final minibatch.  This is intentionally not
# submitted until the head-only shared-trunk control has answered its branch.
#SBATCH --job-name=pg-j2b-postkl
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
RUN_ROOT="$ROOT/formal_1m_seed0_jupyter_full_clean_postepochkl_v1"
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
WORKER_DIR="$ROOT/jupyter_workers/${SLURM_JOB_ID}_${TASK_ID}"
mkdir -p "$WORKER_DIR/jupyter_runtime" "$WORKER_DIR/jupyter_config"
export JUPYTER_RUNTIME_DIR="$WORKER_DIR/jupyter_runtime"
export JUPYTER_CONFIG_DIR="$WORKER_DIR/jupyter_config"
PORT=$((18580 + TASK_ID))
(
  module load apps/binapps/jupyter-notebook/any
  exec jupyter notebook --no-browser --ip=127.0.0.1 --port="$PORT" --NotebookApp.port_retries=0
) >"$WORKER_DIR/jupyter.log" 2>&1 &
JUPYTER_PID=$!
printf 'node=%s\nport=%s\npid=%s\nrun_root=%s\n' "$(hostname)" "$PORT" "$JUPYTER_PID" "$RUN_ROOT" > "$WORKER_DIR/jupyter_worker.txt"
# Snapshot the shared task mapping before spawning the trainer.  This avoids
# the historical failure mode where replacing a shared shell source while its
# child process was running caused bash to parse a truncated tail at exit.
IMMUTABLE_TASK="$WORKER_DIR/run_task.immutable.sh"
cp "$ROOT/run_task.sh" "$IMMUTABLE_TASK"
chmod 700 "$IMMUTABLE_TASK"
sha256sum "$IMMUTABLE_TASK" > "$WORKER_DIR/run_task.immutable.sha256"
set +e
PROCGEN_RUN_ROOT="$RUN_ROOT" \
PROCGEN_TRAINER=train_shared_joint2b_causal_ablation_postepochkl.py \
PROCGEN_ADAPTIVE_LR_MAX=0.05 \
PROCGEN_ADAPTIVE_KL_UPPER=0.02 \
bash "$IMMUTABLE_TASK" "$TASK_ID"
RC=$?
set -e
kill "$JUPYTER_PID" 2>/dev/null || true
wait "$JUPYTER_PID" 2>/dev/null || true
exit "$RC"
