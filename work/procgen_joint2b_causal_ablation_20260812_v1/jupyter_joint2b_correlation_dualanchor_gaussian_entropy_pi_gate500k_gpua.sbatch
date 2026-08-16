#!/bin/bash --login
#SBATCH --job-name=pg-j2b-corgpi
#SBATCH --partition=gpuA
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.err

set -Eeuo pipefail
ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
CODE="$ROOT/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER=train_shared_joint2b_rownorm_correlation_dualanchor_entropy_pi.py
CONFIG=adv_resnet_shared_joint2b_correlation_dualanchor_gaussian_entropy_pi_p01_i002_d03_lr05_500k.yaml
EXPECTED_TRAINER_SHA=756055225255d471de4d0ba42e0baa61f2058cbb97c81b959614a2a9c5193d78
EXPECTED_CONFIG_SHA=d6b59c4760566a0096dd2cde229c0a7eeb71ed6570c7a6a50c62bdb4b1b7ea1e
RUN_ROOT="$ROOT/gate_500k_seed0_jupyter_correlation_dualanchor_gaussian_entropy_pi_v1/correlation_dualanchor_gaussian_entropy_pi_p01_i002_d03"
ENV_NAMES=(bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10)
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ENV_NAME=${ENV_NAMES[$TASK_ID]}
RUN="$RUN_ROOT/$ENV_NAME/seed0"
WORKER_DIR="$ROOT/jupyter_workers/${SLURM_JOB_ID}_${TASK_ID}_correlation_dualanchor_gaussian_entropy_pi_gate500k"

mkdir -p "$(dirname "$RUN")" "$ROOT/slurm"
if ! mkdir "$RUN"; then echo "run already claimed: $RUN" >&2; exit 3; fi
mkdir -p "$WORKER_DIR/jupyter_runtime" "$WORKER_DIR/jupyter_config"
export JUPYTER_CONFIG_DIR="$WORKER_DIR/jupyter_config"
export JUPYTER_DATA_DIR="$WORKER_DIR/jupyter_data"
export JUPYTER_RUNTIME_DIR="$WORKER_DIR/jupyter_runtime"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PORT=$((37000 + SLURM_JOB_ID % 1000 + TASK_ID))
JUPYTER_PID=
cleanup() {
  if [[ -n "${JUPYTER_PID:-}" ]]; then
    kill "$JUPYTER_PID" 2>/dev/null || true
    wait "$JUPYTER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
(
  module load apps/binapps/jupyter-notebook/any
  exec jupyter notebook --no-browser --ip=127.0.0.1 --port="$PORT" --NotebookApp.port_retries=0
) > "$WORKER_DIR/jupyter.log" 2>&1 &
JUPYTER_PID=$!

trainer_sha=$(sha256sum "$CODE/$TRAINER" | awk '{print $1}')
config_sha=$(sha256sum "$CODE/configs/$CONFIG" | awk '{print $1}')
[[ "$trainer_sha" == "$EXPECTED_TRAINER_SHA" ]] || { echo "trainer SHA mismatch: $trainer_sha" >&2; exit 4; }
[[ "$config_sha" == "$EXPECTED_CONFIG_SHA" ]] || { echo "config SHA mismatch: $config_sha" >&2; exit 5; }
{
  echo "ENV_NAME=$ENV_NAME"
  echo "SEED=0"
  echo "TRAINER_SHA256=$trainer_sha"
  echo "CONFIG_SHA256=$config_sha"
  echo "ROLLOUT=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=500000"
  echo "JOINT_SYSTEM_ROWS=1024"
  echo "JOINT_MODE=full_joint_gaussian_unit_all"
  echo "JOINT_CRITIC_SCORE_MODE=gaussian_unit"
  echo "JOINT_RHS_MODE=paired_score_residual"
  echo "JOINT_RECONSTRUCTION_MODE=full_joint"
  echo "JOINT_BLOCK_NORMALIZATION=none"
  echo "JOINT_DAMPING_MODE=correlation_relative"
  echo "NORMALIZED_DAMPING=0.30"
  echo "CORRELATION_NORMALIZED_SOLVE=true"
  echo "CORRELATION_ROW_FLOOR_MODE=per_block_median_relative"
  echo "CORRELATION_ROW_FLOOR_FRACTION=0.01"
  echo "CORRELATION_ACTOR_FISHER_FLOOR=0.50"
  echo "CORRELATION_ACTOR_FISHER_SCALE=sqrt_min_1_F_over_0.50"
  echo "CORRELATION_ACTOR_KERNEL_ANCHOR_MODE=running_highwater_relative"
  echo "CORRELATION_ACTOR_KERNEL_ANCHOR_FRACTION=0.001"
  echo "ENTROPY_RHS_TARGET=2.50"
  echo "ENTROPY_RHS_GAIN=0.10"
  echo "ENTROPY_RHS_INTEGRAL_GAIN=0.02"
  echo "ENTROPY_RHS_MAX_COEF=0.50"
  echo "ENTROPY_INTEGRAL_UPDATE=once_per_full_rollout_fixed_behavior_logits"
  echo "CRITIC_OBJECTIVE_COEF=1.0"
  echo "CRITIC_CURVATURE_COEF=1.0"
  echo "LR_INITIAL_MAX=0.05"
  echo "KL_RANGE=0.005,0.02"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "LINEAR_SOLVE_DTYPE=float64"
  echo "GATE_TRANSITIONS=500000"
  echo "CAUSAL_REASON=test_paired_gaussian_row_reweighting_after_scale_and_entropy_fixes"
} > "$RUN/preflight"

cmd=("$PY" -u "$TRAINER" --config "$CONFIG"
  --env_name "$ENV_NAME" --seed 0 --device 0
  --total_timesteps 500000 --joint_ablation_mode full_joint
  --joint_critic_score_mode gaussian_unit --joint_critic_param_scope all
  --joint_critic_reconstruction_scope all
  --joint_critic_curvature_coef 1.0 --joint_critic_objective_coef 1.0)
printf '%q ' "${cmd[@]}" > "$RUN/command.txt"; printf '\n' >> "$RUN/command.txt"
echo RUNNING > "$RUN/status"
cd "$CODE"
export PROCGEN_METRIC_TRACE_PATH="$RUN/metric_trace.jsonl"
set +e
"${cmd[@]}" > "$RUN/stdout" 2> "$RUN/stderr"
rc=$?
set -e
echo "$rc" > "$RUN/rc"
if [[ $rc -eq 0 ]]; then echo PASS > "$RUN/status"; else echo FAILED > "$RUN/status"; fi
exit "$rc"
