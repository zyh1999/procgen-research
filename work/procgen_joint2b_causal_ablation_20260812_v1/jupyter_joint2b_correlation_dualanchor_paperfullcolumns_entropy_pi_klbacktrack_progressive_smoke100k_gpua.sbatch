#!/bin/bash --login
#SBATCH --job-name=pg-j2b-papfull
#SBATCH --partition=gpuA
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.err

set -Eeuo pipefail
ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
CODE="$ROOT/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER=train_shared_joint2b_rownorm_correlation_dualanchor_entropy_pi_klbacktrack_progressive_fullcolumns.py
CONFIG=adv_resnet_shared_joint2b_correlation_dualanchor_paperfullcolumns_entropy_pi_klbacktrack_progressive_p01_i002_d03_lr05_100k.yaml
EXPECTED_TRAINER_SHA=e2a2d92af2aaa8efe15f305f1c8caa18601d5986d4cf5599dd833e935c894387
EXPECTED_CONFIG_SHA=025411278862c9b1ddb2892ec40541ecb2ce0f49885db45d08a15a453089c977
TOTAL_TIMESTEPS=100000
RUN_ROOT="$ROOT/smoke_100k_seed0_jupyter_correlation_dualanchor_paperfullcolumns_entropy_pi_klbacktrack_progressive_v1/paperfullcolumns_pi_klbacktrack_u005_progressive"
ENV_NAMES=(bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10)
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ENV_NAME=${ENV_NAMES[$TASK_ID]}
RUN="$RUN_ROOT/$ENV_NAME/seed0"
WORKER_DIR="$ROOT/jupyter_workers/${SLURM_JOB_ID}_${TASK_ID}_paperfullcolumns_pi_klbacktrack_progressive_100000"

mkdir -p "$(dirname "$RUN")" "$ROOT/slurm"
if ! mkdir "$RUN"; then echo "run already claimed: $RUN" >&2; exit 3; fi
mkdir -p "$WORKER_DIR/jupyter_runtime" "$WORKER_DIR/jupyter_config"
export JUPYTER_CONFIG_DIR="$WORKER_DIR/jupyter_config"
export JUPYTER_DATA_DIR="$WORKER_DIR/jupyter_data"
export JUPYTER_RUNTIME_DIR="$WORKER_DIR/jupyter_runtime"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PORT=$((39000 + SLURM_JOB_ID % 1000 + TASK_ID))
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
  echo "TRANSITIONS=$TOTAL_TIMESTEPS"
  echo "JOINT_SYSTEM_ROWS=1024"
  echo "JOINT_MODE=full_joint_clean_all"
  echo "JOINT_RHS_MODE=paper_weight_residual_reconstruct"
  echo "JOINT_RECONSTRUCTION_MODE=paper_full_joint_columns"
  echo "JOINT_RHS_COLUMNS=2"
  echo "JOINT_RECONSTRUCTION_ROWS=all_actor_and_critic_rows_per_rhs_column"
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
  echo "BEHAVIOR_KL_GUARD_ENABLED=true"
  echo "BEHAVIOR_KL_GUARD_UPPER=0.005"
  echo "BEHAVIOR_KL_GUARD_BACKOFF=0.50"
  echo "BEHAVIOR_KL_GUARD_MODE=sqrt_backtrack_accept_progressive_rollout_lr"
  echo "BEHAVIOR_KL_GUARD_SAFETY=0.90"
  echo "BEHAVIOR_KL_GUARD_MAX_BACKTRACKS=8"
  echo "BEHAVIOR_KL_GUARD_REFERENCE=full_4096_rollout_chunked_512"
  echo "BEHAVIOR_KL_GUARD_SCOPE=every_minibatch_exact_full_rollout_KL_with_linear_32_step_budget"
  echo "BEHAVIOR_KL_GUARD_ACTION=backtrack_exact_delta_with_progressive_budget_without_persistent_LR_decay"
  echo "BEHAVIOR_KL_GUARD_PERSIST_LR_AFTER_BACKTRACK=false"
  echo "BEHAVIOR_KL_GUARD_PROGRESSIVE_BUDGET=true"
  echo "BEHAVIOR_KL_GUARD_BUDGET_STEPS=32"
  echo "CRITIC_OBJECTIVE_COEF=1.0"
  echo "CRITIC_CURVATURE_COEF=1.0"
  echo "LR_INITIAL_MAX=0.05"
  echo "KL_RANGE=0.005,0.02"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "LINEAR_SOLVE_DTYPE=float64"
  echo "GATE_TRANSITIONS=$TOTAL_TIMESTEPS"
  echo "CAUSAL_REASON=restore_all_joint_rows_for_each_rhs_column_to_preserve_shared_trunk_cross_block_preconditioning"
} > "$RUN/preflight"

cmd=("$PY" -u "$TRAINER" --config "$CONFIG"
  --env_name "$ENV_NAME" --seed 0 --device 0
  --total_timesteps "$TOTAL_TIMESTEPS" --joint_ablation_mode full_joint
  --joint_critic_score_mode clean --joint_critic_param_scope all
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
