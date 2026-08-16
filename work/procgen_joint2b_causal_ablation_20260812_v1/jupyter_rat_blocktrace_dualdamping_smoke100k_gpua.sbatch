#!/bin/bash --login
#SBATCH --job-name=pg-rat-btrace
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
TRAINER=train_shared_rat_blocktrace_dualdamping.py
CONFIG=adv_resnet_shared_rat_blocktrace_dualdamping_100k.yaml
EXPECTED_TRAINER_SHA=1881bf7c3fe3f8d29ded23e25976810ab9127d9bc125d9c89332aa39c1ab61dc
EXPECTED_CONFIG_SHA=a77a94e2c003016f17614ba70fd1a5dcc05a2c12baccf2b7ebb88fda239415bc
TOTAL_TIMESTEPS=100000
RUN_ROOT="$ROOT/smoke_100k_seed0_jupyter_rat_blocktrace_v1/deterministic_task_blocktrace_b_dualdamping"
ENV_NAMES=(bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10)
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ENV_NAME=${ENV_NAMES[$TASK_ID]}
RUN="$RUN_ROOT/$ENV_NAME/seed0"
WORKER_DIR="$ROOT/jupyter_workers/${SLURM_JOB_ID}_${TASK_ID}_rat_blocktrace_100000"

mkdir -p "$(dirname "$RUN")" "$ROOT/slurm"
if ! mkdir "$RUN"; then echo "run already claimed: $RUN" >&2; exit 3; fi
mkdir -p "$WORKER_DIR/jupyter_runtime" "$WORKER_DIR/jupyter_config"
export JUPYTER_CONFIG_DIR="$WORKER_DIR/jupyter_config"
export JUPYTER_DATA_DIR="$WORKER_DIR/jupyter_data"
export JUPYTER_RUNTIME_DIR="$WORKER_DIR/jupyter_runtime"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PORT=$((43000 + SLURM_JOB_ID % 1000 + TASK_ID))
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
[[ "$trainer_sha" == "$EXPECTED_TRAINER_SHA" ]]
[[ "$config_sha" == "$EXPECTED_CONFIG_SHA" ]]
{
  echo "ENV_NAME=$ENV_NAME"
  echo "SEED=0"
  echo "TRAINER_SHA256=$trainer_sha"
  echo "CONFIG_SHA256=$config_sha"
  echo "ROLLOUT=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=$TOTAL_TIMESTEPS"
  echo "JOINT_SYSTEM_ROWS=512"
  echo "JOINT_KERNEL=Hpi_HpiT_over_B_plus_4_Jv_JvT_over_B"
  echo "COMPRESSION=task_block_trace_of_strict_2B_operator"
  echo "RANDOM_SCORE_DRAWS=0"
  echo "EXPECTED_CROSS_BLOCK=0"
  echo "ACTOR_DAMPING=0.03"
  echo "CRITIC_DAMPING=0.5"
  echo "ADAPTIVE_LR_INITIAL=0.004"
  echo "ADAPTIVE_LR_MAX=0.02"
  echo "KL_RANGE=0.005,0.04"
  echo "CRITIC_UPDATE=reference_rat_weighted_mse_backward"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "LINEAR_SOLVE_DTYPE=float64"
} > "$RUN/preflight"

cmd=("$PY" -u "$TRAINER" --config "$CONFIG"
  --env_name "$ENV_NAME" --seed 0 --device 0
  --total_timesteps "$TOTAL_TIMESTEPS"
  --joint_ablation_mode rat_blocktrace_critic
  --joint_critic_score_mode clean --joint_critic_param_scope all
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
