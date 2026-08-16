#!/bin/bash --login
# Capacity-matched separate-backbone control.  Actor and critic each retain
# the enlarged ResNet + 256->1024->256 decision MLP, c_C=lambda_C=1, and every
# matched controller setting.  Disjoint parameter columns make the cross Gram
# block exactly zero while preserving a full-capacity critic.
# This is deliberately isolated from run_task.sh so queued causal controls keep
# their immutable launch mapping.
#SBATCH --job-name=pg-j2b-sep
#SBATCH --partition=gpuA
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.err

set -euo pipefail
ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
CODE="$ROOT/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER=train_separate_matched_joint2b_causal.py
CONFIG=adv_resnet_shared_joint2b_causal.yaml
RUN_ROOT="$ROOT/formal_1m_seed0_jupyter_separate_matched_v1"
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ENV_NAMES=(bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10)
ENV_NAME=${ENV_NAMES[$TASK_ID]}
RUN="$RUN_ROOT/separate_capacity_matched/$ENV_NAME/seed0"
mkdir -p "$(dirname "$RUN")"
if ! mkdir "$RUN"; then
  echo "run already claimed: $RUN" >&2
  exit 3
fi

WORKER_DIR="$ROOT/jupyter_workers/${SLURM_JOB_ID}_${TASK_ID}"
mkdir -p "$WORKER_DIR/jupyter_runtime" "$WORKER_DIR/jupyter_config"
export JUPYTER_RUNTIME_DIR="$WORKER_DIR/jupyter_runtime"
export JUPYTER_CONFIG_DIR="$WORKER_DIR/jupyter_config"
PORT=$((18400 + TASK_ID))
(
  module load apps/binapps/jupyter-notebook/any
  exec jupyter notebook --no-browser --ip=127.0.0.1 --port="$PORT" --NotebookApp.port_retries=0
) >"$WORKER_DIR/jupyter.log" 2>&1 &
JUPYTER_PID=$!
printf 'node=%s\nport=%s\npid=%s\nrun=%s\n' "$(hostname)" "$PORT" "$JUPYTER_PID" "$RUN" > "$WORKER_DIR/jupyter_worker.txt"

trainer_sha=$(sha256sum "$CODE/$TRAINER" | awk '{print $1}')
config_sha=$(sha256sum "$CODE/configs/$CONFIG" | awk '{print $1}')
{
  date --iso-8601=seconds
  hostname
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "SLURM_ARRAY_TASK_ID=$TASK_ID"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "VARIANT=separate_capacity_matched"
  echo "ACTOR_CRITIC_ARCHITECTURE=separate_capacity_matched"
  echo "ENV_NAME=$ENV_NAME"
  echo "SEED=0"
  echo "MODE=full_joint"
  echo "SCORE_MODE=clean"
  echo "CRITIC_PARAM_SCOPE=all"
  echo "CRITIC_RECONSTRUCTION_SCOPE=all"
  echo "EXPECTED_CROSS_BLOCK=exact_zero"
  echo "LAMBDA_C=1.0"
  echo "C_C=1.0"
  echo "LR_MAX=0.05"
  echo "KL_UPPER=0.02"
  echo "ROLLOUT_BATCH=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=1000000"
  echo "DAMPING=0.5"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "TRAINER_SHA256=$trainer_sha"
  echo "CONFIG_SHA256=$config_sha"
} > "$RUN/preflight.txt"

cd "$CODE"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0
export PROCGEN_METRIC_TRACE_PATH="$RUN/metric_trace.jsonl"
cmd=("$PY" -u "$TRAINER" --config "$CONFIG" --env_name "$ENV_NAME" --seed 0 --device 0 --total_timesteps 1000000 --joint_ablation_mode full_joint --joint_critic_score_mode clean --joint_critic_param_scope all --joint_critic_reconstruction_scope all --joint_critic_curvature_coef 1.0 --joint_critic_objective_coef 1.0 --adaptive_lr_max 0.05 --adaptive_kl_upper 0.02)
printf '%q ' "${cmd[@]}" > "$RUN/command.txt"
printf '\n' >> "$RUN/command.txt"
echo RUNNING > "$RUN/status"
set +e
"${cmd[@]}" > "$RUN/stdout" 2> "$RUN/stderr"
RC=$?
set -e
echo "$RC" > "$RUN/rc"
if (( RC == 0 )); then echo PASS > "$RUN/status"; else echo FAIL > "$RUN/status"; fi
kill "$JUPYTER_PID" 2>/dev/null || true
wait "$JUPYTER_PID" 2>/dev/null || true
exit "$RC"
