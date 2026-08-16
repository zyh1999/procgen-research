#!/bin/bash --login
#SBATCH --job-name=pg-j2b-diag03-h
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%1
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.err

set -Eeuo pipefail

ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
CODE="$ROOT/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER=train_shared_joint2b_relative_damping.py
CONFIG=adv_resnet_shared_joint2b_diagonal_damping_03.yaml
EXPECTED_TRAINER_SHA=456bb8deabb8e4f23579bc11492dd14c9183b4df6c14d57c674dc6c199456b79
EXPECTED_CONFIG_SHA=97114c816e28673635ed3755e9f0d5d4c5d24b0eb953db7d6af81fc31aff6a3c
ENV_NAMES=(bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10)
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ENV_NAME=${ENV_NAMES[$TASK_ID]}
RUN_ROOT="$ROOT/gate_1m_seed0_jupyter_diagonal_damping_v1"
RUN="$RUN_ROOT/diag03/$ENV_NAME/seed0"
WORKER_DIR="$ROOT/jupyter_workers/${SLURM_JOB_ID}_${TASK_ID}"

mkdir -p "$(dirname "$RUN")" "$ROOT/slurm"
if ! mkdir "$RUN"; then
  echo "run already claimed: $RUN" >&2
  exit 3
fi
mkdir -p "$WORKER_DIR/jupyter_runtime" "$WORKER_DIR/jupyter_config"
export JUPYTER_CONFIG_DIR="$WORKER_DIR/jupyter_config"
export JUPYTER_DATA_DIR="$WORKER_DIR/jupyter_data"
export JUPYTER_RUNTIME_DIR="$WORKER_DIR/jupyter_runtime"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PORT=$((19700 + TASK_ID))
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
  echo "VARIANT=diag03"
  echo "TRAINER_SHA256=$trainer_sha"
  echo "CONFIG_SHA256=$config_sha"
  echo "ROLLOUT=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=1000000"
  echo "BASE_DAMPING=0.5"
  echo "JOINT_DAMPING_MODE=diagonal_relative"
  echo "JOINT_DIAGONAL_DAMPING_FLOOR=0.3"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "JOINT_SYSTEM_ROWS=1024"
  echo "JOINT_MODE=full_joint_clean_all"
  echo "ADAPTIVE_LR_INITIAL=0.004"
  echo "ADAPTIVE_LR_MAX=0.05"
  echo "ADAPTIVE_KL_LOWER=0.005"
  echo "ADAPTIVE_KL_UPPER=0.04"
  echo "HARDWARE=H200_CAUSAL_GATE"
} > "$RUN/preflight"

cmd=("$PY" -u "$TRAINER" --config "$CONFIG" --env_name "$ENV_NAME" --seed 0 --device 0 --total_timesteps 1000000 --joint_ablation_mode full_joint --joint_critic_score_mode clean --joint_critic_param_scope all --joint_critic_reconstruction_scope all --joint_critic_curvature_coef 1.0 --joint_critic_objective_coef 1.0)
printf '%q ' "${cmd[@]}" > "$RUN/command.txt"
printf '\n' >> "$RUN/command.txt"
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
