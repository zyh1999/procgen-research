#!/bin/bash --login
#SBATCH --job-name=pg-j2b-acguard01
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.err

set -Eeuo pipefail

ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
CODE="$ROOT/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER=train_shared_joint2b_actor_critic_guard.py
CONFIG=adv_resnet_shared_joint2b_actorcriticguard01_lr02.yaml
EXPECTED_TRAINER_SHA=58e73af1dde57ae34250167a28b14c30a508f2ffc5781fdd673791b3b46bb962
EXPECTED_CONFIG_SHA=3a24a057bdb6898e0ca3e6153eddfc7d6272700f5df083f345d84c9f940ffdb0
ENV_NAMES=(
  bigfish-easy-0-10
  bossfight-easy-0-10
  caveflyer-easy-0-10
  coinrun-easy-0-10
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ENV_NAME=${ENV_NAMES[$TASK_ID]}
RUN_ROOT=${J2B_RUN_ROOT:-"$ROOT/gate_3m_seed0_jupyter_actorcriticguard01_v1"}
RUN="$RUN_ROOT/acguard01/$ENV_NAME/seed0"
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
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/vendor_site:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PORT=$((19800 + TASK_ID))
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
  echo "VARIANT=acguard01"
  echo "TRAINER_SHA256=$trainer_sha"
  echo "CONFIG_SHA256=$config_sha"
  echo "ROLLOUT=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=3000000"
  echo "BASE_DAMPING=0.5"
  echo "JOINT_DAMPING_MODE=block_median_floor"
  echo "JOINT_DAMPING_TO_MEDIAN_FLOOR=0.1"
  echo "ACTOR_DAMPING_FROM_CRITIC_FLOOR=0.01"
  echo "ADAPTIVE_LR_INITIAL=0.004"
  echo "ADAPTIVE_LR_MAX=0.02"
  echo "ADAPTIVE_KL_LOWER=0.005"
  echo "ADAPTIVE_KL_UPPER=0.04"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "JOINT_SYSTEM_ROWS=1024"
  echo "JOINT_MODE=full_joint_clean_all"
  echo "CRITIC_OBJECTIVE_COEF=1.0"
  echo "CRITIC_CURVATURE_COEF=1.0"
  echo "LINEAR_SOLVE_DTYPE=float64"
  echo "PROCGEN_IMPORT_ROOT=$ROOT/vendor_site"
} > "$RUN/preflight"

cmd=("$PY" -u "$TRAINER"
  --config "$CONFIG"
  --env_name "$ENV_NAME"
  --seed 0
  --device 0
  --total_timesteps 3000000
  --joint_ablation_mode full_joint
  --joint_critic_score_mode clean
  --joint_critic_param_scope all
  --joint_critic_reconstruction_scope all
  --joint_critic_curvature_coef 1.0
  --joint_critic_objective_coef 1.0)
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
if [[ $rc -eq 0 ]]; then
  echo PASS > "$RUN/status"
else
  echo FAILED > "$RUN/status"
fi
exit "$rc"
