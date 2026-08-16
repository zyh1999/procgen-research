#!/bin/bash --login
#SBATCH --job-name=pg-j2b-sspv3
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

set -Eeuo pipefail

ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
CODE="$ROOT/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER=train_shared_joint2b_spectral_control_v3.py
CONFIG=adv_resnet_shared_joint2b_causal.yaml
EXPECTED_TRAINER_SHA=6deda39584d8570e045ec784983695acb45a3d6fe68120a645ffca81693fe8be
RUN_ROOT="$ROOT/formal_1m_seed0_jupyter_full_clean_spectral_v3"
ENV_NAMES=(bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10)
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ENV_NAME=${ENV_NAMES[$TASK_ID]}
RUN="$RUN_ROOT/full_clean_spectral/$ENV_NAME/seed0"
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
PORT=$((18800 + TASK_ID))
(
  module load apps/binapps/jupyter-notebook/any
  exec jupyter notebook --no-browser --ip=127.0.0.1 --port="$PORT" --NotebookApp.port_retries=0
) > "$WORKER_DIR/jupyter.log" 2>&1 &
JUPYTER_PID=$!
printf 'node=%s\nport=%s\npid=%s\nrun=%s\n' "$(hostname)" "$PORT" "$JUPYTER_PID" "$RUN" > "$WORKER_DIR/jupyter_worker.txt"

trainer_sha=$(sha256sum "$CODE/$TRAINER" | awk '{print $1}')
config_sha=$(sha256sum "$CODE/configs/$CONFIG" | awk '{print $1}')
if [[ "$trainer_sha" != "$EXPECTED_TRAINER_SHA" ]]; then
  echo "trainer SHA mismatch: $trainer_sha" >&2
  exit 4
fi
{
  echo "ENV_NAME=$ENV_NAME"
  echo "SEED=0"
  echo "TRAINER_SHA256=$trainer_sha"
  echo "CONFIG_SHA256=$config_sha"
  echo "ROLLOUT=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=1000000"
  echo "DAMPING=0.5"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "JOINT_SYSTEM_ROWS=1024"
  echo "SPECTRAL_LOGGING_ONLY=1"
  echo "COMPONENT_POLICY_FISHER_DECOMPOSITION=1"
  echo "ACTOR_ONLY_SAME_MINIBATCH_COUNTERFACTUAL=1"
} > "$RUN/preflight"

cmd=("$PY" -u "$TRAINER"
  --config "$CONFIG"
  --env_name "$ENV_NAME"
  --seed 0
  --device 0
  --total_timesteps 1000000
  --joint_ablation_mode full_joint
  --joint_critic_score_mode clean
  --joint_critic_param_scope all
  --joint_critic_reconstruction_scope all
  --joint_critic_curvature_coef 1.0
  --joint_critic_objective_coef 1.0
  --adaptive_lr_max 0.05
  --adaptive_kl_upper 0.02)
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
kill "$JUPYTER_PID" 2>/dev/null || true
wait "$JUPYTER_PID" 2>/dev/null || true
exit "$rc"
