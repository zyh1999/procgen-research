#!/bin/bash --login
#SBATCH --job-name=pg-rat-wcrit-cave-h
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=2
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.err

set -Eeuo pipefail

ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
CODE="$ROOT/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER=train_shared_rat_weightedcritic_reference.py
CONFIG=adv_resnet_shared_rat_weighted_critic_matched.yaml
EXPECTED_TRAINER_SHA=b4daa547ad0a25c0d6234a65bc1e9e9a10f554553fe51305b858c31c2b6878c9
EXPECTED_CONFIG_SHA=bbd4a81317d9e46e7c9116c9eac99fd7fb7fac6fde41b28375da49a058dfd773
RUN_ROOT="$ROOT/probe_1m_seed0_jupyter_rat_reference_combined_h200_v1"
ENV_NAME=caveflyer-easy-0-10
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
RUN="$RUN_ROOT/reference_rat_combined_sampled_b/$ENV_NAME/seed0"
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
PORT=$((19700 + TASK_ID))
(
  module load apps/binapps/jupyter-notebook/any
  exec jupyter notebook --no-browser --ip=127.0.0.1 --port="$PORT" --NotebookApp.port_retries=0
) > "$WORKER_DIR/jupyter.log" 2>&1 &
JUPYTER_PID=$!
printf 'node=%s\nport=%s\npid=%s\nrun=%s\n' "$(hostname)" "$PORT" "$JUPYTER_PID" "$RUN" > "$WORKER_DIR/jupyter_worker.txt"

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
  echo "TRANSITIONS=1000000"
  echo "DAMPING=0.5"
  echo "EVIDENCE_SCOPE=H200_CAUSAL_PROBE_NOT_A100_FORMAL_COMPARISON"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "COMBINED_RANDOMIZED_SCORE_SYSTEM_ROWS=512"
  echo "RAT_RHS_COLUMNS=2"
  echo "REFERENCE_KERNEL=policy_score_plus_2xi_value_score"
  echo "VALUE_SCORE_NOISE=xi_gaussian_unit"
  echo "CRITIC_METRIC=combined_randomized_score_gram"
  echo "ACTOR_RECONSTRUCTION=policy_score_only"
  echo "CRITIC_UPDATE=reference_rat_weighted_mse_backward"
  echo "ADAPTIVE_LR_INITIAL=0.004"
  echo "ADAPTIVE_LR_MAX=0.05"
  echo "ADAPTIVE_KL_UPPER=0.02"
} > "$RUN/preflight"

cmd=("$PY" -u "$TRAINER"
  --config "$CONFIG"
  --env_name "$ENV_NAME"
  --seed 0
  --device 0
  --total_timesteps 1000000
  --joint_ablation_mode rat_weighted_critic
  --joint_critic_score_mode gaussian_unit
  --joint_critic_param_scope all
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
if [[ $rc -eq 0 ]]; then echo PASS > "$RUN/status"; else echo FAILED > "$RUN/status"; fi
kill "$JUPYTER_PID" 2>/dev/null || true
wait "$JUPYTER_PID" 2>/dev/null || true
exit "$rc"
