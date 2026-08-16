#!/usr/bin/env bash
set -euo pipefail

ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
CODE="$ROOT/code"
# Allow an isolated replacement campaign without ever colliding with an
# existing formal run directory.  The default preserves the original layout.
RUN_ROOT="${PROCGEN_RUN_ROOT:-$ROOT/formal_1m_seed0_v2}"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
# An isolated campaign may replace only the trainer implementation while
# retaining the complete task-to-variant mapping and preflight record.  The
# default keeps every existing submitted launcher byte-for-byte equivalent in
# algorithm selection.
TRAINER="${PROCGEN_TRAINER:-train_shared_joint2b_causal_ablation.py}"
CONFIG=adv_resnet_shared_joint2b_causal.yaml

TASK_ID=${1:?task id 0-27 is required}
if (( TASK_ID < 0 || TASK_ID >= 28 )); then
  echo "task id must be in [0, 27], got $TASK_ID" >&2
  exit 2
fi

VARIANTS=(
  actor_only
  curvature_only
  full_clean
  full_head
  full_rademacher
  full_gaussian
  full_clean_conservative
)
ENVS=(
  bigfish-easy-0-10
  bossfight-easy-0-10
  caveflyer-easy-0-10
  coinrun-easy-0-10
)

variant_index=$((TASK_ID / 4))
env_index=$((TASK_ID % 4))
variant=${VARIANTS[$variant_index]}
env_name=${ENVS[$env_index]}
seed=0

mode=full_joint
score=clean
scope=all
lambda_c=1.0
c_c=1.0
lr_max=0.5
kl_upper=0.04
case "$variant" in
  actor_only)
    mode=actor_only
    lambda_c=0.0
    c_c=0.0
    ;;
  curvature_only)
    mode=curvature_only
    c_c=0.0
    ;;
  full_clean)
    ;;
  full_head)
    scope=head_only
    ;;
  full_rademacher)
    score=rademacher
    ;;
  full_gaussian)
    score=gaussian_unit
    ;;
  full_clean_conservative)
    lr_max=0.05
    kl_upper=0.02
    ;;
  *)
    echo "unknown variant: $variant" >&2
    exit 2
    ;;
esac

# A launcher may intentionally hold the algorithmic variant fixed while
# comparing controller settings.  Keep the normal task mapping as the default,
# but make explicit overrides auditable in command.txt and preflight.txt.
if [[ -n "${PROCGEN_ADAPTIVE_LR_MAX:-}" ]]; then
  lr_max="$PROCGEN_ADAPTIVE_LR_MAX"
fi
if [[ -n "${PROCGEN_ADAPTIVE_KL_UPPER:-}" ]]; then
  kl_upper="$PROCGEN_ADAPTIVE_KL_UPPER"
fi

run="$RUN_ROOT/$variant/$env_name/seed$seed"
mkdir -p "$(dirname "$run")"
if ! mkdir "$run" 2>/dev/null; then
  echo "run already claimed: $run" >&2
  exit 3
fi

trainer_sha=$(sha256sum "$CODE/$TRAINER" | awk '{print $1}')
config_sha=$(sha256sum "$CODE/configs/$CONFIG" | awk '{print $1}')
{
  date --iso-8601=seconds
  hostname
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "VARIANT=$variant"
  echo "ENV_NAME=$env_name"
  echo "SEED=$seed"
  echo "MODE=$mode"
  echo "SCORE_MODE=$score"
  echo "CRITIC_PARAM_SCOPE=$scope"
  echo "LAMBDA_C=$lambda_c"
  echo "C_C=$c_c"
  echo "LR_MAX=$lr_max"
  echo "KL_UPPER=$kl_upper"
  echo "ROLLOUT_BATCH=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=1000000"
  echo "DAMPING=0.5"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "TRAINER_SHA256=$trainer_sha"
  echo "TRAINER=$TRAINER"
  echo "CONFIG_SHA256=$config_sha"
} > "$run/preflight.txt"

cd "$CODE"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0
export PROCGEN_METRIC_TRACE_PATH="$run/metric_trace.jsonl"

cmd=(
  "$PY" -u "$TRAINER"
  --config "$CONFIG"
  --env_name "$env_name"
  --seed "$seed"
  --device 0
  --total_timesteps 1000000
  --joint_ablation_mode "$mode"
  --joint_critic_score_mode "$score"
  --joint_critic_param_scope "$scope"
  --joint_critic_curvature_coef "$lambda_c"
  --joint_critic_objective_coef "$c_c"
  --adaptive_lr_max "$lr_max"
  --adaptive_kl_upper "$kl_upper"
)
printf '%q ' "${cmd[@]}" > "$run/command.txt"
printf '\n' >> "$run/command.txt"

echo RUNNING > "$run/status"
set +e
"${cmd[@]}" > "$run/stdout" 2> "$run/stderr"
rc=$?
set -e
echo "$rc" > "$run/rc"
if (( rc == 0 )); then
  echo PASS > "$run/status"
else
  echo FAIL > "$run/status"
fi
exit "$rc"
