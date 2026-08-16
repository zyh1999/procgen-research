#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID=${1:?usage: run_4090_diagonal_damping_gate.sh GPU_ID ENV_NAME}
ENV_NAME=${2:?usage: run_4090_diagonal_damping_gate.sh GPU_ID ENV_NAME}
ROOT=${PROCGEN_GATE_ROOT:-/home/yihe/procgen_joint2b_diagdamp_gate_20260814_v1}
PY=${PROCGEN_GATE_PY:?set PROCGEN_GATE_PY to the validated interpreter}
CODE="$ROOT/code"
TRAINER=train_shared_joint2b_relative_damping.py
TRAINER_SHA=456bb8deabb8e4f23579bc11492dd14c9183b4df6c14d57c674dc6c199456b79
CONFIGS=(
  adv_resnet_shared_joint2b_diagonal_damping_01.yaml
  adv_resnet_shared_joint2b_diagonal_damping_03.yaml
)
CONFIG_SHAS=(
  732f2d55d6ef3e30cf0a913f0a480411b9cb6e86c8240a1a2897cb478343f28c
  97114c816e28673635ed3755e9f0d5d4c5d24b0eb953db7d6af81fc31aff6a3c
)
VARIANTS=(diag01 diag03)
FLOORS=(0.1 0.3)

mkdir -p "$ROOT/controllers" "$ROOT/results"
controller="$ROOT/controllers/gpu${GPU_ID}_${ENV_NAME}"
printf 'pid=%s\ngpu=%s\nenv=%s\nstarted=%s\n' "$$" "$GPU_ID" "$ENV_NAME" "$(date -Is)" > "$controller.state"
trap 'printf "finished=%s\nrc=%s\n" "$(date -Is)" "$?" >> "$controller.state"' EXIT

[[ "$(sha256sum "$CODE/$TRAINER" | awk '{print $1}')" == "$TRAINER_SHA" ]]
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for variant_id in 0 1; do
  config=${CONFIGS[$variant_id]}
  config_sha=${CONFIG_SHAS[$variant_id]}
  variant=${VARIANTS[$variant_id]}
  floor=${FLOORS[$variant_id]}
  [[ "$(sha256sum "$CODE/configs/$config" | awk '{print $1}')" == "$config_sha" ]]
  run="$ROOT/results/$variant/$ENV_NAME/seed0"
  mkdir -p "$(dirname "$run")"
  if ! mkdir "$run"; then
    echo "run already claimed: $run" >&2
    exit 3
  fi
  {
    echo "ENV_NAME=$ENV_NAME"
    echo "SEED=0"
    echo "VARIANT=$variant"
    echo "GPU_ID=$GPU_ID"
    echo "HOST=$(hostname)"
    echo "PYTHON=$PY"
    echo "TRAINER_SHA256=$TRAINER_SHA"
    echo "CONFIG_SHA256=$config_sha"
    echo "ROLLOUT=4096"
    echo "MINIBATCH=512"
    echo "EPOCHS=4"
    echo "TRANSITIONS=1000000"
    echo "BASE_DAMPING=0.5"
    echo "JOINT_DAMPING_MODE=diagonal_relative"
    echo "JOINT_DIAGONAL_DAMPING_FLOOR=$floor"
    echo "MOMENTUM=0"
    echo "KACZMARZ=false"
    echo "JOINT_SYSTEM_ROWS=1024"
    echo "JOINT_MODE=full_joint_clean_all"
    echo "ADAPTIVE_LR_INITIAL=0.004"
    echo "ADAPTIVE_LR_MAX=0.05"
    echo "ADAPTIVE_KL_LOWER=0.005"
    echo "ADAPTIVE_KL_UPPER=0.04"
  } > "$run/preflight"
  cmd=("$PY" -u "$TRAINER" --config "$config" --env_name "$ENV_NAME" --seed 0 --device 0 --total_timesteps 1000000 --joint_ablation_mode full_joint --joint_critic_score_mode clean --joint_critic_param_scope all --joint_critic_reconstruction_scope all --joint_critic_curvature_coef 1.0 --joint_critic_objective_coef 1.0)
  printf '%q ' "${cmd[@]}" > "$run/command.txt"
  printf '\n' >> "$run/command.txt"
  echo RUNNING > "$run/status"
  cd "$CODE"
  export PROCGEN_METRIC_TRACE_PATH="$run/metric_trace.jsonl"
  set +e
  "${cmd[@]}" > "$run/stdout" 2> "$run/stderr"
  rc=$?
  set -e
  echo "$rc" > "$run/rc"
  if [[ $rc -eq 0 ]]; then
    echo PASS > "$run/status"
  else
    echo FAILED > "$run/status"
    exit "$rc"
  fi
done
