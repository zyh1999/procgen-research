#!/usr/bin/env bash
set -euo pipefail

ROOT=/scratch/h99859yz/procgen_det2b_rhscoef_gpuh_20260811_v1
CODE="$ROOT/code"
RUN_ROOT="$ROOT/formal"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER=train_shared_rat_exact_deterministic_ggn_symfp64.py
CONFIG=adv_resnet_shared_det2b_rhsonly_symfp64.yaml
TRAINER_SHA=2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b
CONFIG_SHA=23a00a9307655475d76e6832cb7c8dd9a3f658924eb83f51bd07779ec3c5fa8d

CC_VALUES=(0.03 0.1 0.3 1.0)
CC_TAGS=(0p03 0p1 0p3 1p0)
ENV_NAME=bigfish-easy-0-10
SEED=0

[[ "$(sha256sum "$CODE/$TRAINER" | awk '{print $1}')" == "$TRAINER_SHA" ]]
[[ "$(sha256sum "$CODE/configs/$CONFIG" | awk '{print $1}')" == "$CONFIG_SHA" ]]
grep -q 'joint_H = torch.cat(\[H_pi, critic_h_weight \* J_v\], dim=0)' "$CODE/$TRAINER"
grep -q 'joint_system_rows=2 \* num_sa' "$CODE/$TRAINER"
grep -q 'joint_critic_curvature_coef: 1.0' "$CODE/configs/$CONFIG"

mkdir -p "$RUN_ROOT"
cd "$CODE"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0

pids=()
tags=()
if [[ "$#" -eq 0 ]]; then
  indices=(0 1 2 3)
else
  indices=("$@")
fi

for i in "${indices[@]}"; do
  if [[ ! "$i" =~ ^[0-3]$ ]]; then
    echo "Coefficient index must be one of 0,1,2,3; got: $i" >&2
    exit 2
  fi
  cc=${CC_VALUES[$i]}
  tag=${CC_TAGS[$i]}
  run="$RUN_ROOT/cc_$tag/$ENV_NAME/seed$SEED"
  mkdir -p "$(dirname "$run")"
  if ! mkdir "$run" 2>/dev/null; then
    echo "Skipping already claimed run: $run"
    continue
  fi

  {
    date --iso-8601=seconds
    hostname
    echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "METHOD=deterministic_joint_2B_fullcriticcurvature_rhscoef"
    echo "JOINT_SYSTEM_ROWS=1024"
    echo "CROSS_BLOCKS=retained"
    echo "LAMBDA_C=1.0"
    echo "C_C=$cc"
    echo "ENV_NAME=$ENV_NAME"
    echo "SEED=$SEED"
    echo "ROLLOUT_BATCH=4096"
    echo "MINIBATCH=512"
    echo "EPOCHS=4"
    echo "TRANSITIONS=6000000"
    echo "MOMENTUM=0"
    echo "KACZMARZ=false"
    echo "DAMPING=0.5"
    echo "TRAINER_SHA256=$TRAINER_SHA"
    echo "CONFIG_SHA256=$CONFIG_SHA"
  } > "$run/preflight.txt"

  printf '%q ' "$PY" -u "$TRAINER" --config "$CONFIG" \
    --joint_critic_curvature_coef 1.0 \
    --joint_critic_objective_coef "$cc" \
    --env_name "$ENV_NAME" --seed "$SEED" --device 0 > "$run/command.txt"
  printf '\n' >> "$run/command.txt"

  echo RUNNING > "$run/status"
  (
    set +e
    PYTHONHASHSEED=$SEED "$PY" -u "$TRAINER" \
      --config "$CONFIG" \
      --joint_critic_curvature_coef 1.0 \
      --joint_critic_objective_coef "$cc" \
      --env_name "$ENV_NAME" \
      --seed "$SEED" \
      --device 0 \
      > "$run/stdout" 2> "$run/stderr"
    rc=$?
    echo "$rc" > "$run/rc"
    if [[ "$rc" -eq 0 ]]; then echo PASS > "$run/status"; else echo FAIL > "$run/status"; fi
    exit "$rc"
  ) &
  pids+=("$!")
  tags+=("$tag")
done

overall=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "child ${tags[$i]} failed" >&2
    overall=1
  fi
done
exit "$overall"
