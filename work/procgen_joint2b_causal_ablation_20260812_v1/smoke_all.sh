#!/usr/bin/env bash
set -euo pipefail

ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
CODE="$ROOT/code"
OUT="$ROOT/smoke_fast_v2"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER=train_shared_joint2b_causal_ablation.py
CONFIG=adv_resnet_shared_joint2b_causal_smoke.yaml
mkdir -p "$OUT"
cd "$CODE"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0

run_one() {
  local tag=$1 mode=$2 score=$3 scope=$4 lambda_c=$5 c_c=$6 lr_max=$7 kl_upper=$8
  local run="$OUT/$tag"
  mkdir -p "$run"
  export PROCGEN_METRIC_TRACE_PATH="$run/metric_trace.jsonl"
  echo RUNNING > "$run/status"
  set +e
  "$PY" -u "$TRAINER" --config "$CONFIG" \
    --env_name bigfish-easy-0-10 --seed 4242 --device 0 \
    --total_timesteps 512 \
    --joint_ablation_mode "$mode" \
    --joint_critic_score_mode "$score" \
    --joint_critic_param_scope "$scope" \
    --joint_critic_curvature_coef "$lambda_c" \
    --joint_critic_objective_coef "$c_c" \
    --adaptive_lr_max "$lr_max" \
    --adaptive_kl_upper "$kl_upper" \
    > "$run/stdout" 2> "$run/stderr"
  local rc=$?
  set -e
  echo "$rc" > "$run/rc"
  if (( rc == 0 )); then echo PASS > "$run/status"; else echo FAIL > "$run/status"; fi
  return "$rc"
}

run_one actor_only actor_only clean all 0.0 0.0 0.5 0.04
run_one curvature_only curvature_only clean all 1.0 0.0 0.5 0.04
run_one full_clean full_joint clean all 1.0 1.0 0.5 0.04
run_one full_head full_joint clean head_only 1.0 1.0 0.5 0.04
run_one full_rademacher full_joint rademacher all 1.0 1.0 0.5 0.04
run_one full_gaussian full_joint gaussian_unit all 1.0 1.0 0.5 0.04
run_one full_clean_conservative full_joint clean all 1.0 1.0 0.05 0.02

echo PASS > "$OUT/status"
