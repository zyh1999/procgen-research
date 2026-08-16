#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_ID ENV_NAME" >&2
  exit 2
fi

GPU_ID=$1
ENV_NAME=$2
case "$GPU_ID" in
  0|1) ;;
  *) echo "GPU_ID must be 0 or 1" >&2; exit 2 ;;
esac
case "$ENV_NAME" in
  bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;;
  *) echo "unsupported gate environment: $ENV_NAME" >&2; exit 2 ;;
esac

ROOT=/home/yihe/procgen_joint2b_mujocoported_blocknorm_ws92_20260815_v1
CODE="$ROOT/code"
PY=/home/yihe/.venv/bin/python
TRAINER=train_shared_joint2b_mujocoported_blocknorm.py
CONFIG=adv_resnet_shared_joint2b_mujocoported_blocknorm_d003_lr05.yaml
EXPECTED_TRAINER_SHA=a6a2ff48d3a86892e26dae66ae1039b69e6f3082c9bd24e73751f47a84887a9a
EXPECTED_CONFIG_SHA=75cd1af5c8651c4871ef842140d992d9e4a49827a8947554d8d9121e1d740ffd
RUN="$ROOT/gate_1m_seed0_v1/blocknorm_d003_lr05/$ENV_NAME/seed0"

trainer_sha=$(sha256sum "$CODE/$TRAINER" | awk '{print $1}')
config_sha=$(sha256sum "$CODE/configs/$CONFIG" | awk '{print $1}')
[[ "$trainer_sha" == "$EXPECTED_TRAINER_SHA" ]] || {
  echo "trainer SHA mismatch: $trainer_sha" >&2
  exit 4
}
[[ "$config_sha" == "$EXPECTED_CONFIG_SHA" ]] || {
  echo "config SHA mismatch: $config_sha" >&2
  exit 5
}

mkdir -p "$(dirname "$RUN")"
if ! mkdir "$RUN"; then
  echo "run already claimed: $RUN" >&2
  exit 3
fi

{
  echo "HOST=ws4090-92"
  echo "GPU_ID=$GPU_ID"
  echo "ENV_NAME=$ENV_NAME"
  echo "SEED=0"
  echo "TRAINER_SHA256=$trainer_sha"
  echo "CONFIG_SHA256=$config_sha"
  echo "ROLLOUT=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=1000000"
  echo "JOINT_SYSTEM_ROWS=1024"
  echo "JOINT_MODE=full_joint_clean_all"
  echo "JOINT_RHS_MODE=paired_score_residual"
  echo "JOINT_RECONSTRUCTION_MODE=full_joint"
  echo "JOINT_BLOCK_NORMALIZATION=median_gradient_preserving"
  echo "CRITIC_OBJECTIVE_COEF=1.0"
  echo "CRITIC_CURVATURE_COEF=1.0"
  echo "DAMPING=0.03"
  echo "LR_INITIAL_MAX=0.05"
  echo "KL_RANGE=0.005,0.02"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "LINEAR_SOLVE_DTYPE=float64"
} > "$RUN/preflight"

cmd=("$PY" -u "$TRAINER" --config "$CONFIG"
  --env_name "$ENV_NAME" --seed 0 --device 0
  --total_timesteps 1000000 --joint_ablation_mode full_joint
  --joint_critic_score_mode clean --joint_critic_param_scope all
  --joint_critic_reconstruction_scope all
  --joint_critic_curvature_coef 1.0 --joint_critic_objective_coef 1.0)
printf '%q ' "${cmd[@]}" > "$RUN/command.txt"
printf '\n' >> "$RUN/command.txt"

echo RUNNING > "$RUN/status"
cd "$CODE"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PROCGEN_METRIC_TRACE_PATH="$RUN/metric_trace.jsonl"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
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
