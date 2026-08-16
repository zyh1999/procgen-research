#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_ID ENV_NAME" >&2
  exit 2
fi
GPU_ID=$1
ENV_NAME=$2
case "$GPU_ID" in 0|1) ;; *) echo "GPU_ID must be 0 or 1" >&2; exit 2;; esac
case "$ENV_NAME" in
  bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;;
  *) echo "unsupported environment: $ENV_NAME" >&2; exit 2;;
esac

ROOT=/home/yihe/procgen_joint2b_mujocoported_blocknorm_ws92_20260815_v1
CODE="$ROOT/code"
PY=/home/yihe/.venv/bin/python
TRAINER=train_shared_joint2b_mujocoported_rownorm_schurguard.py
CONFIG=adv_resnet_shared_joint2b_mujocoported_rownorm_schurguard_c1_d003_lr05_1m.yaml
EXPECTED_TRAINER_SHA=26d943237efdc600a8de3219d575cd4a71c85069c964a4843e092c3ca0b3034d
EXPECTED_CONFIG_SHA=b646553a2e0081952dd0afff9eb3d700bbee7b2b8344eb4ed2ba92c38c4d5d8e
RUN="$ROOT/smoke_50k_seed0_rownorm_schurguard_c1_v1/rownorm_schur_c1_d003_lr05/$ENV_NAME/seed0"

trainer_sha=$(sha256sum "$CODE/$TRAINER" | awk '{print $1}')
config_sha=$(sha256sum "$CODE/configs/$CONFIG" | awk '{print $1}')
[[ "$trainer_sha" == "$EXPECTED_TRAINER_SHA" ]] || exit 4
[[ "$config_sha" == "$EXPECTED_CONFIG_SHA" ]] || exit 5
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN" || { echo "run already claimed: $RUN" >&2; exit 3; }

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
  echo "TRANSITIONS=50000"
  echo "JOINT_SYSTEM_ROWS=1024"
  echo "JOINT_MODE=full_joint_clean_all"
  echo "JOINT_RHS_MODE=paired_score_residual"
  echo "JOINT_RECONSTRUCTION_MODE=full_joint"
  echo "JOINT_BLOCK_NORMALIZATION=row_gradient_preserving"
  echo "CRITIC_OBJECTIVE_COEF=1.0"
  echo "CRITIC_CURVATURE_COEF=1.0"
  echo "DAMPING_ACTOR=0.03"
  echo "CRITIC_DAMPING_TO_MEDIAN_FLOOR=1.0"
  echo "SCHUR_GUARD=true"
  echo "LR_INITIAL_MAX=0.05"
  echo "KL_RANGE=0.005,0.02"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "LINEAR_SOLVE_DTYPE=float64"
  echo "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  echo "SMOKE_ONLY=true"
  echo "CAUSAL_REASON=VERIFY_CRITIC_DUAL_BLOCK_DAMPING_WIRING"
} > "$RUN/preflight"

cmd=("$PY" -u "$TRAINER" --config "$CONFIG"
  --env_name "$ENV_NAME" --seed 0 --device 0
  --total_timesteps 50000 --joint_ablation_mode full_joint
  --joint_critic_score_mode clean --joint_critic_param_scope all
  --joint_critic_reconstruction_scope all
  --joint_critic_curvature_coef 1.0 --joint_critic_objective_coef 1.0)
printf '%q ' "${cmd[@]}" > "$RUN/command.txt"; printf '\n' >> "$RUN/command.txt"

echo RUNNING > "$RUN/status"
cd "$CODE"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PROCGEN_METRIC_TRACE_PATH="$RUN/metric_trace.jsonl"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
set +e
"${cmd[@]}" > "$RUN/stdout" 2> "$RUN/stderr"
rc=$?
set -e
echo "$rc" > "$RUN/rc"
if [[ $rc -eq 0 ]]; then echo PASS > "$RUN/status"; else echo FAILED > "$RUN/status"; fi
exit "$rc"
