#!/usr/bin/env bash
set -euo pipefail

CC=${1:?critic RHS coefficient is required}
TAG=${2:?coefficient tag is required}
GPU=${3:?physical GPU index is required}

ROOT=/home/yihe/procgen_det2b_rhscoef_4090_20260811_v1
CODE="$ROOT/code"
RUN="$ROOT/formal/cc_$TAG/bigfish-easy-0-10/seed0"
PY=/home/yihe/.RLvenv/bin/python
DEPS=/home/yihe/.procgen_overlay_20260805:/home/yihe/procgen_paper_deps_20260722
CODE_DEPS=/home/yihe/procgen_gaussian2b_dualtie16_cc0p1_4090_20260805/code
TRAINER=train_shared_rat_exact_deterministic_ggn_symfp64.py
CONFIG=adv_resnet_shared_det2b_rhsonly_symfp64.yaml
TRAINER_SHA=2b50f8cc26bb8c85f91e0b394acc7535f5564ac70036403597d3738d3dab9c1b
CONFIG_SHA=23a00a9307655475d76e6832cb7c8dd9a3f658924eb83f51bd07779ec3c5fa8d
TEST_SHA=8e3fbac2507d35456aae20ea2266a2da9e2487366f47c27ea55308a50296f221

case "$CC:$TAG" in
  0.1:0p1|0.3:0p3) ;;
  *) echo "Unsupported coefficient/tag pair: $CC $TAG" >&2; exit 2 ;;
esac
[[ "$GPU" =~ ^[01]$ ]]

mkdir -p "$(dirname "$RUN")"
if ! mkdir "$RUN" 2>/dev/null; then
  echo "Refusing to overwrite existing run: $RUN" >&2
  exit 6
fi

cd "$CODE"
export PYTHONPATH="$CODE:$CODE_DEPS:$DEPS${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0

[[ "$(sha256sum "$TRAINER" | awk '{print $1}')" == "$TRAINER_SHA" ]]
[[ "$(sha256sum "configs/$CONFIG" | awk '{print $1}')" == "$CONFIG_SHA" ]]
[[ "$(sha256sum test_symfp64_algebra.py | awk '{print $1}')" == "$TEST_SHA" ]]
grep -q 'joint_H = torch.cat(\[H_pi, critic_h_weight \* J_v\], dim=0)' "$TRAINER"
grep -q 'joint_system_rows=2 \* num_sa' "$TRAINER"
grep -q 'joint_critic_curvature_coef: 1.0' "configs/$CONFIG"

existing_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [[ ! "$existing_mib" =~ ^[0-9]+$ ]] || (( existing_mib > 6000 )); then
  echo "RESOURCE_BLOCKED existing_mib=$existing_mib" > "$RUN/status"
  exit 7
fi

echo RUNNING > "$RUN/status"
{
  date --iso-8601=seconds
  hostname
  echo "PHYSICAL_GPU=$GPU"
  echo "METHOD=deterministic_joint_2B_fullcriticcurvature_rhscoef"
  echo "JOINT_SYSTEM_ROWS=1024"
  echo "CROSS_BLOCKS=retained"
  echo "LAMBDA_C=1.0"
  echo "C_C=$CC"
  echo "ENV_NAME=bigfish-easy-0-10"
  echo "SEED=0"
  echo "ROLLOUT_BATCH=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=6000000"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "DAMPING=0.5"
  echo "EXISTING_GPU_MEMORY_MIB=$existing_mib"
  echo "TRAINER_SHA256=$TRAINER_SHA"
  echo "CONFIG_SHA256=$CONFIG_SHA"
  "$PY" -V
  nvidia-smi -i "$GPU"
} > "$RUN/preflight.txt" 2>&1

printf '%q ' "$PY" -u "$TRAINER" --config "$CONFIG" \
  --joint_critic_curvature_coef 1.0 \
  --joint_critic_objective_coef "$CC" \
  --env_name bigfish-easy-0-10 --seed 0 --device 0 > "$RUN/command.txt"
printf '\n' >> "$RUN/command.txt"

(
  while true; do
    date +%s
    nvidia-smi -i "$GPU" --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
    sleep 15
  done
) > "$RUN/gpu.csv" 2>&1 &
MON_PID=$!
cleanup_monitor() { kill "$MON_PID" 2>/dev/null || true; }
trap cleanup_monitor EXIT TERM INT

set +e
"$PY" -u "$TRAINER" \
  --config "$CONFIG" \
  --joint_critic_curvature_coef 1.0 \
  --joint_critic_objective_coef "$CC" \
  --env_name bigfish-easy-0-10 \
  --seed 0 \
  --device 0 \
  > "$RUN/stdout" 2> "$RUN/stderr"
rc=$?
set -e
echo "$rc" > "$RUN/rc"
if [[ "$rc" -eq 0 ]]; then echo PASS > "$RUN/status"; else echo FAIL > "$RUN/status"; fi
exit "$rc"
