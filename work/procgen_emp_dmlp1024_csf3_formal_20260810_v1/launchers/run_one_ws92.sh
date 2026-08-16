#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yihe/procgen_emp_dmlp1024_ws92_20260810_v1
PY=/home/yihe/.venv/bin/python
CONFIG=emp_a_dmlp1024_formal.yaml
ENV_NAME=${1:?environment is required}
SEED=${2:?seed is required}
PHYSICAL_GPU=${3:?physical GPU is required}
RUN="$ROOT/formal/$ENV_NAME/seed$SEED"

mkdir -p "$(dirname "$RUN")"
if ! mkdir "$RUN" 2>/dev/null; then
  echo "Refusing duplicate run: $RUN" >&2
  exit 6
fi

cd "$ROOT"
export PYTHONPATH="/home/yihe/procgen_ppo_overlay_20260720:/home/yihe/procgen_paper_deps_20260722:$ROOT"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$SEED"

echo RUNNING > "$RUN/status"
{
  date --iso-8601=seconds
  hostname
  echo "METHOD=nonppg_empirical_fisher_npg_shared_critic_adam"
  echo "ARCH=resnet_hidden256_decision_mlp_1024_256"
  echo "ROLLOUT_BATCH=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TOTAL_TRANSITIONS=6000000"
  echo "PHASIC=false"
  echo "OFFICIAL_PPG_AUXILIARY=false"
  echo "ENV_NAME=$ENV_NAME"
  echo "SEED=$SEED"
  echo "PHYSICAL_GPU=$PHYSICAL_GPU"
  sha256sum train_phasic_ef_ggn.py actor_ef_ablation.py phasic_ef_ggn.py utils/utils.py "configs/$CONFIG"
  "$PY" -V
  nvidia-smi
} > "$RUN/preflight.txt" 2>&1

set +e
"$PY" -u train_phasic_ef_ggn.py \
  --config "$CONFIG" --env_name "$ENV_NAME" --seed "$SEED" --device 0 \
  > "$RUN/stdout" 2> "$RUN/stderr"
RC=$?
set -e

echo "$RC" > "$RUN/rc"
if [[ "$RC" -eq 0 ]]; then
  echo PASS > "$RUN/status"
else
  echo FAIL > "$RUN/status"
fi
exit "$RC"
