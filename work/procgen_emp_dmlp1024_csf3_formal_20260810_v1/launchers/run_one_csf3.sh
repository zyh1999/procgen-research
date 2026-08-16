#!/usr/bin/env bash
set -euo pipefail

ROOT=/scratch/h99859yz/procgen_emp_dmlp1024_csf3_20260810_v1
CODE="$ROOT"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
CONFIG=emp_a_dmlp1024_formal.yaml

ENV_NAME=${1:?environment is required}
SEED=${2:?seed is required}
RUN_SUFFIX=${3:-}
RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
RUN_LABEL="seed$SEED"
if [[ -n "$RUN_SUFFIX" ]]; then
  RUN_LABEL="${RUN_LABEL}_${RUN_SUFFIX}"
elif [[ "$RESTART_COUNT" -gt 0 ]]; then
  RUN_LABEL="${RUN_LABEL}_requeue${RESTART_COUNT}"
fi
RUN="$ROOT/formal/$ENV_NAME/$RUN_LABEL"

mkdir -p "$(dirname "$RUN")"
if ! mkdir "$RUN" 2>/dev/null; then
  echo "Refusing duplicate run: $RUN" >&2
  exit 6
fi

cd "$CODE"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED="$SEED"

echo RUNNING > "$RUN/status"
{
  date --iso-8601=seconds
  hostname
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
  echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
  echo "SLURM_RESTART_COUNT=$RESTART_COUNT"
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
