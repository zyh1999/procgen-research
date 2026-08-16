#!/bin/bash
#SBATCH --account=bdman37
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1/formal_4env_x3seed_6m_20260812_v1/slurm/%x-%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1/formal_4env_x3seed_6m_20260812_v1/slurm/%x-%j.err

set -uo pipefail

ROOT=/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1
CODE="$ROOT/code"
PY=/nobackup/projects/bdman37/yihe/ppc64le/envs/rat/bin/python
CAMPAIGN="$ROOT/formal_4env_x3seed_6m_20260812_v1"
CONFIG=ppo_resnet_shared_dmlp1024.yaml
ENV_NAME=${ENV_NAME:?ENV_NAME is required}
ENV_TAG=${ENV_TAG:?ENV_TAG is required}

case "$ENV_TAG:$ENV_NAME" in
  bigfish:bigfish-easy-0-10|bossfight:bossfight-easy-0-10|caveflyer:caveflyer-easy-0-10|coinrun:coinrun-easy-0-10) ;;
  *) echo "refused unknown environment mapping $ENV_TAG:$ENV_NAME" >&2; exit 2 ;;
esac

BUNDLE_ROOT="$CAMPAIGN/$ENV_TAG"
if ! mkdir "$BUNDLE_ROOT" 2>/dev/null; then
  echo "refusing to overwrite existing bundle root $BUNDLE_ROOT" >&2
  exit 6
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0

echo RUNNING > "$BUNDLE_ROOT/status"
{
  date --iso-8601=seconds
  hostname
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "ENV_NAME=$ENV_NAME"
  echo "ENV_TAG=$ENV_TAG"
  echo "CONCURRENT_CHILDREN=3"
  echo "SEEDS=0,1,2"
  echo "ALGORITHM=pure_ppo"
  echo "DECISION_MLP=256-1024-256"
  echo "ACTIVE_AUXILIARY_HEAD=0"
  echo "EXPECTED_PARAMETERS=1464547"
  echo "BUDGET_PER_CHILD=6000000"
  "$PY" -V
  "$PY" -c 'import torch, procgen, gym3, yaml; print("torch", torch.__version__, "cuda", torch.version.cuda); print("procgen", procgen.__file__); print("gym3", gym3.__file__); print("yaml", yaml.__version__)'
  nvidia-smi
} > "$BUNDLE_ROOT/preflight.txt" 2>&1

(
  while true; do
    date +%s
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    sleep 15
  done
) > "$BUNDLE_ROOT/gpu.csv" 2>&1 &
MONITOR_PID=$!

PIDS=()
SEEDS=(0 1 2)
for SEED in "${SEEDS[@]}"; do
  CHILD="$BUNDLE_ROOT/seed$SEED"
  mkdir "$CHILD"
  cp -a "$CODE" "$CHILD/code"
  echo RUNNING > "$CHILD/status"
  {
    printf '%q ' "$PY" -u train_shared.py --config "$CONFIG" --env_name "$ENV_NAME" --seed "$SEED" --device 0
    printf '\n'
  } > "$CHILD/command.txt"
  (
    cd "$CHILD/code"
    exec "$PY" -u train_shared.py \
      --config "$CONFIG" \
      --env_name "$ENV_NAME" \
      --seed "$SEED" \
      --device 0
  ) > "$CHILD/stdout" 2> "$CHILD/stderr" &
  CHILD_PID=$!
  PIDS+=("$CHILD_PID")
  printf 'RUNNING\tseed=%s\tpid=%s\tutc=%s\n' "$SEED" "$CHILD_PID" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$CHILD/launch.status"
done

terminate_children() {
  echo TERMINATING > "$BUNDLE_ROOT/status"
  for PID in "${PIDS[@]}"; do
    kill "$PID" 2>/dev/null || true
  done
  kill "$MONITOR_PID" 2>/dev/null || true
}
trap terminate_children TERM INT

OVERALL=0
printf 'seed\tpid\trc\tstatus\n' > "$BUNDLE_ROOT/seed_status.tsv"
for INDEX in 0 1 2; do
  SEED=${SEEDS[$INDEX]}
  PID=${PIDS[$INDEX]}
  set +e
  wait "$PID"
  RC=$?
  set -e
  echo "$RC" > "$BUNDLE_ROOT/seed$SEED/rc"
  if [[ "$RC" -eq 0 ]]; then
    CHILD_STATUS=PASS
  else
    CHILD_STATUS=FAIL
    OVERALL=1
  fi
  echo "$CHILD_STATUS" > "$BUNDLE_ROOT/seed$SEED/status"
  printf '%s\t%s\t%s\t%s\n' "$SEED" "$PID" "$RC" "$CHILD_STATUS" >> "$BUNDLE_ROOT/seed_status.tsv"
done

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
echo "$OVERALL" > "$BUNDLE_ROOT/returncode"
if [[ "$OVERALL" -eq 0 ]]; then
  echo PASS > "$BUNDLE_ROOT/status"
else
  echo FAIL > "$BUNDLE_ROOT/status"
fi
exit "$OVERALL"
