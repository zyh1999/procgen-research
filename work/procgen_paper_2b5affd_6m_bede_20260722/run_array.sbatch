#!/bin/bash
#SBATCH --account=bdman37
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --array=0-7%8
#SBATCH --signal=B:TERM@60

set -u

: "${BUNDLE_ROOT:?BUNDLE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${METHOD:?METHOD must be rat or ppo}"

CODE="$BUNDLE_ROOT/source"
PY=/nobackup/projects/bdman37/yihe/ppc64le/envs/procgen_author/bin/python

case "$METHOD" in
  rat)
    CONFIG=adv_resnet_shared.yaml
    ;;
  ppo)
    CONFIG=ppo_resnet_shared.yaml
    ;;
  *)
    echo "Unknown METHOD=$METHOD" >&2
    exit 2
    ;;
esac

ENVS=(
  bigfish-easy-0-10
  bossfight-easy-0-10
  caveflyer-easy-0-10
  coinrun-easy-0-10
  jumper-easy-0-10
  maze-easy-0-10
  miner-easy-0-10
  starpilot-easy-0-10
)

ENV_NAME="${ENVS[$SLURM_ARRAY_TASK_ID]}"
RUN="$RUN_ROOT/$METHOD/$ENV_NAME"
mkdir -p "$RUN"
cd "$CODE"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo RUNNING > "$RUN/status"
{
  date --iso-8601=seconds
  hostname
  echo "SOURCE_COMMIT=2b5affd64cbb3c624b4bc1f4767f449df231ffb2"
  echo "CONTROLLED_OVERRIDE=timesteps_per_proc_easy:6000000"
  echo "METHOD=$METHOD"
  echo "CONFIG=$CONFIG"
  echo "ENV_NAME=$ENV_NAME"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  sha256sum train_shared.py "configs/$CONFIG"
  "$PY" -V
  "$PY" -c 'import importlib.metadata as m, torch, procgen, gym3; print("torch", torch.__version__, "cuda", torch.version.cuda); print("procgen", m.version("procgen"), procgen.__file__); print("gym3", m.version("gym3"), gym3.__file__)'
  nvidia-smi
} > "$RUN/preflight.txt" 2>&1

(
  while true; do
    date +%s
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    sleep 10
  done
) > "$RUN/gpu.csv" 2>&1 &
MON_PID=$!

PIDS=()
for SEED in 0 1 2 3 4; do
  {
    echo "$PY -u train_shared.py --config $CONFIG --env_name $ENV_NAME --seed $SEED --device 0"
  } > "$RUN/seed${SEED}.command.txt"
  "$PY" -u train_shared.py \
    --config "$CONFIG" \
    --env_name "$ENV_NAME" \
    --seed "$SEED" \
    --device 0 \
    > "$RUN/seed${SEED}.stdout" 2> "$RUN/seed${SEED}.stderr" &
  PIDS+=("$!")
done

terminate_children() {
  echo TERMINATING > "$RUN/status"
  for PID in "${PIDS[@]}"; do
    kill "$PID" 2>/dev/null || true
  done
  kill "$MON_PID" 2>/dev/null || true
}
trap terminate_children TERM INT

OVERALL=0
for IDX in 0 1 2 3 4; do
  set +e
  wait "${PIDS[$IDX]}"
  RC=$?
  set -e
  echo "$RC" > "$RUN/seed${IDX}.rc"
  if test "$RC" -ne 0; then
    OVERALL=1
  fi
done

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true

echo "$OVERALL" > "$RUN/returncode"
if test "$OVERALL" -eq 0; then
  echo PASS > "$RUN/status"
else
  echo FAIL > "$RUN/status"
fi
exit "$OVERALL"
