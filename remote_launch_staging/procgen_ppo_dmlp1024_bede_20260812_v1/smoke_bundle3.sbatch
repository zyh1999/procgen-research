#!/bin/bash
#SBATCH --job-name=pg-pdmlp-smoke
#SBATCH --account=bdman37
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1/smoke/slurm-%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1/smoke/slurm-%j.err

set -uo pipefail

ROOT=/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1
CODE="$ROOT/code"
PY=/nobackup/projects/bdman37/yihe/ppc64le/envs/rat/bin/python
CONFIG=ppo_resnet_shared_dmlp1024_smoke.yaml
RUN="$ROOT/smoke/run-$SLURM_JOB_ID"
mkdir "$RUN"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0
export PYTHONHASHSEED=0

echo RUNNING > "$RUN/status"
{
  date --iso-8601=seconds
  hostname
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "CONCURRENT_CHILDREN=3"
  "$PY" -V
  "$PY" -c 'import torch, procgen, gym3, yaml; print("torch", torch.__version__, "cuda", torch.version.cuda); print("procgen", procgen.__file__); print("gym3", gym3.__file__); print("yaml", yaml.__version__)'
  nvidia-smi
} > "$RUN/preflight.txt" 2>&1

(
  while true; do
    date +%s
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    sleep 5
  done
) > "$RUN/gpu.csv" 2>&1 &
MONITOR_PID=$!

PIDS=()
SEEDS=(0 1 2)
for SEED in "${SEEDS[@]}"; do
  CHILD="$RUN/seed$SEED"
  mkdir "$CHILD"
  cp -a "$CODE" "$CHILD/code"
  echo RUNNING > "$CHILD/status"
  (
    cd "$CHILD/code"
    exec "$PY" -u train_shared.py \
      --config "$CONFIG" \
      --env_name bigfish-easy-0-10 \
      --seed "$SEED" \
      --device 0
  ) > "$CHILD/stdout" 2> "$CHILD/stderr" &
  CHILD_PID=$!
  PIDS+=("$CHILD_PID")
done

terminate_children() {
  echo TERMINATING > "$RUN/status"
  for PID in "${PIDS[@]}"; do
    kill "$PID" 2>/dev/null || true
  done
  kill "$MONITOR_PID" 2>/dev/null || true
}
trap terminate_children TERM INT

OVERALL=0
printf 'seed\tpid\trc\tstatus\n' > "$RUN/seed_status.tsv"
for INDEX in 0 1 2; do
  SEED=${SEEDS[$INDEX]}
  PID=${PIDS[$INDEX]}
  set +e
  wait "$PID"
  RC=$?
  set -e
  echo "$RC" > "$RUN/seed$SEED/rc"
  if [[ "$RC" -eq 0 ]]; then CHILD_STATUS=PASS; else CHILD_STATUS=FAIL; OVERALL=1; fi
  echo "$CHILD_STATUS" > "$RUN/seed$SEED/status"
  printf '%s\t%s\t%s\t%s\n' "$SEED" "$PID" "$RC" "$CHILD_STATUS" >> "$RUN/seed_status.tsv"
done

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
echo "$OVERALL" > "$RUN/returncode"
if [[ "$OVERALL" -eq 0 ]]; then echo PASS > "$RUN/status"; else echo FAIL > "$RUN/status"; fi
exit "$OVERALL"
