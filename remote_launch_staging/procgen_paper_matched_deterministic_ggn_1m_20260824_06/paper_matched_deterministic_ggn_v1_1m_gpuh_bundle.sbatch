#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_paper_matched_deterministic_ggn_1m_20260824_06/scheduler_gpuh/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_paper_matched_deterministic_ggn_1m_20260824_06/scheduler_gpuh/%x_%j.err

set -uo pipefail
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV is required}
case "$ENV_NAME" in
  bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;;
  *) echo "invalid environment: $ENV_NAME" >&2; exit 64 ;;
esac

CAMPAIGN=/scratch/h99859yz/procgen_paper_matched_deterministic_ggn_1m_20260824_06
CODE="$CAMPAIGN/code"
PYTHON=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_paper_matched_deterministic_ggn_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_paper_matched_deterministic_ggn_v1_1m.yaml"
LAUNCHER="$CAMPAIGN/frozen/paper_matched_deterministic_ggn_v1_1m_gpuh_bundle.sbatch"
PREFLIGHT="$CAMPAIGN/frozen/gpuh_bundle_compatibility_preflight.py"
METHOD=PAPER_MATCHED_DETERMINISTIC_GGN_V1
BUNDLE_ROOT="$CAMPAIGN/runs_gpuh/$METHOD/$ENV_NAME/bundle_$SLURM_JOB_ID"

if [ -e "$BUNDLE_ROOT" ]; then
  echo "bundle collision: $BUNDLE_ROOT" >&2
  exit 90
fi
mkdir -p "$BUNDLE_ROOT" "$CODE/logs"
sha256sum "$TRAINER" "$CONFIG" "$LAUNCHER" "$PREFLIGHT" > "$BUNDLE_ROOT/preflight.sha256"
hostname -f > "$BUNDLE_ROOT/hostname"
nvidia-smi -L > "$BUNDLE_ROOT/gpu.txt" 2>&1

if ! "$PYTHON" "$PREFLIGHT" > "$BUNDLE_ROOT/compatibility.out" 2> "$BUNDLE_ROOT/compatibility.err"; then
  echo PREFLIGHT_FAIL > "$BUNDLE_ROOT/status"
  exit 70
fi
grep -qx GPUH_BUNDLE_COMPATIBILITY_PASS <(grep GPUH_BUNDLE_COMPATIBILITY_PASS "$BUNDLE_ROOT/compatibility.out") || {
  echo PREFLIGHT_FAIL > "$BUNDLE_ROOT/status"
  exit 71
}
echo PREFLIGHT_PASS > "$BUNDLE_ROOT/status"

run_child() {
  local seed=$1
  local root="$CAMPAIGN/runs_gpuh/$METHOD/$ENV_NAME/seed${seed}/1m"
  if [ -e "$root" ]; then
    echo "child collision: $root" >&2
    return 90
  fi
  mkdir -p "$root"
  echo RUNNING > "$root/status"
  printf '%s\n' "$SLURM_JOB_ID" > "$root/job_id"
  printf '%s\n' "$seed" > "$root/seed"
  hostname -f > "$root/hostname"
  nvidia-smi -L > "$root/gpu.txt" 2>&1
  sha256sum "$TRAINER" "$CONFIG" "$LAUNCHER" "$PREFLIGHT" > "$root/preflight.sha256"
  local cmd=("$PYTHON" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name "$ENV_NAME" --seed "$seed" --device 0)
  printf '%q ' "${cmd[@]}" > "$root/command.txt"
  printf '\n' >> "$root/command.txt"
  touch "$root/start.marker" "$root/scientific_started.marker"
  (
    cd "$CODE"
    OMP_NUM_THREADS=1 "${cmd[@]}" > "$root/stdout.log" 2> "$root/stderr.log"
  )
  local rc=$?
  printf '%s\n' "$rc" > "$root/rc"
  local logdir=""
  if [ -d "$CODE/logs" ]; then
    logdir=$(find "$CODE/logs" -mindepth 2 -maxdepth 2 -type d -name "${ENV_NAME}.*_${seed}" -newer "$root/start.marker" -print | sort | tail -1)
  fi
  if [ -n "$logdir" ] && [ -d "$logdir" ]; then
    printf '%s\n' "$logdir" > "$root/source_log_dir"
    cp "$logdir/progress.csv" "$root/progress.csv" 2>/dev/null || true
    cp "$logdir/metric_trace.jsonl" "$root/metric_trace.jsonl" 2>/dev/null || true
    cp "$logdir/model.ckpt" "$root/model.ckpt" 2>/dev/null || true
  fi
  if [ "$rc" -eq 0 ] && [ -s "$root/progress.csv" ] && [ -s "$root/metric_trace.jsonl" ] && [ -s "$root/model.ckpt" ]; then
    echo PASS > "$root/status"
    return 0
  fi
  echo FAIL > "$root/status"
  return 1
}

pids=()
for seed in 0 1 2 3 4 5 6 7; do
  run_child "$seed" &
  pids+=("$!")
done
echo SCIENTIFIC_CHILDREN_STARTED > "$BUNDLE_ROOT/status"

bundle_rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || bundle_rc=1
done
printf '%s\n' "$bundle_rc" > "$BUNDLE_ROOT/rc"
if [ "$bundle_rc" -eq 0 ]; then
  echo PASS > "$BUNDLE_ROOT/status"
else
  echo FAIL > "$BUNDLE_ROOT/status"
fi
exit "$bundle_rc"
