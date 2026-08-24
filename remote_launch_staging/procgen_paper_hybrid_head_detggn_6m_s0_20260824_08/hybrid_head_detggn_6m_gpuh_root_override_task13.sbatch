#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/scheduler/%x_%j.err

set -uo pipefail
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
case "$ENV_NAME" in bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;; *) exit 64;; esac
# ROOT_OVERRIDE_BEGIN
BASE_LAUNCHER_SHA=ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e
TASK_ID=PROCGEN-HYBRID-HEAD-ROOT-OVERRIDE-6M-MISSING3-20260824-13
TASK11_CAMPAIGN=/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08
CAMPAIGN=${PROCGEN_CAMPAIGN_ROOT:?PROCGEN_CAMPAIGN_ROOT required}
case "$CAMPAIGN" in /*) ;; *) echo "campaign must be absolute: $CAMPAIGN" >&2; exit 65;; esac
CAMPAIGN=$(readlink -m -- "$CAMPAIGN")
if [ "$CAMPAIGN" = "$TASK11_CAMPAIGN" ] || [[ "$CAMPAIGN" = "$TASK11_CAMPAIGN"/* ]]; then
  echo "campaign resolves into immutable Task 11 campaign: $CAMPAIGN" >&2
  exit 66
fi
# ROOT_OVERRIDE_END
CODE="$CAMPAIGN/code"
FROZEN="$CAMPAIGN/frozen"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_paper_hybrid_head_detggn_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_paper_hybrid_head_detggn_v1_6m.yaml"
LAUNCHER="$FROZEN/hybrid_head_detggn_6m_gpuh.sbatch"
PREFLIGHT="$FROZEN/gpuh_preflight.py"
METHOD=PAPER_HYBRID_SHARED_PAPER_HEAD_DETGGN_V1
ROOT="$CAMPAIGN/runs/$METHOD/$ENV_NAME/seed0/6m"
TRAINER_SHA=7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54
CONFIG_SHA=9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda

if [ -e "$ROOT" ]; then echo "collision: $ROOT" >&2; exit 90; fi
mkdir -p "$ROOT" "$CODE/logs"
# ROOT_PROVENANCE_BEGIN
printf '%s\n' "$BASE_LAUNCHER_SHA" > "$ROOT/base_launcher.sha256"
printf '%s\n' "$TASK_ID" > "$ROOT/task_id.txt"
printf '%s\n' "$CAMPAIGN" > "$ROOT/campaign_root.txt"
# ROOT_PROVENANCE_END
echo PREFLIGHT_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$LAUNCHER" "$PREFLIGHT" > "$ROOT/preflight.sha256"
if ! "$PY" "$PREFLIGHT" "$TRAINER" "$CONFIG" "$ROOT/parameter_partition.json" "$TRAINER_SHA" "$CONFIG_SHA" > "$ROOT/compatibility.out" 2> "$ROOT/compatibility.err"; then
  echo INFRASTRUCTURE_PREFLIGHT_FAIL > "$ROOT/status"; exit 70
fi
grep -q GPUH_HYBRID_HEAD_COMPATIBILITY_PASS "$ROOT/compatibility.out" || { echo INFRASTRUCTURE_PREFLIGHT_FAIL > "$ROOT/status"; exit 71; }

CMD=("$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name "$ENV_NAME" --seed 0 --device 0)
printf '%q ' "${CMD[@]}" > "$ROOT/command.txt"; printf '\n' >> "$ROOT/command.txt"
touch "$ROOT/start.marker" "$ROOT/scientific_started.marker"
echo RUNNING > "$ROOT/status"

sync_artifacts() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    LOGDIR=$(find "$CODE/logs" -mindepth 2 -maxdepth 2 -type d -name "${ENV_NAME}.*_0" -newer "$ROOT/start.marker" -print 2>/dev/null | sort | tail -1)
    if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
      printf '%s\n' "$LOGDIR" > "$ROOT/source_log_dir"
      cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
      cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
      cp "$LOGDIR/model.ckpt" "$ROOT/model.ckpt" 2>/dev/null || true
      cp "$LOGDIR/parameter_partition.json" "$ROOT/training_parameter_partition.json" 2>/dev/null || true
    fi
    sleep 60
  done
}
(cd "$CODE"; OMP_NUM_THREADS=8 "${CMD[@]}" > "$ROOT/stdout.log" 2> "$ROOT/stderr.log") &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$ROOT/trainer_pid"
sync_artifacts & SYNC_PID=$!
wait "$TRAIN_PID"; RC=$?
wait "$SYNC_PID" 2>/dev/null || true
LOGDIR=$(cat "$ROOT/source_log_dir" 2>/dev/null || true)
if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
  cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
  cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
  cp "$LOGDIR/model.ckpt" "$ROOT/model.ckpt" 2>/dev/null || true
  cp "$LOGDIR/parameter_partition.json" "$ROOT/training_parameter_partition.json" 2>/dev/null || true
fi
printf '%s\n' "$RC" > "$ROOT/rc"
if [ "$RC" -eq 0 ] && [ -s "$ROOT/progress.csv" ] && [ -s "$ROOT/metric_trace.jsonl" ] && [ -s "$ROOT/model.ckpt" ]; then echo PASS > "$ROOT/status"; else echo FAIL > "$ROOT/status"; fi
exit "$RC"
