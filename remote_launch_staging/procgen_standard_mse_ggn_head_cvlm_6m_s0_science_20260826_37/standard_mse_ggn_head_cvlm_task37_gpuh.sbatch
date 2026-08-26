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
#SBATCH --output=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_science_20260826_37/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_science_20260826_37/scheduler/%x_%j.err

set -uo pipefail
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
case "$ENV_NAME" in bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;; *) exit 64;; esac
TASK_ID=PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-6M-S0-SCIENCE-20260826-37
CAMPAIGN=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_6m_s0_science_20260826_37
BUNDLE_ROOT="$CAMPAIGN/bundle"
CODE="$BUNDLE_ROOT/code"
FROZEN="$BUNDLE_ROOT/frozen"
TASK36=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_audit_path_recovery_20260826_36/preflight/$ENV_NAME
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_det_standard_mse_ggn_head_cvlm_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml"
ORIGINAL_LAUNCHER="$FROZEN/standard_mse_ggn_head_cvlm_6m_gpuh.sbatch"
PREFLIGHT="$FROZEN/gpuh_preflight.py"
AUDIT="$FROZEN/audit_task34r.py"
ADAPTER=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_audit_path_recovery_20260826_36/frozen/audit_path_adapter_task36.py
METHOD=DET_STANDARD_MSE_GGN_HEAD_CVLM_V1
ROOT="$CAMPAIGN/runs/$METHOD/$ENV_NAME/seed0/6m"

if [ -e "$ROOT" ]; then echo "collision: $ROOT" >&2; exit 90; fi
test "$(sha256sum "$BUNDLE_ROOT/BUNDLE_MANIFEST.json" | awk '{print $1}')" = 287a744078b10054d107974125bac6b5fac43fd944b6200ec54720cd2695c9af || exit 61
test "$(sha256sum "$TRAINER" | awk '{print $1}')" = ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc || exit 62
test "$(sha256sum "$CONFIG" | awk '{print $1}')" = 52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132 || exit 63
test "$(sha256sum "$PREFLIGHT" | awk '{print $1}')" = 2baac759c26c4fb663ac24ff68a6de8b640bb9bd0c443ec8f65106de3d36759a || exit 65
test "$(sha256sum "$AUDIT" | awk '{print $1}')" = 9d4929685bd8368f861770f155c5fcc0b7f86b91e9cfc92481987b0dc8ec2723 || exit 66
test "$(sha256sum "$ORIGINAL_LAUNCHER" | awk '{print $1}')" = 6dffce265e55f87fa5be848ec5bd9940fcecb6203f621a1454fb4f6a9c74caca || exit 67
test "$(sha256sum "$ADAPTER" | awk '{print $1}')" = 7b8cd684f448b730720e4acd1a9c6762faac95778339471770bd40b11f889dd4 || exit 68
test -f "$TASK36/precheck_pass.marker" || exit 69
test "$(cat "$TASK36/status")" = PRECHECK_PASS || exit 70
test "$(cat "$TASK36/rc")" = 0 || exit 71
grep -q GPUH_STANDARD_MSE_GGN_HEAD_CVLM_COMPATIBILITY_PASS "$TASK36/preflight.out" || exit 72

mkdir -p "$ROOT" "$CODE/logs"
printf '%s\n' "$TASK_ID" > "$ROOT/task_id.txt"
printf '%s\n' "$CAMPAIGN" > "$ROOT/campaign_root.txt"
printf '%s\n' "$TASK36" > "$ROOT/reused_task36_precheck.txt"
echo IDENTITY_VERIFIED > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$BUNDLE_ROOT/BUNDLE_MANIFEST.json" "$TRAINER" "$CONFIG" "$ORIGINAL_LAUNCHER" "$PREFLIGHT" "$AUDIT" "$ADAPTER" > "$ROOT/frozen_identity.sha256"

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
