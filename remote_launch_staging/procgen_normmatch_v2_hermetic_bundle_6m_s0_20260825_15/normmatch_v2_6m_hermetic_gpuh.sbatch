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
#SBATCH --output=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15/scheduler/%x_%j.err

set -uo pipefail
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
case "$ENV_NAME" in bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;; *) exit 64;; esac
CAMPAIGN=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
VERIFY="$CAMPAIGN/tools/verify_hermetic_bundle.py"
VARIANT="$CAMPAIGN/frozen/normmatch_v2_6m_hermetic_gpuh.sbatch"
read -r BUNDLE_SHA BUNDLE_NAME < "$CAMPAIGN/bundle/BUNDLE_SHA256SUMS"
BUNDLE="$CAMPAIGN/bundle/$BUNDLE_NAME"
DEPLOY=$(mktemp -d "${TMPDIR:-/tmp}/procgen-nm2-science-${SLURM_JOB_ID}.XXXXXX")
cleanup() { rm -rf "$DEPLOY"; }
trap cleanup EXIT
"$PY" "$VERIFY" "$BUNDLE" "$BUNDLE_SHA" "$DEPLOY" || exit 65
CODE="$DEPLOY/code"
FROZEN="$DEPLOY/frozen"
TRAINER="$CODE/train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
CONFIG="$CODE/configs/adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml"
LAUNCHER="$FROZEN/normmatch_v2_6m_gpuh.sbatch"
PREFLIGHT="$FROZEN/gpuh_preflight_normmatch_v2.py"
METHOD=PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2
ROOT="$CAMPAIGN/runs/$METHOD/$ENV_NAME/seed0/6m"
ACCEPTED="$CAMPAIGN/accepted_preflight/$ENV_NAME"
if [ -e "$ROOT" ]; then echo "collision: $ROOT" >&2; exit 90; fi
(cd "$DEPLOY" && sha256sum -c "$FROZEN/SCIENTIFIC_SHA256SUMS") || exit 66
test "$(cat "$ACCEPTED/status")" = PRECHECK_PASS || exit 68
test "$(cat "$ACCEPTED/rc")" = 0 || exit 69
grep -q GPUH_HYBRID_HEAD_COMPATIBILITY_PASS "$ACCEPTED/compatibility.out" || exit 70
grep -q "connectivity_environment=$ENV_NAME" "$ACCEPTED/compatibility.out" || exit 71
grep -q actual_network_head_proposal_normmatch=PASS "$ACCEPTED/compatibility.out" || exit 72
export PYTHONNOUSERSITE=1
export PYTHONPATH="$CODE"
mkdir -p "$ROOT" "$CODE/logs"
echo PREFLIGHT_ACCEPTED > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$BUNDLE" "$TRAINER" "$CONFIG" "$LAUNCHER" "$PREFLIGHT" "$VARIANT" > "$ROOT/frozen.sha256"
cp "$ACCEPTED/compatibility.out" "$ROOT/compatibility.out"
cp "$ACCEPTED/connectivity_probe.json" "$ROOT/connectivity_probe.json"
cp "$ACCEPTED/structural_manifest.json" "$ROOT/structural_manifest.json"
cp "$CAMPAIGN/bundle/BUNDLE_MANIFEST.json" "$ROOT/BUNDLE_MANIFEST.json"

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
