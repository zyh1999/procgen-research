#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --job-name=pg-nm2-closure
#SBATCH --output=/scratch/h99859yz/procgen_normmatch_v2_runtime_generated_closure_audit_6m_s0_20260825_22/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_normmatch_v2_runtime_generated_closure_audit_6m_s0_20260825_22/scheduler/%x_%j.err

set -euo pipefail
TASK15=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15
CAMPAIGN=/scratch/h99859yz/procgen_normmatch_v2_runtime_generated_closure_audit_6m_s0_20260825_22
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
VERIFY="$TASK15/tools/verify_hermetic_bundle.py"
PROBE="$CAMPAIGN/tools/runtime_generated_closure_probe.py"
ANALYZE="$CAMPAIGN/tools/analyze_closure_reproductions.py"
read -r BUNDLE_SHA BUNDLE_NAME < "$TASK15/bundle/BUNDLE_SHA256SUMS"
BUNDLE="$TASK15/bundle/$BUNDLE_NAME"
OUT="$CAMPAIGN/provenance/$SLURM_JOB_ID"
mkdir -p "$OUT"
finalize() {
  RC=$1
  date -Is > "$OUT/end_time.txt"
  printf '%s\n' "$RC" > "$OUT/rc"
  if [ "$RC" -eq 0 ]; then echo CLOSURE_PROVENANCE_COMPLETE > "$OUT/status"; else echo CLOSURE_PROVENANCE_FAILED > "$OUT/status"; fi
}
date -Is > "$OUT/start_time.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$OUT/job_id"
hostname -f > "$OUT/hostname"
nvidia-smi -L > "$OUT/gpu.txt" 2>&1
DEPLOY1=$(mktemp -d "${TMPDIR:-/tmp}/task22-deploy1-${SLURM_JOB_ID}.XXXXXX")
DEPLOY2=$(mktemp -d "${TMPDIR:-/tmp}/task22-deploy2-${SLURM_JOB_ID}.XXXXXX")
EMPTY1=$(mktemp -d "${TMPDIR:-/tmp}/task22-empty1-${SLURM_JOB_ID}.XXXXXX")
EMPTY2=$(mktemp -d "${TMPDIR:-/tmp}/task22-empty2-${SLURM_JOB_ID}.XXXXXX")
chmod 700 "$EMPTY1" "$EMPTY2"
cleanup() { rm -rf "$DEPLOY1" "$DEPLOY2" "$EMPTY1" "$EMPTY2"; }
on_exit() { RC=$?; cleanup; finalize "$RC"; }
trap on_exit EXIT
"$PY" "$VERIFY" "$BUNDLE" "$BUNDLE_SHA" "$DEPLOY1" > "$OUT/bundle_verify_1.out" 2> "$OUT/bundle_verify_1.err"
"$PY" "$VERIFY" "$BUNDLE" "$BUNDLE_SHA" "$DEPLOY2" > "$OUT/bundle_verify_2.out" 2> "$OUT/bundle_verify_2.err"
(cd "$EMPTY1"; env -u PYTHONPATH PYTHONNOUSERSITE=1 "$PY" "$PROBE" "$DEPLOY1" "$OUT/reproduction_1.json" "$EMPTY1") > "$OUT/reproduction_1.out" 2> "$OUT/reproduction_1.err"
(cd "$EMPTY2"; env -u PYTHONPATH PYTHONNOUSERSITE=1 "$PY" "$PROBE" "$DEPLOY2" "$OUT/reproduction_2.json" "$EMPTY2") > "$OUT/reproduction_2.out" 2> "$OUT/reproduction_2.err"
"$PY" "$ANALYZE" "$OUT/reproduction_1.json" "$OUT/reproduction_2.json" "$OUT/closure_decision.json" > "$OUT/analysis.out" 2> "$OUT/analysis.err"
grep -q TASK22_NORMALIZED_CLOSURE_EQUAL_PASS "$OUT/analysis.out"
grep -q TASK22_CLOSURE_PROVENANCE_BLOCKED_SYNTHETIC_ORIGIN_NO_ARTIFACT "$OUT/analysis.out"
sha256sum "$BUNDLE" "$PROBE" "$ANALYZE" > "$OUT/input.sha256"
echo TASK22_CLOSURE_PROVENANCE_COMPLETE
