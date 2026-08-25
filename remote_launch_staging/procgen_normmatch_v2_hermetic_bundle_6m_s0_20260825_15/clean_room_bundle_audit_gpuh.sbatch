#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --job-name=pg-nm2-bundle-audit
#SBATCH --output=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15/audit/slurm_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15/audit/slurm_%j.err

set -euo pipefail
CAMPAIGN=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
VERIFY="$CAMPAIGN/tools/verify_hermetic_bundle.py"
PROBE="$CAMPAIGN/tools/clean_room_probe.py"
read -r BUNDLE_SHA BUNDLE_NAME < "$CAMPAIGN/bundle/BUNDLE_SHA256SUMS"
BUNDLE="$CAMPAIGN/bundle/$BUNDLE_NAME"
OUT="$CAMPAIGN/audit/$SLURM_JOB_ID"
mkdir -p "$OUT"
finalize() {
  RC=$1
  date -Is > "$OUT/end_time.txt"
  printf '%s\n' "$RC" > "$OUT/rc"
  if [ "$RC" -eq 0 ]; then echo AUDIT_PASS > "$OUT/status"; else echo AUDIT_FAIL > "$OUT/status"; fi
}
date -Is > "$OUT/start_time.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$OUT/job_id"
hostname -f > "$OUT/hostname"
nvidia-smi -L > "$OUT/gpu.txt" 2>&1
DEPLOY=$(mktemp -d "${TMPDIR:-/tmp}/procgen-nm2-audit-${SLURM_JOB_ID}.XXXXXX")
EMPTY=$(mktemp -d "${TMPDIR:-/tmp}/procgen-nm2-empty-${SLURM_JOB_ID}.XXXXXX")
cleanup() { rm -rf "$DEPLOY" "$EMPTY"; }
on_exit() { RC=$?; cleanup; finalize "$RC"; }
trap on_exit EXIT
"$PY" "$VERIFY" "$BUNDLE" "$BUNDLE_SHA" "$DEPLOY" > "$OUT/bundle_verify.out" 2> "$OUT/bundle_verify.err"
(cd "$EMPTY"; env -u PYTHONPATH PYTHONNOUSERSITE=1 BUNDLE_ROOT="$DEPLOY" AUDIT_OUTPUT="$OUT/clean_room_audit.json" PROBE="$PROBE" "$PY" -c 'import os,sys; sys.path.insert(0, os.path.join(os.environ["BUNDLE_ROOT"], "code")); exec(compile(open(os.environ["PROBE"]).read(), os.environ["PROBE"], "exec"))') > "$OUT/clean_room.out" 2> "$OUT/clean_room.err"
grep -q CLEAN_ROOM_BUNDLE_AUDIT_PASS "$OUT/clean_room.out"
sha256sum "$BUNDLE" "$CAMPAIGN/bundle/BUNDLE_MANIFEST.json" "$VERIFY" "$PROBE" > "$OUT/input.sha256"
echo CLEAN_ROOM_BUNDLE_JOB_PASS
