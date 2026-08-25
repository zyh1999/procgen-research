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
#SBATCH --job-name=pg-nm2-id-audit
#SBATCH --output=/scratch/h99859yz/procgen_normmatch_v2_policy_path_identity_recovery_6m_s0_20260825_20/audit/slurm_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_normmatch_v2_policy_path_identity_recovery_6m_s0_20260825_20/audit/slurm_%j.err

set -euo pipefail
TASK15=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15
TASK16=/scratch/h99859yz/procgen_normmatch_v2_syspath_audit_recovery_6m_s0_20260825_16
TASK17=/scratch/h99859yz/procgen_normmatch_v2_interpreter_path_audit_6m_s0_20260825_17
TASK18=/scratch/h99859yz/procgen_normmatch_v2_torch_generated_origin_audit_6m_s0_20260825_18
TASK19=/scratch/h99859yz/procgen_normmatch_v2_bare_exec_namespace_recovery_6m_s0_20260825_19
CAMPAIGN=/scratch/h99859yz/procgen_normmatch_v2_policy_path_identity_recovery_6m_s0_20260825_20
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
VERIFY="$TASK15/tools/verify_hermetic_bundle.py"
POLICY="$TASK18/tools/origin_safety.py"
SUPPORT="$CAMPAIGN/tools/policy_namespace_support.py"
BASE_SAFETY="$TASK17/tools/origin_safety.py"
BASELINE="$TASK18/provenance/reproduction_1.json"
PREPARE="$TASK19/tools/prepare_designated_empty_task19.py"
PROBE="$TASK19/tools/clean_room_probe_task19.py"
read -r BUNDLE_SHA BUNDLE_NAME < "$TASK15/bundle/BUNDLE_SHA256SUMS"
BUNDLE="$TASK15/bundle/$BUNDLE_NAME"
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
DEPLOY=$(mktemp -d "${TMPDIR:-/tmp}/procgen-nm2-task20-deploy-${SLURM_JOB_ID}.XXXXXX")
EMPTY=$(mktemp -d "${TMPDIR:-/tmp}/procgen-nm2-task20-empty-${SLURM_JOB_ID}.XXXXXX")
cleanup() { rm -rf "$DEPLOY" "$EMPTY"; }
on_exit() { RC=$?; cleanup; finalize "$RC"; }
trap on_exit EXIT
"$PY" "$VERIFY" "$BUNDLE" "$BUNDLE_SHA" "$DEPLOY" > "$OUT/bundle_verify.out" 2> "$OUT/bundle_verify.err"
NORMMATCH_V2_ORIGIN_POLICY_PATH="$POLICY" NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT="$SUPPORT" TASK17_ORIGIN_SAFETY_BASE="$BASE_SAFETY" TORCH_GENERATED_PROVENANCE_BASELINE="$BASELINE" "$PY" "$PREPARE" "$EMPTY" "$OUT/designated_empty_prestart.json" "$OUT/origin_policy_prestart.json" > "$OUT/designated_empty_prestart.out" 2> "$OUT/designated_empty_prestart.err"
(cd "$EMPTY"; env -u PYTHONPATH PYTHONNOUSERSITE=1 BUNDLE_ROOT="$DEPLOY" AUDIT_OUTPUT="$OUT/clean_room_audit.json" IMPORT_ORIGIN_OUTPUT="$OUT/import_origin_manifest.json" DESIGNATED_EMPTY="$EMPTY" DESIGNATED_PRESTART_JSON="$OUT/designated_empty_prestart.json" ORIGIN_POLICY_PRESTART_LEDGER="$OUT/origin_policy_prestart.json" NORMMATCH_V2_ORIGIN_POLICY_PATH="$POLICY" NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT="$SUPPORT" TASK17_ORIGIN_SAFETY_BASE="$BASE_SAFETY" TORCH_GENERATED_PROVENANCE_BASELINE="$BASELINE" PROBE="$PROBE" FORBIDDEN_SOURCE_ROOTS="/Users/user/.codex/worktrees/7edd/procgen:/Users/user/Documents/procgen:$TASK15/staging:$TASK16/staging:$TASK17/staging:$TASK18/staging:$TASK19/staging:$CAMPAIGN/staging" "$PY" -c 'import os,sys; sys.path.insert(0, os.path.join(os.environ["BUNDLE_ROOT"], "code")); exec(compile(open(os.environ["PROBE"], "rb").read(), os.environ["PROBE"], "exec"))') > "$OUT/clean_room.out" 2> "$OUT/clean_room.err"
grep -q TASK19_EXPLICIT_POLICY_RUNTIME_PASS "$OUT/clean_room.out"
grep -q CLEAN_ROOM_BUNDLE_AUDIT_PASS "$OUT/clean_room.out"
grep -q IMPORT_ORIGIN_AUDIT_PASS "$OUT/clean_room.out"
sha256sum "$BUNDLE" "$TASK15/bundle/BUNDLE_MANIFEST.json" "$VERIFY" "$POLICY" "$SUPPORT" "$BASE_SAFETY" "$BASELINE" "$PREPARE" "$PROBE" > "$OUT/input.sha256"
echo TASK20_CLEAN_ROOM_AUDIT_PASS
