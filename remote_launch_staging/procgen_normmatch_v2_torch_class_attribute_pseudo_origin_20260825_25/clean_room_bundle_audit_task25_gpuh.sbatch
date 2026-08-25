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
#SBATCH --job-name=pg-nm2-t25-audit
#SBATCH --output=/scratch/h99859yz/procgen_normmatch_v2_torch_class_attribute_pseudo_origin_20260825_25/audit/slurm_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_normmatch_v2_torch_class_attribute_pseudo_origin_20260825_25/audit/slurm_%j.err

set -euo pipefail
TASK15=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15
TASK17=/scratch/h99859yz/procgen_normmatch_v2_interpreter_path_audit_6m_s0_20260825_17
TASK18=/scratch/h99859yz/procgen_normmatch_v2_torch_generated_origin_audit_6m_s0_20260825_18
TASK19=/scratch/h99859yz/procgen_normmatch_v2_bare_exec_namespace_recovery_6m_s0_20260825_19
TASK21=/scratch/h99859yz/procgen_normmatch_v2_py39_lstat_compatibility_6m_s0_20260825_21
TASK23=/scratch/h99859yz/procgen_normmatch_v2_torch_pseudo_origin_nonreentrant_closure_20260825_23
CAMPAIGN=/scratch/h99859yz/procgen_normmatch_v2_torch_class_attribute_pseudo_origin_20260825_25
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
VERIFY="$TASK15/tools/verify_hermetic_bundle.py"
POLICY="$TASK18/tools/origin_safety.py"
SUPPORT="$TASK21/tools/policy_namespace_support.py"
BASE_SAFETY="$TASK17/tools/origin_safety.py"
BASELINE="$TASK18/provenance/reproduction_1.json"
PREPARE="$TASK19/tools/prepare_designated_empty_task19.py"
PROBE="$TASK23/tools/clean_room_probe_task23.py"
BASE_EXTENSION="$TASK23/tools/pseudo_origin_extension.py"
EXTENSION="$CAMPAIGN/tools/class_attribute_classifier.py"
read -r BUNDLE_SHA BUNDLE_NAME < "$TASK15/bundle/BUNDLE_SHA256SUMS"
BUNDLE="$TASK15/bundle/$BUNDLE_NAME"
OUT="$CAMPAIGN/audit/$SLURM_JOB_ID"
mkdir -p "$OUT"
finalize() { RC=$1; date -Is > "$OUT/end_time.txt"; printf '%s\n' "$RC" > "$OUT/rc"; if [ "$RC" -eq 0 ]; then echo AUDIT_PASS > "$OUT/status"; else echo AUDIT_FAIL > "$OUT/status"; fi; }
date -Is > "$OUT/start_time.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$OUT/job_id"
hostname -f > "$OUT/hostname"
nvidia-smi -L > "$OUT/gpu.txt" 2>&1
DEPLOY=$(mktemp -d "${TMPDIR:-/tmp}/task25-audit-deploy-${SLURM_JOB_ID}.XXXXXX")
EMPTY=$(mktemp -d "${TMPDIR:-/tmp}/task25-audit-empty-${SLURM_JOB_ID}.XXXXXX")
cleanup() { rm -rf "$DEPLOY" "$EMPTY"; }
on_exit() { RC=$?; cleanup; finalize "$RC"; }
trap on_exit EXIT
"$PY" "$VERIFY" "$BUNDLE" "$BUNDLE_SHA" "$DEPLOY" > "$OUT/bundle_verify.out" 2> "$OUT/bundle_verify.err"
NORMMATCH_V2_ORIGIN_POLICY_PATH="$POLICY" NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT="$SUPPORT" TASK17_ORIGIN_SAFETY_BASE="$BASE_SAFETY" TORCH_GENERATED_PROVENANCE_BASELINE="$BASELINE" "$PY" "$PREPARE" "$EMPTY" "$OUT/designated_empty_prestart.json" "$OUT/origin_policy_prestart.json" > "$OUT/designated_empty_prestart.out" 2> "$OUT/designated_empty_prestart.err"
(cd "$EMPTY"; env -u PYTHONPATH PYTHONNOUSERSITE=1 BUNDLE_ROOT="$DEPLOY" AUDIT_OUTPUT="$OUT/clean_room_audit.json" IMPORT_ORIGIN_OUTPUT="$OUT/import_origin_manifest.json" DESIGNATED_EMPTY="$EMPTY" DESIGNATED_PRESTART_JSON="$OUT/designated_empty_prestart.json" ORIGIN_POLICY_PRESTART_LEDGER="$OUT/origin_policy_prestart.json" NORMMATCH_V2_ORIGIN_POLICY_PATH="$POLICY" NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT="$SUPPORT" TASK17_ORIGIN_SAFETY_BASE="$BASE_SAFETY" TORCH_GENERATED_PROVENANCE_BASELINE="$BASELINE" TASK23_PSEUDO_ORIGIN_BASE="$BASE_EXTENSION" TASK23_PSEUDO_ORIGIN_EXTENSION="$EXTENSION" PROBE="$PROBE" FORBIDDEN_SOURCE_ROOTS="/Users/user/.codex/worktrees/7edd/procgen:/Users/user/Documents/procgen:$CAMPAIGN/staging" "$PY" -c 'import os,sys; sys.path.insert(0, os.path.join(os.environ["BUNDLE_ROOT"], "code")); exec(compile(open(os.environ["PROBE"], "rb").read(), os.environ["PROBE"], "exec"))') > "$OUT/clean_room.out" 2> "$OUT/clean_room.err"
grep -q TASK23_PSEUDO_ORIGIN_FORMAL_AUDIT_PASS "$OUT/clean_room.out"
grep -q CLEAN_ROOM_BUNDLE_AUDIT_PASS "$OUT/clean_room.out"
grep -q IMPORT_ORIGIN_AUDIT_PASS "$OUT/clean_room.out"
sha256sum "$BUNDLE" "$POLICY" "$SUPPORT" "$BASE_SAFETY" "$BASELINE" "$PROBE" "$BASE_EXTENSION" "$EXTENSION" > "$OUT/input.sha256"
echo TASK25_FORMAL_CLEAN_ROOM_AUDIT_PASS
