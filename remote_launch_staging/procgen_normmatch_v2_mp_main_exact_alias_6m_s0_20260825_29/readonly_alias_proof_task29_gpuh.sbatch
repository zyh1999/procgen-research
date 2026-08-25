#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --job-name=pg-nm2-t29-proof
#SBATCH --output=/scratch/h99859yz/procgen_normmatch_v2_mp_main_exact_alias_6m_s0_20260825_29/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_normmatch_v2_mp_main_exact_alias_6m_s0_20260825_29/scheduler/%x_%j.err

set -euo pipefail
TASK15=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15
TASK17=/scratch/h99859yz/procgen_normmatch_v2_interpreter_path_audit_6m_s0_20260825_17
TASK18=/scratch/h99859yz/procgen_normmatch_v2_torch_generated_origin_audit_6m_s0_20260825_18
TASK21=/scratch/h99859yz/procgen_normmatch_v2_py39_lstat_compatibility_6m_s0_20260825_21
TASK23=/scratch/h99859yz/procgen_normmatch_v2_torch_pseudo_origin_nonreentrant_closure_20260825_23
TASK25=/scratch/h99859yz/procgen_normmatch_v2_torch_class_attribute_pseudo_origin_20260825_25
TASK28R=/scratch/h99859yz/procgen_normmatch_v2_exact_probe_alias_recovery_6m_s0_20260825_28r
CAMPAIGN=/scratch/h99859yz/procgen_normmatch_v2_mp_main_exact_alias_6m_s0_20260825_29
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
VERIFY="$TASK15/tools/verify_hermetic_bundle.py"
POLICY="$TASK18/tools/origin_safety.py"
SUPPORT="$TASK21/tools/policy_namespace_support.py"
BASE_SAFETY="$TASK17/tools/origin_safety.py"
BASELINE="$TASK18/provenance/reproduction_1.json"
PROBE="$TASK23/tools/runtime_closure_probe_task23.py"
BASE_EXTENSION="$TASK23/tools/pseudo_origin_extension.py"
BASE_CLASSIFIER="$TASK25/tools/class_attribute_classifier.py"
TASK28R_EXTENSION="$TASK28R/exact_probe_alias_extension_task28r.py"
PROOF_EXTENSION="$CAMPAIGN/readonly_alias_proof_extension_task29.py"
NEW_PREFLIGHT="$TASK28R/gpuh_preflight_normmatch_v2_task27.py"
AUDIT_HELPER="$TASK28R/ast_runtime_call_audit_task26.py"
SCIENCE_LAUNCHER="$TASK15/normmatch_v2_6m_hermetic_gpuh.sbatch"
read -r BUNDLE_SHA BUNDLE_NAME < "$TASK15/bundle/BUNDLE_SHA256SUMS"
BUNDLE="$TASK15/bundle/$BUNDLE_NAME"
OUT="$CAMPAIGN/readonly_proof/$SLURM_JOB_ID"
mkdir -p "$OUT"
finalize() { RC=$1; date -Is > "$OUT/end_time.txt"; printf '%s\n' "$RC" > "$OUT/rc"; if [ "$RC" -eq 0 ]; then echo READONLY_PROOF_PASS > "$OUT/status"; else echo READONLY_PROOF_FAIL > "$OUT/status"; fi; }
date -Is > "$OUT/start_time.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$OUT/job_id"
hostname -f > "$OUT/hostname"
nvidia-smi -L > "$OUT/gpu.txt" 2>&1
DEPLOY=$(mktemp -d "${TMPDIR:-/tmp}/task29-proof-deploy-${SLURM_JOB_ID}.XXXXXX")
EMPTY=$(mktemp -d "${TMPDIR:-/tmp}/task29-proof-empty-${SLURM_JOB_ID}.XXXXXX")
chmod 700 "$EMPTY"
cleanup() { rm -rf "$DEPLOY" "$EMPTY"; }
on_exit() { RC=$?; cleanup; finalize "$RC"; }
trap on_exit EXIT
"$PY" "$VERIFY" "$BUNDLE" "$BUNDLE_SHA" "$DEPLOY" > "$OUT/bundle_verify.out" 2> "$OUT/bundle_verify.err"
mv "$DEPLOY/frozen/gpuh_preflight_normmatch_v2.py" "$DEPLOY/frozen/gpuh_preflight_normmatch_v2_task25_frozen.py"
cp "$NEW_PREFLIGHT" "$DEPLOY/frozen/gpuh_preflight_normmatch_v2.py"
cp "$AUDIT_HELPER" "$DEPLOY/frozen/ast_runtime_call_audit_task26.py"
set +e
(cd "$EMPTY"; env -u PYTHONPATH PYTHONNOUSERSITE=1 NORMMATCH_V2_ORIGIN_POLICY_PATH="$POLICY" NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT="$SUPPORT" TASK17_ORIGIN_SAFETY_BASE="$BASE_SAFETY" TORCH_GENERATED_PROVENANCE_BASELINE="$BASELINE" TASK23_PSEUDO_ORIGIN_BASE="$BASE_EXTENSION" TASK23_PSEUDO_ORIGIN_EXTENSION="$PROOF_EXTENSION" TASK28R_CLASSIFIER_BASE="$BASE_CLASSIFIER" TASK28R_PROCESS_ROLE=closure-audit-entrypoint TASK28R_FROZEN_SCIENCE_LAUNCHER="$SCIENCE_LAUNCHER" TASK29_TASK28R_EXTENSION="$TASK28R_EXTENSION" TASK29_READONLY_ALIAS_LEDGER="$OUT/readonly_alias_ledger.json" FORBIDDEN_SOURCE_ROOTS="/Users/user/.codex/worktrees/7edd/procgen:/Users/user/Documents/procgen:$CAMPAIGN/staging" "$PY" "$PROBE" "$DEPLOY" "$OUT/expected_downstream_reproduction.json" "$EMPTY") > "$OUT/probe.out" 2> "$OUT/probe.err"
PROBE_RC=$?
set -e
printf '%s\n' "$PROBE_RC" > "$OUT/probe_rc"
test "$PROBE_RC" -ne 0
grep -q 'TASK29_READONLY_LIVE_ALIAS_PROOF_PASS' "$OUT/readonly_alias_ledger.json"
grep -q 'bundle module absent from manifest or hash mismatch: __mp_main__' "$OUT/probe.err"
sha256sum "$PROOF_EXTENSION" "$TASK28R_EXTENSION" "$PROBE" /usr/lib64/python3.9/multiprocessing/__init__.py /usr/lib64/python3.9/multiprocessing/spawn.py > "$OUT/input.sha256"
echo TASK29_READONLY_ALIAS_PROOF_PASS
