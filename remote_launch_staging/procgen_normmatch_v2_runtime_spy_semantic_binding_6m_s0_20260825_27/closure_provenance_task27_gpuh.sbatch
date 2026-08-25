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
#SBATCH --job-name=pg-nm2-t27-closure
#SBATCH --output=/scratch/h99859yz/procgen_normmatch_v2_runtime_spy_semantic_binding_6m_s0_20260825_27/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_normmatch_v2_runtime_spy_semantic_binding_6m_s0_20260825_27/scheduler/%x_%j.err

set -euo pipefail
TASK15=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15
TASK17=/scratch/h99859yz/procgen_normmatch_v2_interpreter_path_audit_6m_s0_20260825_17
TASK18=/scratch/h99859yz/procgen_normmatch_v2_torch_generated_origin_audit_6m_s0_20260825_18
TASK21=/scratch/h99859yz/procgen_normmatch_v2_py39_lstat_compatibility_6m_s0_20260825_21
TASK23=/scratch/h99859yz/procgen_normmatch_v2_torch_pseudo_origin_nonreentrant_closure_20260825_23
TASK25=/scratch/h99859yz/procgen_normmatch_v2_torch_class_attribute_pseudo_origin_20260825_25
CAMPAIGN=/scratch/h99859yz/procgen_normmatch_v2_runtime_spy_semantic_binding_6m_s0_20260825_27
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
VERIFY="$TASK15/tools/verify_hermetic_bundle.py"
POLICY="$TASK18/tools/origin_safety.py"
SUPPORT="$TASK21/tools/policy_namespace_support.py"
BASE_SAFETY="$TASK17/tools/origin_safety.py"
BASELINE="$TASK18/provenance/reproduction_1.json"
PROBE="$TASK23/tools/runtime_closure_probe_task23.py"
ANALYZE="$TASK23/tools/analyze_closure_reproductions.py"
BASE_EXTENSION="$TASK23/tools/pseudo_origin_extension.py"
EXTENSION="$TASK25/tools/class_attribute_classifier.py"
NEW_PREFLIGHT="$CAMPAIGN/gpuh_preflight_normmatch_v2_task27.py"
AUDIT_HELPER="$CAMPAIGN/ast_runtime_call_audit_task26.py"
read -r BUNDLE_SHA BUNDLE_NAME < "$TASK15/bundle/BUNDLE_SHA256SUMS"
BUNDLE="$TASK15/bundle/$BUNDLE_NAME"
OUT="$CAMPAIGN/provenance/$SLURM_JOB_ID"
mkdir -p "$OUT"
finalize() { RC=$1; date -Is > "$OUT/end_time.txt"; printf '%s\n' "$RC" > "$OUT/rc"; if [ "$RC" -eq 0 ]; then echo CLOSURE_PASS > "$OUT/status"; else echo CLOSURE_FAIL > "$OUT/status"; fi; }
date -Is > "$OUT/start_time.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$OUT/job_id"
hostname -f > "$OUT/hostname"
nvidia-smi -L > "$OUT/gpu.txt" 2>&1
DEPLOY1=$(mktemp -d "${TMPDIR:-/tmp}/task27-deploy1-${SLURM_JOB_ID}.XXXXXX")
DEPLOY2=$(mktemp -d "${TMPDIR:-/tmp}/task27-deploy2-${SLURM_JOB_ID}.XXXXXX")
EMPTY1=$(mktemp -d "${TMPDIR:-/tmp}/task27-empty1-${SLURM_JOB_ID}.XXXXXX")
EMPTY2=$(mktemp -d "${TMPDIR:-/tmp}/task27-empty2-${SLURM_JOB_ID}.XXXXXX")
chmod 700 "$EMPTY1" "$EMPTY2"
cleanup() { rm -rf "$DEPLOY1" "$DEPLOY2" "$EMPTY1" "$EMPTY2"; }
on_exit() { RC=$?; cleanup; finalize "$RC"; }
trap on_exit EXIT
"$PY" "$VERIFY" "$BUNDLE" "$BUNDLE_SHA" "$DEPLOY1" > "$OUT/bundle_verify_1.out" 2> "$OUT/bundle_verify_1.err"
"$PY" "$VERIFY" "$BUNDLE" "$BUNDLE_SHA" "$DEPLOY2" > "$OUT/bundle_verify_2.out" 2> "$OUT/bundle_verify_2.err"
for DEPLOY in "$DEPLOY1" "$DEPLOY2"; do
  mv "$DEPLOY/frozen/gpuh_preflight_normmatch_v2.py" "$DEPLOY/frozen/gpuh_preflight_normmatch_v2_task25_frozen.py"
  cp "$NEW_PREFLIGHT" "$DEPLOY/frozen/gpuh_preflight_normmatch_v2.py"
  cp "$AUDIT_HELPER" "$DEPLOY/frozen/ast_runtime_call_audit_task26.py"
done
(cd "$EMPTY1"; env -u PYTHONPATH PYTHONNOUSERSITE=1 NORMMATCH_V2_ORIGIN_POLICY_PATH="$POLICY" NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT="$SUPPORT" TASK17_ORIGIN_SAFETY_BASE="$BASE_SAFETY" TORCH_GENERATED_PROVENANCE_BASELINE="$BASELINE" TASK23_PSEUDO_ORIGIN_BASE="$BASE_EXTENSION" TASK23_PSEUDO_ORIGIN_EXTENSION="$EXTENSION" FORBIDDEN_SOURCE_ROOTS="/Users/user/.codex/worktrees/7edd/procgen:/Users/user/Documents/procgen:$CAMPAIGN/staging" "$PY" "$PROBE" "$DEPLOY1" "$OUT/reproduction_1.json" "$EMPTY1") > "$OUT/reproduction_1.out" 2> "$OUT/reproduction_1.err"
(cd "$EMPTY2"; env -u PYTHONPATH PYTHONNOUSERSITE=1 NORMMATCH_V2_ORIGIN_POLICY_PATH="$POLICY" NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT="$SUPPORT" TASK17_ORIGIN_SAFETY_BASE="$BASE_SAFETY" TORCH_GENERATED_PROVENANCE_BASELINE="$BASELINE" TASK23_PSEUDO_ORIGIN_BASE="$BASE_EXTENSION" TASK23_PSEUDO_ORIGIN_EXTENSION="$EXTENSION" FORBIDDEN_SOURCE_ROOTS="/Users/user/.codex/worktrees/7edd/procgen:/Users/user/Documents/procgen:$CAMPAIGN/staging" "$PY" "$PROBE" "$DEPLOY2" "$OUT/reproduction_2.json" "$EMPTY2") > "$OUT/reproduction_2.out" 2> "$OUT/reproduction_2.err"
"$PY" "$ANALYZE" "$OUT/reproduction_1.json" "$OUT/reproduction_2.json" "$OUT/closure_decision.json" > "$OUT/analysis.out" 2> "$OUT/analysis.err"
grep -q TASK23_CLOSURE_PASS "$OUT/analysis.out"
sha256sum "$BUNDLE" "$PROBE" "$ANALYZE" "$TASK23/tools/nonreentrant_audit_hook.py" "$BASE_EXTENSION" "$EXTENSION" "$NEW_PREFLIGHT" "$AUDIT_HELPER" > "$OUT/input.sha256"
echo TASK27_CLOSURE_PASS
