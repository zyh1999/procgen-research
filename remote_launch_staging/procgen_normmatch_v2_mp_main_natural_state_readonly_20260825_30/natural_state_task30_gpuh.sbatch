#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --job-name=pg-nm2-t30-ro
#SBATCH --output=/scratch/h99859yz/procgen_normmatch_v2_mp_main_natural_state_readonly_20260825_30/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_normmatch_v2_mp_main_natural_state_readonly_20260825_30/scheduler/%x_%j.err

set -euo pipefail
TASK15=/scratch/h99859yz/procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15
TASK17=/scratch/h99859yz/procgen_normmatch_v2_interpreter_path_audit_6m_s0_20260825_17
TASK18=/scratch/h99859yz/procgen_normmatch_v2_torch_generated_origin_audit_6m_s0_20260825_18
TASK21=/scratch/h99859yz/procgen_normmatch_v2_py39_lstat_compatibility_6m_s0_20260825_21
TASK23=/scratch/h99859yz/procgen_normmatch_v2_torch_pseudo_origin_nonreentrant_closure_20260825_23
TASK25=/scratch/h99859yz/procgen_normmatch_v2_torch_class_attribute_pseudo_origin_20260825_25
TASK28R=/scratch/h99859yz/procgen_normmatch_v2_exact_probe_alias_recovery_6m_s0_20260825_28r
CAMPAIGN=/scratch/h99859yz/procgen_normmatch_v2_mp_main_natural_state_readonly_20260825_30
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
VERIFY="$TASK15/tools/verify_hermetic_bundle.py"
POLICY="$TASK18/tools/origin_safety.py"
SUPPORT="$TASK21/tools/policy_namespace_support.py"
BASE_SAFETY="$TASK17/tools/origin_safety.py"
BASELINE="$TASK18/provenance/reproduction_1.json"
PROBE="$TASK23/tools/runtime_closure_probe_task23.py"
BASE_EXTENSION="$TASK23/tools/pseudo_origin_extension.py"
BASE_CLASSIFIER="$TASK25/tools/class_attribute_classifier.py"
EXTENSION="$TASK28R/exact_probe_alias_extension_task28r.py"
NEW_PREFLIGHT="$TASK28R/gpuh_preflight_normmatch_v2_task27.py"
AUDIT_HELPER="$TASK28R/ast_runtime_call_audit_task26.py"
SCIENCE_LAUNCHER="$TASK15/normmatch_v2_6m_hermetic_gpuh.sbatch"
OBSERVER="$CAMPAIGN/observer"
ANALYZE="$CAMPAIGN/analyze_natural_state_task30.py"
TRAINER_REMOTE="$TASK15/staging/code/train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
read -r BUNDLE_SHA BUNDLE_NAME < "$TASK15/bundle/BUNDLE_SHA256SUMS"
BUNDLE="$TASK15/bundle/$BUNDLE_NAME"
OUT="$CAMPAIGN/provenance/$SLURM_JOB_ID"
mkdir -p "$OUT"
date -Is > "$OUT/start_time.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$OUT/job_id"
hostname -f > "$OUT/hostname"
nvidia-smi -L > "$OUT/gpu.txt" 2>&1
DEPLOYS=()
EMPTIES=()
cleanup() { for value in "${DEPLOYS[@]}" "${EMPTIES[@]}"; do test -z "$value" || rm -rf "$value"; done; }
finalize() { RC=$1; date -Is > "$OUT/end_time.txt"; printf '%s\n' "$RC" > "$OUT/rc"; if [ "$RC" -eq 0 ]; then echo READONLY_PROVENANCE_COMPLETE > "$OUT/status"; else echo READONLY_PROVENANCE_FAIL > "$OUT/status"; fi; }
on_exit() { RC=$?; cleanup; finalize "$RC"; }
trap on_exit EXIT

for NAME in obs1 obs2 obs3 control; do
  CASE="$OUT/$NAME"
  mkdir -p "$CASE"
  DEPLOY=$(mktemp -d "${TMPDIR:-/tmp}/task30-deploy-${NAME}-${SLURM_JOB_ID}.XXXXXX")
  EMPTY=$(mktemp -d "${TMPDIR:-/tmp}/task30-empty-${NAME}-${SLURM_JOB_ID}.XXXXXX")
  DEPLOYS+=("$DEPLOY")
  EMPTIES+=("$EMPTY")
  chmod 700 "$EMPTY"
  "$PY" "$VERIFY" "$BUNDLE" "$BUNDLE_SHA" "$DEPLOY" > "$CASE/bundle_verify.out" 2> "$CASE/bundle_verify.err"
  mv "$DEPLOY/frozen/gpuh_preflight_normmatch_v2.py" "$DEPLOY/frozen/gpuh_preflight_normmatch_v2_task25_frozen.py"
  cp "$NEW_PREFLIGHT" "$DEPLOY/frozen/gpuh_preflight_normmatch_v2.py"
  cp "$AUDIT_HELPER" "$DEPLOY/frozen/ast_runtime_call_audit_task26.py"
  TRAINER="$DEPLOY/code/train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
  set +e
  if [ "$NAME" = control ]; then
    (cd "$EMPTY"; env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 NORMMATCH_V2_ORIGIN_POLICY_PATH="$POLICY" NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT="$SUPPORT" TASK17_ORIGIN_SAFETY_BASE="$BASE_SAFETY" TORCH_GENERATED_PROVENANCE_BASELINE="$BASELINE" TASK23_PSEUDO_ORIGIN_BASE="$BASE_EXTENSION" TASK23_PSEUDO_ORIGIN_EXTENSION="$EXTENSION" TASK28R_CLASSIFIER_BASE="$BASE_CLASSIFIER" TASK28R_PROCESS_ROLE=closure-audit-entrypoint TASK28R_FROZEN_SCIENCE_LAUNCHER="$SCIENCE_LAUNCHER" FORBIDDEN_SOURCE_ROOTS="/Users/user/.codex/worktrees/7edd/procgen:/Users/user/Documents/procgen:$CAMPAIGN/staging" "$PY" -X importtime "$PROBE" "$DEPLOY" "$CASE/expected_reproduction.json" "$EMPTY") > "$CASE/probe.out" 2> "$CASE/importtime.err"
  else
    (cd "$EMPTY"; env PYTHONPATH="$OBSERVER" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 TASK30_PROBE_PATH="$PROBE" TASK30_PREFLIGHT_BASE_PATH="$DEPLOY/frozen/gpuh_preflight_normmatch_v2_task25_frozen.py" TASK30_TRAINER_PATH="$TRAINER" TASK30_OBSERVER_LEDGER="$CASE/natural_state_ledger.json" NORMMATCH_V2_ORIGIN_POLICY_PATH="$POLICY" NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT="$SUPPORT" TASK17_ORIGIN_SAFETY_BASE="$BASE_SAFETY" TORCH_GENERATED_PROVENANCE_BASELINE="$BASELINE" TASK23_PSEUDO_ORIGIN_BASE="$BASE_EXTENSION" TASK23_PSEUDO_ORIGIN_EXTENSION="$EXTENSION" TASK28R_CLASSIFIER_BASE="$BASE_CLASSIFIER" TASK28R_PROCESS_ROLE=closure-audit-entrypoint TASK28R_FROZEN_SCIENCE_LAUNCHER="$SCIENCE_LAUNCHER" FORBIDDEN_SOURCE_ROOTS="/Users/user/.codex/worktrees/7edd/procgen:/Users/user/Documents/procgen:$CAMPAIGN/staging" "$PY" -X importtime "$PROBE" "$DEPLOY" "$CASE/expected_reproduction.json" "$EMPTY") > "$CASE/probe.out" 2> "$CASE/importtime.err"
  fi
  PROBE_RC=$?
  set -e
  printf '%s\n' "$PROBE_RC" > "$CASE/probe_rc"
  cp "$CASE/importtime.err" "$CASE/probe.err"
  test "$PROBE_RC" -ne 0
  grep -q 'bundle module absent from manifest or hash mismatch: __mp_main__' "$CASE/probe.err"
  if [ "$NAME" != control ]; then
    test -s "$CASE/natural_state_ledger.json"
  fi
done

"$PY" "$ANALYZE" "$OUT" "${DEPLOYS[0]}" "${DEPLOYS[1]}" "${DEPLOYS[2]}" "${DEPLOYS[3]}" > "$OUT/analysis.out" 2> "$OUT/analysis.err"
grep -Eq 'TASK30_UNIQUE_CONCLUSION=(NATURAL_MP_MAIN_RELATIONSHIP_PROVEN|NO_SAFE_ALIAS_RELATION|OBSERVER_PERTURBED|INSUFFICIENT_EVIDENCE)' "$OUT/analysis.out"
sha256sum "$PROBE" "$EXTENSION" "$NEW_PREFLIGHT" "${DEPLOYS[0]}/frozen/gpuh_preflight_normmatch_v2_task25_frozen.py" "$AUDIT_HELPER" "$OBSERVER/sitecustomize.py" "$ANALYZE" "$BUNDLE" "$TASK15/bundle/BUNDLE_MANIFEST.json" "$SCIENCE_LAUNCHER" > "$OUT/input.sha256"
echo TASK30_READONLY_PROVENANCE_COMPLETE
