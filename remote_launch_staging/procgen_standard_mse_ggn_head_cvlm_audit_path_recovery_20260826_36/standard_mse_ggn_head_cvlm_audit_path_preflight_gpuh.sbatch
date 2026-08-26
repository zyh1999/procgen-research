#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_audit_path_recovery_20260826_36/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_audit_path_recovery_20260826_36/scheduler/%x_%j.err

set -uo pipefail
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
case "$ENV_NAME" in
  bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;;
  *) exit 64 ;;
esac
TASK_ID=PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-AUDIT-PATH-RECOVERY-20260826-36
METHOD=DET_STANDARD_MSE_GGN_HEAD_CVLM_V1
CAMPAIGN=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_audit_path_recovery_20260826_36
TASK35_CAMPAIGN=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_hermetic_preflight_20260826_35r
EVIDENCE="$CAMPAIGN/preflight/$ENV_NAME"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
ARCHIVE_SHA=3a9d9720ae7b3c9e6d13a2fd63521d51bba8cb62e7ae7ae2498553b57c00609f
MANIFEST_SHA=287a744078b10054d107974125bac6b5fac43fd944b6200ec54720cd2695c9af
ARCHIVE="$TASK35_CAMPAIGN/bundle/task35r_source_${ARCHIVE_SHA}.tar"
VERIFY="$TASK35_CAMPAIGN/frozen/verify_hermetic_bundle_task35r.py"
IMPORT_SMOKE="$TASK35_CAMPAIGN/frozen/hermetic_import_smoke_task35r.py"
ADAPTER="$CAMPAIGN/frozen/audit_path_adapter_task36.py"
LAUNCHER="$CAMPAIGN/frozen/standard_mse_ggn_head_cvlm_audit_path_preflight_gpuh.sbatch"
TRAINER_SHA=ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc
CONFIG_SHA=52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132

if [ -e "$EVIDENCE" ]; then
  echo "preflight root collision: $EVIDENCE" >&2
  exit 90
fi
mkdir -p "$EVIDENCE"
echo PREFLIGHT_RUNNING > "$EVIDENCE/status"
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE/job_id"
printf '%s\n' "$TASK_ID" > "$EVIDENCE/task_id.txt"
printf '%s\n' "$METHOD" > "$EVIDENCE/method.txt"
hostname -f > "$EVIDENCE/hostname"
nvidia-smi -L > "$EVIDENCE/gpu.txt" 2>&1
scontrol show job -dd "$SLURM_JOB_ID" > "$EVIDENCE/scontrol_job.txt" 2>&1 || true
sha256sum "$ARCHIVE" "$VERIFY" "$IMPORT_SMOKE" "$ADAPTER" "$LAUNCHER" > "$EVIDENCE/deployment_input.sha256"

fail_precheck() {
  local rc=$1
  printf '%s\n' "$rc" > "$EVIDENCE/rc"
  echo PRECHECK_FAIL > "$EVIDENCE/status"
  exit "$rc"
}

BUNDLE_ROOT="$EVIDENCE/bundle"
"$PY" "$VERIFY" "$ARCHIVE" "$ARCHIVE_SHA" "$MANIFEST_SHA" "$BUNDLE_ROOT" \
  > "$EVIDENCE/bundle_verify.out" 2> "$EVIDENCE/bundle_verify.err" || fail_precheck $?

TRAINER="$BUNDLE_ROOT/code/train_shared_det_standard_mse_ggn_head_cvlm_v1.py"
CONFIG="$BUNDLE_ROOT/code/configs/adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml"
PREFLIGHT="$BUNDLE_ROOT/frozen/gpuh_preflight.py"
AUDIT="$BUNDLE_ROOT/frozen/audit_task34r.py"
ORIGINAL_SCIENCE_LAUNCHER="$BUNDLE_ROOT/frozen/standard_mse_ggn_head_cvlm_6m_gpuh.sbatch"
sha256sum "$TRAINER" "$CONFIG" "$PREFLIGHT" "$AUDIT" "$ORIGINAL_SCIENCE_LAUNCHER" \
  > "$EVIDENCE/frozen_input.sha256"

EMPTY_CWD="$EVIDENCE/empty_cwd"
mkdir "$EMPTY_CWD"
(
  cd "$EMPTY_CWD"
  env -u PYTHONPATH "$PY" "$IMPORT_SMOKE" "$BUNDLE_ROOT" "$EVIDENCE/module_origins.json"
) > "$EVIDENCE/import_smoke.out" 2> "$EVIDENCE/import_smoke.err" || fail_precheck $?

(
  cd "$EMPTY_CWD"
  env -u PYTHONPATH \
    TASK07_SOURCE_ROOT="$BUNDLE_ROOT/audit_sources/task07" \
    TASK13_SOURCE_ROOT="$BUNDLE_ROOT/audit_sources/task13" \
    TASK32_SOURCE_ROOT="$BUNDLE_ROOT/audit_sources/task32" \
    "$PY" "$ADAPTER" "$BUNDLE_ROOT" \
      "$EVIDENCE/historical_scaling_ledger.json" "$EVIDENCE/trainer_identity_ledger.json"
) > "$EVIDENCE/historical_audit.out" 2> "$EVIDENCE/historical_audit.err" || fail_precheck $?
grep -q TASK34R_HISTORICAL_SCALING_AUDIT_PASS "$EVIDENCE/historical_audit.out" || fail_precheck 73
grep -q TASK36_AUDIT_PATH_ADAPTER_PASS "$EVIDENCE/historical_audit.out" || fail_precheck 74

(
  cd "$EMPTY_CWD"
  env -u PYTHONPATH PROCGEN_ENV="$ENV_NAME" "$PY" "$PREFLIGHT" \
    "$TRAINER" "$CONFIG" "$EVIDENCE/parameter_partition.json" "$TRAINER_SHA" "$CONFIG_SHA"
) > "$EVIDENCE/preflight.out" 2> "$EVIDENCE/preflight.err"
rc=$?
if [ "$rc" -ne 0 ]; then
  fail_precheck "$rc"
fi
grep -q GPUH_STANDARD_MSE_GGN_HEAD_CVLM_COMPATIBILITY_PASS "$EVIDENCE/preflight.out" || fail_precheck 75
printf '%s\n' 0 > "$EVIDENCE/rc"
echo PRECHECK_PASS > "$EVIDENCE/status"
touch "$EVIDENCE/precheck_pass.marker"
exit 0
