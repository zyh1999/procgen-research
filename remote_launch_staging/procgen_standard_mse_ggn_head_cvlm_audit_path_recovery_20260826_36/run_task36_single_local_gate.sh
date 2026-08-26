#!/bin/bash --login
set -uo pipefail

TASK_ID=PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-AUDIT-PATH-RECOVERY-20260826-36
CAMPAIGN=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_audit_path_recovery_20260826_36
TASK35_CAMPAIGN=/scratch/h99859yz/procgen_standard_mse_ggn_head_cvlm_hermetic_preflight_20260826_35r
GATE="$CAMPAIGN/local_gate"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
ARCHIVE_SHA=3a9d9720ae7b3c9e6d13a2fd63521d51bba8cb62e7ae7ae2498553b57c00609f
MANIFEST_SHA=287a744078b10054d107974125bac6b5fac43fd944b6200ec54720cd2695c9af
ARCHIVE="$TASK35_CAMPAIGN/bundle/task35r_source_${ARCHIVE_SHA}.tar"
VERIFY="$TASK35_CAMPAIGN/frozen/verify_hermetic_bundle_task35r.py"
IMPORT_SMOKE="$TASK35_CAMPAIGN/frozen/hermetic_import_smoke_task35r.py"
ADAPTER="$CAMPAIGN/frozen/audit_path_adapter_task36.py"

if [ -e "$GATE" ]; then
  echo "Task36 complete local gate is single-use: $GATE exists" >&2
  exit 91
fi
mkdir -p "$GATE"
printf '%s\n' "$TASK_ID" > "$GATE/task_id.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$GATE/started_utc.txt"
echo LOCAL_GATE_RUNNING > "$GATE/status"

fail_gate() {
  local rc=$1
  printf '%s\n' "$rc" > "$GATE/rc"
  echo LOCAL_GATE_FAIL > "$GATE/status"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$GATE/ended_utc.txt"
  exit "$rc"
}

sha256sum "$ARCHIVE" "$VERIFY" "$IMPORT_SMOKE" "$ADAPTER" > "$GATE/input.sha256" || fail_gate $?
test "$(sha256sum "$ARCHIVE" | awk '{print $1}')" = "$ARCHIVE_SHA" || fail_gate 61

BUNDLE_ROOT="$GATE/bundle"
"$PY" "$VERIFY" "$ARCHIVE" "$ARCHIVE_SHA" "$MANIFEST_SHA" "$BUNDLE_ROOT" \
  > "$GATE/bundle_verify.out" 2> "$GATE/bundle_verify.err" || fail_gate $?

EMPTY_CWD="$GATE/empty_cwd"
mkdir "$EMPTY_CWD"
(
  cd "$EMPTY_CWD"
  env -u PYTHONPATH "$PY" "$IMPORT_SMOKE" "$BUNDLE_ROOT" "$GATE/module_origins.json"
) > "$GATE/import_smoke.out" 2> "$GATE/import_smoke.err" || fail_gate $?

(
  cd "$EMPTY_CWD"
  env -u PYTHONPATH \
    TASK07_SOURCE_ROOT="$BUNDLE_ROOT/audit_sources/task07" \
    TASK13_SOURCE_ROOT="$BUNDLE_ROOT/audit_sources/task13" \
    TASK32_SOURCE_ROOT="$BUNDLE_ROOT/audit_sources/task32" \
    "$PY" "$ADAPTER" "$BUNDLE_ROOT" \
      "$GATE/historical_scaling_ledger.json" "$GATE/trainer_identity_ledger.json"
) > "$GATE/historical_audit.out" 2> "$GATE/historical_audit.err" || fail_gate $?

grep -q TASK34R_HISTORICAL_SCALING_AUDIT_PASS "$GATE/historical_audit.out" || fail_gate 73
grep -q TASK36_AUDIT_PATH_ADAPTER_PASS "$GATE/historical_audit.out" || fail_gate 74

for env_name in bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10; do
  if [ -e "$CAMPAIGN/preflight/$env_name" ]; then
    echo "fresh preflight root collision: $env_name" >&2
    fail_gate 90
  fi
done

printf '%s\n' 0 > "$GATE/rc"
echo LOCAL_GATE_PASS > "$GATE/status"
touch "$GATE/local_gate_pass.marker"
date -u +%Y-%m-%dT%H:%M:%SZ > "$GATE/ended_utc.txt"
echo TASK36_SINGLE_LOCAL_GATE_PASS
exit 0
