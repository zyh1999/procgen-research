#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
prior_root="/root/procgen_ef_actor_ablation_20260806_v1"
physical_gpu=6
variant="ACTOR_K"
config="actor_K_exactrat_adaptivekl_official_ppg_formal.yaml"
pipeline_status="${project_root}/status/K_PIPELINE.status"

mkdir -p "${project_root}/status" "${project_root}/supervisor"

printf 'WAITING_FOR_GPU6 start_utc=%s waits_for=%s/status/ACTOR_I.status\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${prior_root}" \
  > "${pipeline_status}"

while true; do
  prior_state=""
  if [[ -f "${prior_root}/status/ACTOR_I.status" ]]; then
    prior_state="$(head -n 1 "${prior_root}/status/ACTOR_I.status")"
  fi
  active_pids="$(nvidia-smi -i "${physical_gpu}" \
    --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | sed '/^[[:space:]]*$/d')"
  if [[ "${prior_state}" == COMPLETED* && -z "${active_pids}" ]]; then
    break
  fi
  sleep 10
done

printf 'RUNNING start_utc=%s physical_gpu=%s seeds=0,1,2 target_transitions=6000000\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
  > "${pipeline_status}"

set +e
bash "${project_root}/launchers/run_variant_serial_continue.sh" \
  "${variant}" "${config}" "${physical_gpu}"
return_code="$?"
set -e

if [[ "${return_code}" -eq 0 ]]; then
  printf 'COMPLETED end_utc=%s physical_gpu=%s seeds=0,1,2 failures=0\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
    > "${pipeline_status}"
else
  printf 'COMPLETED_WITH_FAILURES end_utc=%s physical_gpu=%s returncode=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" "${return_code}" \
    > "${pipeline_status}"
fi
exit "${return_code}"
