#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
status_path="${project_root}/status/ACTOR_HANDOFF.status"
mkdir -p "${project_root}/status" "${project_root}/supervisor"

printf 'WAITING_FOR_GPU start_utc=%s physical_gpus=4,5,6,7\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_path}"

while true; do
  busy=0
  for physical_gpu in 4 5 6 7; do
    active_pids="$(nvidia-smi -i "${physical_gpu}" \
      --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | sed '/^[[:space:]]*$/d')"
    if [[ -n "${active_pids}" ]]; then
      busy=1
    fi
  done
  if [[ "${busy}" -eq 0 ]]; then
    break
  fi
  sleep 10
done

printf 'LAUNCHING start_utc=%s physical_gpus=4,5,6,7\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_path}"
bash "${project_root}/launchers/launch_actor_formal_direct.sh"
printf 'LAUNCHED end_utc=%s physical_gpus=4,5,6,7\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_path}"
