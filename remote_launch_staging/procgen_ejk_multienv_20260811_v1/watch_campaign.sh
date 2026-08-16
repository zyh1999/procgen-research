#!/usr/bin/env bash
set -euo pipefail

campaign_root="/root/procgen_ejk_multienv_20260811_v1"
status_root="${campaign_root}/status"
heartbeat="${status_root}/HEARTBEAT.tsv"
total="$(wc -l < "${status_root}/TASKS.tsv")"

printf 'utc\ttotal\tcompleted\tfailed\trunning\tclaimed\tworkers_alive\terror_hits\n' > "${heartbeat}"

while true; do
  completed="$(grep -l '^COMPLETED' "${status_root}"/tasks/*.status 2>/dev/null | wc -l || true)"
  failed="$(grep -l '^FAILED' "${status_root}"/tasks/*.status 2>/dev/null | wc -l || true)"
  running="$(grep -l '^RUNNING' "${status_root}"/tasks/*.status 2>/dev/null | wc -l || true)"
  claimed="$(grep -l '^CLAIMED' "${status_root}"/tasks/*.status 2>/dev/null | wc -l || true)"
  workers_alive=0
  if [[ -f "${status_root}/PIDS.tsv" ]]; then
    while IFS=$'\t' read -r kind gpu pid path; do
      if [[ "${kind}" == WORKER_* ]] && kill -0 "${pid}" 2>/dev/null; then
        workers_alive=$((workers_alive + 1))
      fi
    done < "${status_root}/PIDS.tsv"
  fi
  error_hits="$(grep -Eih 'Traceback|RuntimeError|out of memory|CUDA error|nan|infinity|nonfinite' "${campaign_root}"/run_logs/*.stderr.log "${campaign_root}"/run_logs/*.stdout.log 2>/dev/null | wc -l || true)"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${now}" "${total}" "${completed}" "${failed}" "${running}" "${claimed}" \
    "${workers_alive}" "${error_hits}" >> "${heartbeat}"

  if [[ $((completed + failed)) -eq "${total}" ]]; then
    if [[ "${failed}" -eq 0 ]]; then
      printf 'COMPLETED end_utc=%s tasks=%s failures=0\n' "${now}" "${total}" \
        > "${status_root}/CAMPAIGN.status"
    else
      printf 'COMPLETED_WITH_FAILURES end_utc=%s tasks=%s failures=%s\n' \
        "${now}" "${total}" "${failed}" > "${status_root}/CAMPAIGN.status"
    fi
    exit 0
  fi

  if [[ "${workers_alive}" -eq 0 && "${running}" -eq 0 && "${claimed}" -eq 0 ]]; then
    printf 'FAILED_OR_STALLED utc=%s tasks=%s completed=%s failed=%s\n' \
      "${now}" "${total}" "${completed}" "${failed}" \
      > "${status_root}/CAMPAIGN.status"
    exit 1
  fi

  printf 'RUNNING utc=%s tasks=%s completed=%s failed=%s running=%s claimed=%s workers_alive=%s physical_gpus=4,5,6,7\n' \
    "${now}" "${total}" "${completed}" "${failed}" "${running}" "${claimed}" \
    "${workers_alive}" > "${status_root}/CAMPAIGN.status"
  sleep 30
done
