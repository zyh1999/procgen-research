#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "$0")/.." && pwd)"
pids_path="${project_root}/status/PIDS_ACTOR_FORMAL.tsv"
status_path="${project_root}/status/ACTOR_FORMAL_SWEEP.status"

while true; do
  live=0
  while IFS=$'\t' read -r variant _gpu pid _log; do
    if [[ "${variant}" == "WATCHER" ]]; then
      continue
    fi
    if kill -0 "${pid}" 2>/dev/null; then
      live=1
    fi
  done < "${pids_path}"
  if [[ "${live}" -eq 0 ]]; then
    break
  fi
  sleep 30
done

failures=0
for variant in G H I J; do
  aggregate="${project_root}/status/ACTOR_${variant}.status"
  if [[ ! -f "${aggregate}" ]] || \
     ! grep -q '^COMPLETED ' "${aggregate}"; then
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -eq 0 ]]; then
  printf 'COMPLETED end_utc=%s failures=0\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_path}"
else
  printf 'COMPLETED_WITH_FAILURES end_utc=%s variant_failures=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${failures}" > "${status_path}"
fi
