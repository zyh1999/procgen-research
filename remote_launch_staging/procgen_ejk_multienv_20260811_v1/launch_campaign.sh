#!/usr/bin/env bash
set -euo pipefail

campaign_root="/root/procgen_ejk_multienv_20260811_v1"
status_root="${campaign_root}/status"
launcher_root="${campaign_root}/launchers"
supervisor_root="${campaign_root}/supervisor"

if grep -q '^RUNNING' "${status_root}/CAMPAIGN.status" 2>/dev/null; then
  echo "REFUSE: campaign is already marked RUNNING" >&2
  exit 3
fi

: > "${status_root}/INITIAL_GPU_OCCUPANCY.tsv"
for physical_gpu in 4 5 6 7; do
  active_pids="$(nvidia-smi -i "${physical_gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
  if [[ -n "${active_pids}" ]]; then
    printf '%s\tWAITING\t%s\n' "${physical_gpu}" "${active_pids//$'\n'/,}" \
      >> "${status_root}/INITIAL_GPU_OCCUPANCY.tsv"
  else
    printf '%s\tIDLE\tnone\n' "${physical_gpu}" \
      >> "${status_root}/INITIAL_GPU_OCCUPANCY.tsv"
  fi
done

: > "${status_root}/PIDS.tsv"
for physical_gpu in 4 5 6 7; do
  worker_id="WORKER_GPU${physical_gpu}"
  worker_log="${supervisor_root}/${worker_id}.log"
  nohup setsid bash "${launcher_root}/worker.sh" "${physical_gpu}" "${worker_id}" \
    > "${worker_log}" 2>&1 < /dev/null &
  printf '%s\t%s\t%s\t%s\n' \
    "${worker_id}" "${physical_gpu}" "$!" "${worker_log}" \
    >> "${status_root}/PIDS.tsv"
done

watcher_log="${supervisor_root}/WATCHER.log"
nohup setsid bash "${launcher_root}/watch_campaign.sh" \
  > "${watcher_log}" 2>&1 < /dev/null &
printf 'WATCHER\tNA\t%s\t%s\n' "$!" "${watcher_log}" >> "${status_root}/PIDS.tsv"

printf 'RUNNING_OR_WAITING start_utc=%s tasks=27 methods=E_v2,J,K environments=coinrun,bossfight,caveflyer seeds=0,1,2 transitions_per_seed=6000000 physical_gpus=4,5,6,7 no_training_smoke=1\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${status_root}/CAMPAIGN.status"

cat "${status_root}/PIDS.tsv"
