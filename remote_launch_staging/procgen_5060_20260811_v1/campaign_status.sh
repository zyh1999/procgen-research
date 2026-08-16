#!/usr/bin/env bash
set -euo pipefail

campaign_root="${HOME}/rlstack5060/campaigns/procgen_actorj_dmlp1024_recovery_20260811_v1"
workspace="${HOME}/rlstack5060/workspaces/procgen"

echo "snapshot_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
docker ps --format 'container={{.Names}} status={{.Status}} image={{.Image}}'
for worker_status in "${campaign_root}"/status/worker_gpu*.status; do
  [[ -e "${worker_status}" ]] || continue
  worker_pid="$(sed -n 's/.*worker_pid=\([0-9][0-9]*\).*/\1/p' "${worker_status}")"
  [[ -n "${worker_pid}" ]] || continue
  ps --ppid "${worker_pid}" -o pid=,user=,args= 2>/dev/null \
    | sed 's/^[[:space:]]*/owned_child=/' || true
done
for path in "${campaign_root}"/status/*.status; do
  [[ -e "${path}" ]] || continue
  printf 'status_file=%s ' "$(basename "${path}")"
  tr '\n' ' ' < "${path}"
  echo
done
find "${workspace}/logs" -name progress.csv -type f -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -6 | while read -r _ path; do
      echo "progress_file=${path}"
      tail -2 "${path}"
    done

