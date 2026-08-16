#!/usr/bin/env bash
set -euo pipefail

stack_root="${HOME}/rlstack5060"
campaign_root="${stack_root}/campaigns/procgen_actorj_dmlp1024_recovery_20260811_v1"
workspace="${stack_root}/workspaces/procgen"
status_command="${campaign_root}/campaign_status.sh"
stop_ledger="${campaign_root}/status/BOUNDED_STOP_LEDGER.tsv"

original_command="${SSH_ORIGINAL_COMMAND:-}"
if [[ -z "${original_command}" || "${original_command}" == status ]]; then
  exec "${status_command}"
fi

read -r -a argv <<< "${original_command}"
if [[ "${#argv[@]}" -ne 3 ]]; then
  echo "refused: unsupported command shape" >&2
  exit 64
fi
if [[ "${argv[0]}" != '$HOME/.local/bin/codex_run_stop' ]]; then
  echo "refused: only the bounded stop entry is allowed" >&2
  exit 64
fi

target_pid="${argv[1]}"
run_root="${argv[2]}"
[[ "${target_pid}" =~ ^[0-9]+$ ]] || {
  echo "refused: invalid PID" >&2
  exit 65
}
[[ "${run_root}" == "${workspace}" ]] || {
  echo "refused: run root is not the fixed Procgen workspace" >&2
  exit 65
}

owner="$(ps -o user= -p "${target_pid}" | awk '{$1=$1; print}')"
command_line="$(ps -o args= -p "${target_pid}")"
parent_pid="$(ps -o ppid= -p "${target_pid}" | awk '{$1=$1; print}')"
[[ "${owner}" == "$(id -un)" ]] || {
  echo "refused: PID is not owned by the campaign user" >&2
  exit 66
}
[[ "${command_line}" == *"docker run"* && "${command_line}" == *"${workspace}"* ]] || {
  echo "refused: PID is not a fixed-workspace Docker client" >&2
  exit 66
}

worker_match=0
for worker_status in "${campaign_root}"/status/worker_gpu*.status; do
  [[ -e "${worker_status}" ]] || continue
  if grep -Eq "worker_pid=${parent_pid}([^0-9]|$)" "${worker_status}"; then
    worker_match=1
    break
  fi
done
[[ "${worker_match}" -eq 1 ]] || {
  echo "refused: Docker client is not a child of a registered worker" >&2
  exit 66
}

container_name="$(sed -n 's/.*--name \([^ ]*\).*/\1/p' <<< "${command_line}")"
case "${container_name}" in
  pg5060-gpu0-j_dmlp1024_bigfish_s0|\
  pg5060-gpu0-j_dmlp1024_coinrun_s0|\
  pg5060-gpu1-j_dmlp1024_caveflyer_s0) ;;
  *) echo "refused: container is outside the fixed recovery queue" >&2; exit 66 ;;
esac
docker inspect "${container_name}" >/dev/null 2>&1 || {
  echo "refused: target container is not live" >&2
  exit 67
}

printf 'REQUESTED\tutc=%s\tpid=%s\tparent_worker=%s\tcontainer=%s\trun_root=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${target_pid}" "${parent_pid}" \
  "${container_name}" "${run_root}" >> "${stop_ledger}"
docker stop --time 30 "${container_name}" >/dev/null
printf 'STOPPED\tutc=%s\tpid=%s\tparent_worker=%s\tcontainer=%s\trun_root=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${target_pid}" "${parent_pid}" \
  "${container_name}" "${run_root}" >> "${stop_ledger}"
echo "stopped bounded container=${container_name} pid=${target_pid}"
