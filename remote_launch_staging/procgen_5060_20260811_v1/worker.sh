#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 GPU TASK_ID [TASK_ID ...]" >&2
  exit 2
fi

gpu="$1"
shift
case "${gpu}" in
  0|1) ;;
  *) echo "GPU must be 0 or 1" >&2; exit 2 ;;
esac

campaign_root="${HOME}/rlstack5060/campaigns/procgen_actorj_dmlp1024_recovery_20260811_v1"
workspace="${HOME}/rlstack5060/workspaces/procgen"
status_root="${campaign_root}/status"
log_root="${campaign_root}/run_logs"
worker_status="${status_root}/worker_gpu${gpu}.status"
mkdir -p "${status_root}" "${log_root}"

task_config() {
  case "$1" in
    J_DMLP1024_BIGFISH_S0) echo actor_j_dmlp1024_bigfish.yaml ;;
    J_DMLP1024_CAVEFLYER_S0) echo actor_j_dmlp1024_caveflyer.yaml ;;
    J_DMLP1024_COINRUN_S0) echo actor_j_dmlp1024_coinrun.yaml ;;
    *) return 1 ;;
  esac
}

write_worker_status() {
  local state="$1"
  local task_id="${2:-none}"
  printf '%s\tutc=%s\tgpu=%s\tworker_pid=%s\ttask=%s\n' \
    "${state}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${gpu}" "$$" "${task_id}" \
    > "${worker_status}.tmp"
  mv "${worker_status}.tmp" "${worker_status}"
}

write_worker_status RUNNING none

for task_id in "$@"; do
  config="$(task_config "${task_id}")" || {
    echo "unknown task: ${task_id}" >&2
    exit 3
  }
  task_status="${status_root}/${task_id}.status"
  stdout_path="${log_root}/${task_id}.stdout.log"
  stderr_path="${log_root}/${task_id}.stderr.log"
  command_path="${status_root}/${task_id}.command"

  if [[ -e "${task_status}" ]]; then
    echo "refusing existing task status: ${task_status}" >&2
    exit 4
  fi

  printf 'docker_gpu=%q image=%q python=%q config=%q seed=0\n' \
    "${gpu}" "rlstack5060/procgen:cu128" "train_phasic_ef_ggn.py" "${config}" \
    > "${command_path}"
  printf 'STARTING\tutc=%s\tgpu=%s\ttask=%s\tconfig=%s\tseed=0\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${gpu}" "${task_id}" "${config}" \
    > "${task_status}"
  write_worker_status RUNNING "${task_id}"

  container_name="pg5060-gpu${gpu}-${task_id,,}"
  set +e
  docker run --rm --name "${container_name}" \
    --gpus "device=${gpu}" \
    --ipc=host \
    --shm-size=12g \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e OMP_NUM_THREADS=4 \
    -e MKL_NUM_THREADS=4 \
    -v "${workspace}:/workspace/procgen" \
    -w /workspace/procgen \
    rlstack5060/procgen:cu128 \
    python -u train_phasic_ef_ggn.py \
      --config "${config}" --device 0 --seed 0 \
      > "${stdout_path}" 2> "${stderr_path}"
  return_code="$?"
  set -e

  if [[ "${return_code}" -eq 0 ]]; then
    state=COMPLETED
  else
    state=FAILED
  fi
  printf '%s\tutc=%s\treturncode=%s\tgpu=%s\ttask=%s\tconfig=%s\tseed=0\tstdout=%s\tstderr=%s\n' \
    "${state}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${return_code}" \
    "${gpu}" "${task_id}" "${config}" "${stdout_path}" "${stderr_path}" \
    > "${task_status}"
done

write_worker_status COMPLETED queue_empty

