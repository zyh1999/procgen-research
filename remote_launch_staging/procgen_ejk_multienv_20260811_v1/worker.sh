#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 PHYSICAL_GPU WORKER_ID" >&2
  exit 2
fi

physical_gpu="$1"
worker_id="$2"
campaign_root="/root/procgen_ejk_multienv_20260811_v1"
tasks_path="${campaign_root}/status/TASKS.tsv"
worker_status="${campaign_root}/status/workers/${worker_id}.status"
worker_claim="${campaign_root}/status/workers/${worker_id}.claim"
claim_lock="${campaign_root}/status/CLAIM.lock"

case "${physical_gpu}" in
  4|5|6|7) ;;
  *) echo "REFUSE: physical GPU must be one of 4,5,6,7" >&2; exit 3 ;;
esac

printf 'RUNNING start_utc=%s worker=%s physical_gpu=%s pid=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${worker_id}" "${physical_gpu}" "$$" \
  > "${worker_status}"

while true; do
  : > "${worker_claim}"
  exec 9>"${claim_lock}"
  flock -x 9
  while IFS=$'\t' read -r task_id method env_name seed config; do
    task_status="${campaign_root}/status/tasks/${task_id}.status"
    if [[ ! -e "${task_status}" ]]; then
      printf 'CLAIMED utc=%s worker=%s worker_pid=%s physical_gpu=%s task=%s method=%s env=%s seed=%s config=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${worker_id}" "$$" \
        "${physical_gpu}" "${task_id}" "${method}" "${env_name}" "${seed}" "${config}" \
        > "${task_status}"
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${task_id}" "${method}" "${env_name}" "${seed}" "${config}" \
        > "${worker_claim}"
      break
    fi
  done < "${tasks_path}"
  flock -u 9
  exec 9>&-

  if [[ ! -s "${worker_claim}" ]]; then
    break
  fi

  IFS=$'\t' read -r task_id method env_name seed config < "${worker_claim}"
  task_status="${campaign_root}/status/tasks/${task_id}.status"
  command_path="${campaign_root}/status/tasks/${task_id}.command"
  stdout_path="${campaign_root}/run_logs/${task_id}.stdout.log"
  stderr_path="${campaign_root}/run_logs/${task_id}.stderr.log"

  case "${method}" in
    E)
      method_root="${campaign_root}/E"
      python_bin="/root/procgen_matched_ppg_20260806_E_v2/.venv/bin/python"
      ;;
    J)
      method_root="${campaign_root}/J"
      python_bin="/root/procgen_ef_actor_ablation_20260806_v1/.venv/bin/python"
      ;;
    K)
      method_root="${campaign_root}/K"
      python_bin="/root/procgen_ef_adaptivekl_exactrat_20260806_v1/.venv/bin/python"
      ;;
    *)
      printf 'FAILED utc=%s reason=unknown_method method=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${method}" > "${task_status}"
      continue
      ;;
  esac

  while [[ -n "$(nvidia-smi -i "${physical_gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; do
    sleep 15
  done

  printf 'cd %q && CUDA_VISIBLE_DEVICES=%q %q -u train_phasic_ef_ggn.py --config %q --device 0 --seed %q\n' \
    "${method_root}" "${physical_gpu}" "${python_bin}" "${config}" "${seed}" \
    > "${command_path}"

  (
    cd "${method_root}"
    exec env CUDA_VISIBLE_DEVICES="${physical_gpu}" \
      "${python_bin}" -u train_phasic_ef_ggn.py \
        --config "${config}" --device 0 --seed "${seed}"
  ) > "${stdout_path}" 2> "${stderr_path}" &
  trainer_pid="$!"

  printf 'RUNNING start_utc=%s worker=%s worker_pid=%s trainer_pid=%s physical_gpu=%s task=%s method=%s env=%s seed=%s config=%s stdout=%s stderr=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${worker_id}" "$$" "${trainer_pid}" \
    "${physical_gpu}" "${task_id}" "${method}" "${env_name}" "${seed}" "${config}" \
    "${stdout_path}" "${stderr_path}" > "${task_status}"

  set +e
  wait "${trainer_pid}"
  return_code="$?"
  set -e

  if [[ "${return_code}" -eq 0 ]]; then
    printf 'COMPLETED end_utc=%s returncode=0 worker=%s physical_gpu=%s task=%s method=%s env=%s seed=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${worker_id}" "${physical_gpu}" \
      "${task_id}" "${method}" "${env_name}" "${seed}" > "${task_status}"
  else
    printf 'FAILED end_utc=%s returncode=%s worker=%s physical_gpu=%s task=%s method=%s env=%s seed=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${return_code}" "${worker_id}" \
      "${physical_gpu}" "${task_id}" "${method}" "${env_name}" "${seed}" > "${task_status}"
  fi
done

printf 'COMPLETED end_utc=%s worker=%s physical_gpu=%s pid=%s queue_empty=1\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${worker_id}" "${physical_gpu}" "$$" \
  > "${worker_status}"
