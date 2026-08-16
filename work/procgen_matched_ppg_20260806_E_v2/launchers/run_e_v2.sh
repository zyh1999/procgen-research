#!/usr/bin/env bash
set -u

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 PHYSICAL_GPU" >&2
  exit 2
fi

physical_gpu="$1"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-${project_root}/.venv/bin/python}"
status_root="${project_root}/status"
run_log_root="${project_root}/run_logs"
supervisor_root="${project_root}/supervisor"
mkdir -p "${status_root}" "${run_log_root}" "${supervisor_root}"

wait_for_gpu() {
  while [[ -n "$(nvidia-smi -i "${physical_gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; do
    sleep 15
  done
}

run_one() {
  local phase="$1"
  local config="$2"
  local seed="$3"
  local stdout_path="${run_log_root}/${phase}_seed${seed}.stdout.log"
  local status_path="${status_root}/${phase}_seed${seed}.status"
  wait_for_gpu
  printf 'RUNNING start_utc=%s physical_gpu=%s launcher_pid=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" "$$" \
    > "${status_path}"
  set +e
  (
    cd "${project_root}"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
      "${python_bin}" -u train_phasic_ef_ggn.py \
        --config "${config}" --device 0 --seed "${seed}"
  ) 2>&1 | tee "${stdout_path}"
  local return_code="${PIPESTATUS[0]}"
  set -e
  if [[ "${return_code}" -eq 0 ]]; then
    printf 'COMPLETED end_utc=%s physical_gpu=%s returncode=0\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
      > "${status_path}"
  else
    printf 'FAILED end_utc=%s physical_gpu=%s returncode=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" "${return_code}" \
      > "${status_path}"
  fi
  return "${return_code}"
}

printf 'SMOKE_RUNNING start_utc=%s physical_gpu=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
  > "${status_root}/E_V2_PIPELINE.status"
if ! run_one E_V2_SMOKE matched_ppg_E_v2_bigfish_smoke.yaml 991; then
  printf 'FAILED_SMOKE end_utc=%s physical_gpu=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
    > "${status_root}/E_V2_PIPELINE.status"
  exit 1
fi
if ! (cd "${project_root}" && "${python_bin}" validate_e_v2_smoke.py); then
  printf 'FAILED_SMOKE_GATE end_utc=%s physical_gpu=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
    > "${status_root}/E_V2_PIPELINE.status"
  exit 1
fi

printf 'RUNNING start_utc=%s target_transitions=6000000 seeds=0,1,2 physical_gpu=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
  > "${status_root}/E_V2_FORMAL.status"
printf 'FORMAL_RUNNING start_utc=%s physical_gpu=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
  > "${status_root}/E_V2_PIPELINE.status"

aggregate_return_code=0
for seed in 0 1 2; do
  if ! run_one E_V2_FORMAL matched_ppg_E_v2_bigfish_formal.yaml "${seed}"; then
    aggregate_return_code=1
  fi
done

if [[ "${aggregate_return_code}" -eq 0 ]]; then
  printf 'COMPLETED end_utc=%s seeds=0,1,2\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${status_root}/E_V2_FORMAL.status"
  printf 'COMPLETED end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${status_root}/E_V2_PIPELINE.status"
else
  printf 'FAILED_PARTIAL end_utc=%s inspect_seed_status=1\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${status_root}/E_V2_FORMAL.status"
  printf 'FAILED_PARTIAL end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${status_root}/E_V2_PIPELINE.status"
fi

exit "${aggregate_return_code}"
