#!/usr/bin/env bash
set -u

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 FORMAL_VARIANT CONFIG_BASENAME PHYSICAL_GPU" >&2
  exit 2
fi

variant="$1"
config_basename="$2"
physical_gpu="$3"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
run_log_root="${project_root}/run_logs"
status_root="${project_root}/status"
python_bin="${PYTHON_BIN:-${project_root}/.venv/bin/python}"
mkdir -p "${run_log_root}" "${status_root}"
aggregate_return_code=0

for seed in 1 2; do
  status_path="${status_root}/${variant}_seed${seed}.status"
  if [[ -f "${status_path}" ]] && grep -Eq '^(COMPLETED|FAILED)' "${status_path}"; then
    continue
  fi

  while [[ -n "$(nvidia-smi -i "${physical_gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')" ]]; do
    sleep 15
  done

  stdout_path="${run_log_root}/${variant}_seed${seed}.stdout.log"
  command_path="${status_root}/${variant}_seed${seed}.command"
  printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=${physical_gpu} ${python_bin} -u train_phasic_ef_ggn.py --config ${config_basename} --device 0 --seed ${seed}" \
    > "${command_path}"
  printf 'RUNNING start_utc=%s physical_gpu=%s pid=%s continuation=1\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" "$$" \
    > "${status_path}"

  set +e
  (
    cd "${project_root}"
    CUDA_VISIBLE_DEVICES="${physical_gpu}" \
      "${python_bin}" -u train_phasic_ef_ggn.py \
        --config "${config_basename}" \
        --device 0 \
        --seed "${seed}"
  ) 2>&1 | tee "${stdout_path}"
  return_code="${PIPESTATUS[0]}"
  set -e

  if [[ "${return_code}" -ne 0 ]]; then
    printf 'FAILED end_utc=%s physical_gpu=%s returncode=%s continuation=1\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" "${return_code}" \
      > "${status_path}"
    aggregate_return_code="${return_code}"
  else
    printf 'COMPLETED end_utc=%s physical_gpu=%s returncode=0 continuation=1\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
      > "${status_path}"
  fi
done

exit "${aggregate_return_code}"
