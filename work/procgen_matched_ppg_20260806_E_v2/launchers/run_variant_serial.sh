#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 VARIANT CONFIG_BASENAME PHYSICAL_GPU" >&2
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
if [[ ! -x "${python_bin}" ]]; then
  echo "Python runtime is not executable: ${python_bin}" >&2
  exit 4
fi

for seed in 0 1 2; do
  stdout_path="${run_log_root}/${variant}_seed${seed}.stdout.log"
  status_path="${status_root}/${variant}_seed${seed}.status"
  command_path="${status_root}/${variant}_seed${seed}.command"
  printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=${physical_gpu} ${python_bin} -u train_phasic_ef_ggn.py --config ${config_basename} --device 0 --seed ${seed}" \
    > "${command_path}"
  printf 'RUNNING start_utc=%s physical_gpu=%s pid=%s\n' \
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
    printf 'FAILED end_utc=%s physical_gpu=%s returncode=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" "${return_code}" \
      > "${status_path}"
    exit "${return_code}"
  fi
  printf 'COMPLETED end_utc=%s physical_gpu=%s returncode=0\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
    > "${status_path}"
done
