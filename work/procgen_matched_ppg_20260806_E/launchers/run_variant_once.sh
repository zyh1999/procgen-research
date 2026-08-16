#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 VARIANT CONFIG_BASENAME PHYSICAL_GPU SEED" >&2
  exit 2
fi

variant="$1"
config_basename="$2"
physical_gpu="$3"
seed="$4"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
run_log_root="${project_root}/run_logs"
status_root="${project_root}/status"
python_bin="${PYTHON_BIN:-${project_root}/.venv/bin/python}"
mkdir -p "${run_log_root}" "${status_root}"
stdout_path="${run_log_root}/${variant}_seed${seed}.stdout.log"
status_path="${status_root}/${variant}_seed${seed}.status"
printf 'RUNNING start_utc=%s physical_gpu=%s pid=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" "$$" \
  > "${status_path}"
set +e
(
  cd "${project_root}"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" \
    "${python_bin}" -u train_phasic_ef_ggn.py \
      --config "${config_basename}" --device 0 --seed "${seed}"
) 2>&1 | tee "${stdout_path}"
return_code="${PIPESTATUS[0]}"
set -e
if [[ "${return_code}" -eq 0 ]]; then
  printf 'COMPLETED end_utc=%s physical_gpu=%s returncode=0\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" > "${status_path}"
else
  printf 'FAILED end_utc=%s physical_gpu=%s returncode=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" "${return_code}" \
    > "${status_path}"
fi
exit "${return_code}"
