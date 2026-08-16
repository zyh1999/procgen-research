#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${project_root}/supervisor" "${project_root}/status"

for physical_gpu in 4 5 6 7; do
  active_pids="$(nvidia-smi -i "${physical_gpu}" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
  if [[ -n "${active_pids}" ]]; then
    echo "REFUSE: physical GPU${physical_gpu} has compute PID(s): ${active_pids}" >&2
    exit 3
  fi
done

launch_one() {
  local variant="$1"
  local config="$2"
  local physical_gpu="$3"
  local supervisor_log="${project_root}/supervisor/${variant}.log"
  nohup bash "${project_root}/launchers/run_variant_serial.sh" \
    "${variant}" "${config}" "${physical_gpu}" \
    > "${supervisor_log}" 2>&1 &
  local launcher_pid="$!"
  printf '%s\t%s\t%s\t%s\n' \
    "${variant}" "${physical_gpu}" "${launcher_pid}" "${supervisor_log}" \
    >> "${project_root}/status/PIDS.tsv"
}

: > "${project_root}/status/PIDS.tsv"
launch_one A phasic_A_bigfish.yaml 4
launch_one B phasic_B_bigfish.yaml 5
launch_one C phasic_C_bigfish.yaml 6
launch_one D phasic_D_bigfish.yaml 7

cat "${project_root}/status/PIDS.tsv"
