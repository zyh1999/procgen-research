#!/usr/bin/env bash
set -u

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 ORIGINAL_LAUNCHER_PID FORMAL_VARIANT CONFIG_BASENAME PHYSICAL_GPU" >&2
  exit 2
fi

original_launcher_pid="$1"
variant="$2"
config_basename="$3"
physical_gpu="$4"
project_root="$(cd "$(dirname "$0")/.." && pwd)"

while kill -0 "${original_launcher_pid}" 2>/dev/null; do
  sleep 15
done

exec bash "${project_root}/launchers/run_missing_formal_seeds.sh" \
  "${variant}" "${config_basename}" "${physical_gpu}"
