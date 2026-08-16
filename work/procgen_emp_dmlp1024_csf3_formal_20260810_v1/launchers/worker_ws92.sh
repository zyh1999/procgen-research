#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yihe/procgen_emp_dmlp1024_ws92_20260810_v1
GPU=${1:?GPU is required}
mkdir -p "$ROOT/locks" "$ROOT/supervisor"
exec 9>"$ROOT/locks/gpu${GPU}.lock"
flock -n 9 || { echo "GPU worker $GPU already exists" >&2; exit 7; }

case "$GPU" in
  0) ENVS=(bigfish-easy-0-10 caveflyer-easy-0-10) ;;
  1) ENVS=(bossfight-easy-0-10 coinrun-easy-0-10) ;;
  *) echo "Only GPU 0 or 1 is valid" >&2; exit 2 ;;
esac

for SEED in 0 1 2; do
  for ENV_NAME in "${ENVS[@]}"; do
    bash "$ROOT/launchers/run_one_ws92.sh" "$ENV_NAME" "$SEED" "$GPU"
  done
done
