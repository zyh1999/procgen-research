#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yihe/procgen_emp_dmlp1024_ws92_20260810_v1
mkdir -p "$ROOT/supervisor" "$ROOT/locks" "$ROOT/formal" "$ROOT/status"

for GPU in 0 1; do
  setsid nohup bash "$ROOT/launchers/worker_ws92.sh" "$GPU" \
    > "$ROOT/supervisor/gpu${GPU}.log" 2>&1 < /dev/null &
  echo "$!" > "$ROOT/status/gpu${GPU}.worker.pid"
done
