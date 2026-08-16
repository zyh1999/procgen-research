#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID ENV..." >&2
  exit 2
fi
GPU_ID=$1
shift
ROOT=/home/yihe/procgen_joint2b_mujocoported_blocknorm_ws92_20260815_v1
for ENV_NAME in "$@"; do
  "$ROOT/launchers/run_ws92_mujocoported_rownorm_d05_1m_allocator_env.sh" \
    "$GPU_ID" "$ENV_NAME"
done
