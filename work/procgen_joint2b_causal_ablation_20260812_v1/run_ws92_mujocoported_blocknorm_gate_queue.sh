#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU_ID ENV_NAME [ENV_NAME ...]" >&2
  exit 2
fi

GPU_ID=$1
shift
ROOT=/home/yihe/procgen_joint2b_mujocoported_blocknorm_ws92_20260815_v1
RUNNER="$ROOT/launchers/run_ws92_mujocoported_blocknorm_gate_env.sh"
QUEUE_DIR="$ROOT/gate_1m_seed0_v1/queue_gpu${GPU_ID}"
mkdir -p "$QUEUE_DIR"
printf '%s\n' "$@" > "$QUEUE_DIR/environments"
echo RUNNING > "$QUEUE_DIR/status"

for ENV_NAME in "$@"; do
  echo "$ENV_NAME" > "$QUEUE_DIR/current_environment"
  if ! "$RUNNER" "$GPU_ID" "$ENV_NAME"; then
    echo "$ENV_NAME" > "$QUEUE_DIR/failed_environment"
    echo FAILED > "$QUEUE_DIR/status"
    exit 1
  fi
  echo "$ENV_NAME" >> "$QUEUE_DIR/completed_environments"
done

rm -f "$QUEUE_DIR/current_environment"
echo PASS > "$QUEUE_DIR/status"
