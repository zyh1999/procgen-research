#!/usr/bin/env bash
set -u

if [[ -z "${PYTHON_BIN:-}" || -z "${RUN_ROOT:-}" || -z "${GPU_ID:-}" || -z "${TASK_START:-}" || -z "${TASK_END:-}" || -z "${WAIT_PID:-}" ]]; then
  echo "Required: PYTHON_BIN=... RUN_ROOT=... GPU_ID=... TASK_START=... TASK_END=... WAIT_PID=0 $0" >&2
  exit 2
fi

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
QUEUE_STATUS="$RUN_ROOT.queue_status"
QUEUE_LOG="$RUN_ROOT.queue_log"
MAX_BASELINE_MIB="${MAX_BASELINE_MIB:-6000}"
MAX_BASELINE_UTIL="${MAX_BASELINE_UTIL:-85}"

if [[ -e "$RUN_ROOT" || -e "$QUEUE_STATUS" ]]; then
  echo "Refusing to reuse queued or run root: $RUN_ROOT" >&2
  exit 3
fi

mkdir -p "$(dirname "$RUN_ROOT")"
{
  echo "queued_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "wait_pid=$WAIT_PID"
  echo "gpu_id=$GPU_ID"
  echo "task_start=$TASK_START"
  echo "task_end=$TASK_END"
  echo "max_baseline_mib=$MAX_BASELINE_MIB"
  echo "max_baseline_util=$MAX_BASELINE_UTIL"
} > "$QUEUE_LOG"

if [[ "$WAIT_PID" != "0" ]]; then
  echo QUEUED_WAIT_PPO > "$QUEUE_STATUS"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
fi

echo QUEUED_WAIT_GPU > "$QUEUE_STATUS"
while true; do
  read -r util mem < <(nvidia-smi -i "$GPU_ID" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  {
    echo "check_at=$(date --iso-8601=seconds) util=$util mem_mib=$mem"
  } >> "$QUEUE_LOG"
  if [[ "$mem" -lt "$MAX_BASELINE_MIB" && "$util" -lt "$MAX_BASELINE_UTIL" ]]; then
    break
  fi
  sleep 60
done

echo STARTING > "$QUEUE_STATUS"
exec env PYTHONPATH="${PYTHONPATH:-}" \
  PYTHON_BIN="$PYTHON_BIN" \
  RUN_ROOT="$RUN_ROOT" \
  GPU_ID="$GPU_ID" \
  TASK_START="$TASK_START" \
  TASK_END="$TASK_END" \
  bash "$BUNDLE_ROOT/launch_exactrat_8x5.sh"
