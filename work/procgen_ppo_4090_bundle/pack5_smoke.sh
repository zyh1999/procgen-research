#!/usr/bin/env bash
set -u

if [[ -z "${PYTHON_BIN:-}" || -z "${GPU_ID:-}" || -z "${SMOKE_ROOT:-}" ]]; then
  echo "Required: PYTHON_BIN=/path/to/python GPU_ID=0 SMOKE_ROOT=/path/to/new/smoke $0" >&2
  exit 2
fi

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE_ROOT="$BUNDLE_ROOT/source/trust-region-main"

if [[ -e "$SMOKE_ROOT" ]]; then
  echo "Refusing to reuse existing SMOKE_ROOT: $SMOKE_ROOT" >&2
  exit 3
fi
mkdir -p "$SMOKE_ROOT"

(
  while true; do
    date +%s
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    sleep 1
  done
) > "$SMOKE_ROOT/gpu.csv" 2>&1 &
MON_PID=$!

PIDS=()
for seed in 990 991 992 993 994; do
  (
    cd "$CODE_ROOT"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u train_shared.py \
      --config ppo_resnet_shared_smoke.yaml \
      --env_name bigfish-easy-0-10 \
      --seed "$seed" \
      --device 0
  ) > "$SMOKE_ROOT/seed${seed}.stdout" 2> "$SMOKE_ROOT/seed${seed}.stderr" &
  PIDS+=("$!")
done

OVERALL=0
for index in 0 1 2 3 4; do
  seed=$((990 + index))
  if wait "${PIDS[$index]}"; then
    rc=0
  else
    rc=$?
    OVERALL=1
  fi
  echo "$rc" > "$SMOKE_ROOT/seed${seed}.rc"
done

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
echo "$OVERALL" > "$SMOKE_ROOT/returncode"
if [[ "$OVERALL" -eq 0 ]]; then
  echo PASS > "$SMOKE_ROOT/status"
else
  echo FAIL > "$SMOKE_ROOT/status"
fi
exit "$OVERALL"
