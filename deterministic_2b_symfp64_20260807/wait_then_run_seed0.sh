#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 GPU ENV_NAME WAIT_PID RUN_ROOT" >&2
  exit 2
fi

gpu="$1"
env_name="$2"
wait_pid="$3"
run_root="$4"
code_root="/root/procgen_goal1_20260806/code"
trainer="train_shared_rat_exact_deterministic_ggn_symfp64.py"
config="adv_resnet_shared_exact_deterministic_ggn_symfp64.yaml"
seed=0

mkdir -p "$run_root/logs" "$run_root/status"
printf 'WAITING gpu=%s env=%s predecessor_pid=%s time=%s\n' \
  "$gpu" "$env_name" "$wait_pid" "$(date -Is)" \
  > "$run_root/status/${env_name}.status"

while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 60
done

while nvidia-smi --id="$gpu" --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
  sleep 60
done

printf 'RUNNING gpu=%s env=%s seed=%s time=%s\n' \
  "$gpu" "$env_name" "$seed" "$(date -Is)" \
  > "$run_root/status/${env_name}.status"

cd "$code_root"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1
log="$run_root/logs/${env_name}.seed${seed}.stdout.log"
set +e
"/root/procgen_goal1_20260806/.venv/bin/python" -u "$trainer" \
  --config "$config" \
  --env_name "$env_name" \
  --seed "$seed" \
  --device 0 \
  >"$log" 2>&1
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  state="PASS"
else
  state="FAIL"
fi
printf '%s gpu=%s env=%s seed=%s rc=%s time=%s\n' \
  "$state" "$gpu" "$env_name" "$seed" "$rc" "$(date -Is)" \
  > "$run_root/status/${env_name}.status"
exit "$rc"
