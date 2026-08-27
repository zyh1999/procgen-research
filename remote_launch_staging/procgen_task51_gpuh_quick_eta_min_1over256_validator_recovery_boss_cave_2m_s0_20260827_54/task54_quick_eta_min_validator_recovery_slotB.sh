#!/bin/bash
set -uo pipefail
umask 027

TASK_ID=PROCGEN-TASK51-GPUH-QUICK-ETA-MIN-1OVER256-VALIDATOR-RECOVERY-BOSS-CAVE-2M-S0-20260827-54
CAMPAIGN=/scratch/h99859yz/procgen_task51_gpuh_quick_eta_min_1over256_validator_recovery_boss_cave_2m_s0_20260827_54
CODE="$CAMPAIGN/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_full_shared_joint2b_fixedlr_dualtrust_eta_min_1over256_v1.py"
TRAINER_SHA=49d0a05d9bcaaa6d6aeb1b49751beee70403d3c0d61ed2de999b2ac7a5a3be9b
BETA1_SHA=ac3f389a1788ab09c6687feaccb9e246f462c4bf3b6dfc897c74d0dfa1956239
BETA4_SHA=c8b7d8e3a37aa496d10fa058ac3319c18830b7b04842c584df394b82afd56303

if [ "${SLURM_JOB_ID:-}" != "19487252" ]; then
  echo "wrong parent allocation" >&2
  exit 80
fi
if [ "$(hostname -s)" != "node822" ]; then
  echo "wrong allocated node" >&2
  exit 81
fi
mkdir -p "$CAMPAIGN/controller"
printf '%s\n' "$SLURM_JOB_ID" > "$CAMPAIGN/controller/parent_allocation_job"
printf '%s\n' "${SLURM_STEP_ID:-unknown}" > "$CAMPAIGN/controller/slurm_step_id"
hostname -f > "$CAMPAIGN/controller/node"
date -Is > "$CAMPAIGN/controller/launch_time"
nvidia-smi -L > "$CAMPAIGN/controller/gpu.txt" 2>&1
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$CAMPAIGN/controller/gpu_before.csv" 2>&1

sync_one() {
  local root="$1" env_name="$2"
  local logdir
  logdir=$(find "$root/runtime/logs" -mindepth 2 -maxdepth 2 -type d -name "${env_name}.*_0" -print 2>/dev/null | sort | tail -1)
  if [ -n "$logdir" ] && [ -d "$logdir" ]; then
    printf '%s\n' "$logdir" > "$root/source_log_dir"
    cp "$logdir/progress.csv" "$root/progress.csv" 2>/dev/null || true
    cp "$logdir/metric_trace.jsonl" "$root/metric_trace.jsonl" 2>/dev/null || true
    cp "$logdir/phase_switch.jsonl" "$root/phase_switch.jsonl" 2>/dev/null || true
    cp "$logdir/rollout_scheduler.jsonl" "$root/rollout_scheduler.jsonl" 2>/dev/null || true
    cp "$logdir/model.ckpt" "$root/model.ckpt" 2>/dev/null || true
  fi
}

run_cell() {
  local arm="$1" env_name="$2" method="$3" config_name="$4" config_sha="$5"
  local root="$CAMPAIGN/runs/$method/$env_name/seed0/2m_quick_eta_min_1over256_validator_recovery"
  local config="$CODE/configs/$config_name"
  local train_pid rc
  if [ -e "$root" ]; then
    echo "root collision: $root" >&2
    return 90
  fi
  mkdir -p "$root/runtime"
  ln -s "$CODE/configs" "$root/runtime/configs"
  printf '%s\n' "$TASK_ID" > "$root/task_id.txt"
  printf '%s\n' "$method" > "$root/method.txt"
  printf '%s\n' "$arm" > "$root/arm.txt"
  printf '%s\n' "$env_name" > "$root/env.txt"
  printf '%s\n' "$SLURM_JOB_ID" > "$root/parent_allocation_job"
  printf '%s\n' "${SLURM_STEP_ID:-unknown}" > "$root/slurm_step_id"
  hostname -f > "$root/hostname"
  echo STARTING > "$root/status"
  nvidia-smi -L > "$root/gpu.txt" 2>&1
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$root/gpu_at_start.csv" 2>&1
  sha256sum "$TRAINER" "$config" "$CAMPAIGN/frozen/task54_quick_eta_min_validator_recovery_slotB.sh" > "$root/frozen_identity.sha256"
  if [ "$(sha256sum "$TRAINER" | awk '{print $1}')" != "$TRAINER_SHA" ] || [ "$(sha256sum "$config" | awk '{print $1}')" != "$config_sha" ]; then
    echo HASH_FAIL > "$root/status"
    return 70
  fi
  if ! (cd "$CODE"; PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c 'import torch, procgen; import train_full_shared_joint2b_fixedlr_dualtrust_eta_min_1over256_v1; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 1; print(torch.__version__); print(torch.cuda.get_device_name(0))') > "$root/minimal_start_check.out" 2> "$root/minimal_start_check.err"; then
    echo STARTUP_FAIL > "$root/status"
    return 71
  fi
  local cmd=("$PY" -u "$TRAINER" --config "$config_name" --env_name "$env_name" --seed 0 --device 0)
  printf '%q ' "${cmd[@]}" > "$root/command.txt"; printf '\n' >> "$root/command.txt"
  touch "$root/scientific_started.marker"
  echo RUNNING > "$root/status"
  (cd "$root/runtime"; env PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 PYTHONHASHSEED=0 "${cmd[@]}" > "$root/stdout.log" 2> "$root/stderr.log") &
  train_pid=$!
  printf '%s\n' "$train_pid" > "$root/trainer_pid"
  while kill -0 "$train_pid" 2>/dev/null; do
    sync_one "$root" "$env_name"
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$root/gpu_latest.csv" 2>&1 || true
    sleep 30
  done
  wait "$train_pid"; rc=$?
  sync_one "$root" "$env_name"
  printf '%s\n' "$rc" > "$root/rc"
  if [ "$rc" -eq 0 ] && [ -s "$root/progress.csv" ] && [ -s "$root/metric_trace.jsonl" ] && [ -s "$root/phase_switch.jsonl" ] && [ -s "$root/rollout_scheduler.jsonl" ]; then
    echo PASS > "$root/status"
  else
    echo FAIL > "$root/status"
  fi
  return "$rc"
}

run_cell beta1 bossfight-easy-0-10 FULL_SHARED_JOINT2B_FIXEDLR_DUALTRUST_BETA1_V1 adv_resnet_full_shared_joint2b_fixedlr_dualtrust_beta1_v1_2m_quick_eta_min_1over256.yaml "$BETA1_SHA" & P1=$!
run_cell beta1 caveflyer-easy-0-10 FULL_SHARED_JOINT2B_FIXEDLR_DUALTRUST_BETA1_V1 adv_resnet_full_shared_joint2b_fixedlr_dualtrust_beta1_v1_2m_quick_eta_min_1over256.yaml "$BETA1_SHA" & P2=$!
run_cell beta4 bossfight-easy-0-10 FULL_SHARED_JOINT2B_FIXEDLR_DUALTRUST_BETA4_V1 adv_resnet_full_shared_joint2b_fixedlr_dualtrust_beta4_v1_2m_quick_eta_min_1over256.yaml "$BETA4_SHA" & P3=$!
run_cell beta4 caveflyer-easy-0-10 FULL_SHARED_JOINT2B_FIXEDLR_DUALTRUST_BETA4_V1 adv_resnet_full_shared_joint2b_fixedlr_dualtrust_beta4_v1_2m_quick_eta_min_1over256.yaml "$BETA4_SHA" & P4=$!
printf '%s\n' "$P1" "$P2" "$P3" "$P4" > "$CAMPAIGN/controller/worker_pids"
echo RUNNING > "$CAMPAIGN/controller/status"

wait "$P1"; R1=$?
wait "$P2"; R2=$?
wait "$P3"; R3=$?
wait "$P4"; R4=$?
printf 'beta1_boss=%s\nbeta1_cave=%s\nbeta4_boss=%s\nbeta4_cave=%s\n' "$R1" "$R2" "$R3" "$R4" > "$CAMPAIGN/controller/rcs"
echo TERMINAL > "$CAMPAIGN/controller/status"
if [ "$R1" -eq 0 ] && [ "$R2" -eq 0 ] && [ "$R3" -eq 0 ] && [ "$R4" -eq 0 ]; then exit 0; else exit 1; fi
