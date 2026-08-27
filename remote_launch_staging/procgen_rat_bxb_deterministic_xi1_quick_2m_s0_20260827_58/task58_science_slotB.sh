#!/bin/bash
set -uo pipefail
umask 027

TASK_ID=PROCGEN-RAT-BXB-DETERMINISTIC-XI1-QUICK-2M-S0-20260827-58
METHOD=SHARED_RAT_BXB_DETERMINISTIC_XI1_V1
CAMPAIGN=/scratch/h99859yz/procgen_rat_bxb_deterministic_xi1_quick_2m_s0_20260827_58
CODE="$CAMPAIGN/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_rat_weightedcritic_reference_publicmomentum.py"
CONFIG_NAME=adv_resnet_shared_rat_bxb_deterministic_xi1_v1_2m.yaml
CONFIG="$CODE/configs/$CONFIG_NAME"
TRAINER_SHA=77e005311daa52e9e09352042adedff13c0e587a60a13c92be25a592b4f355cb
CONFIG_SHA=fb4e3787e9e52212bb076744753e57e941a3bf37c82738844083ac6297f208fe

if [ "${SLURM_JOB_ID:-}" != "19487252" ]; then
  echo "wrong parent allocation" >&2
  exit 80
fi
if [ "$(hostname -s)" != "node822" ]; then
  echo "wrong allocated node" >&2
  exit 81
fi
if [ "$(cat "$CAMPAIGN/gate/status" 2>/dev/null)" != "PRECHECK_PASS" ]; then
  echo "gate is not PRECHECK_PASS" >&2
  exit 82
fi
if [ "$(sha256sum "$TRAINER" | awk '{print $1}')" != "$TRAINER_SHA" ] || [ "$(sha256sum "$CONFIG" | awk '{print $1}')" != "$CONFIG_SHA" ]; then
  echo "frozen identity mismatch" >&2
  exit 83
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
    if [ -f "$logdir/model.ckpt" ]; then
      stat "$logdir/model.ckpt" > "$root/checkpoint_stat_only.txt" 2>/dev/null || true
    fi
  fi
}

run_cell() {
  local env_name="$1"
  local root="$CAMPAIGN/runs/$METHOD/$env_name/seed0/2m_quick"
  local train_pid rc
  if [ -e "$root" ]; then
    echo "root collision: $root" >&2
    return 90
  fi
  mkdir -p "$root/runtime"
  ln -s "$CODE/configs" "$root/runtime/configs"
  printf '%s\n' "$TASK_ID" > "$root/task_id.txt"
  printf '%s\n' "$METHOD" > "$root/method.txt"
  printf '%s\n' "$env_name" > "$root/env.txt"
  printf '%s\n' "$SLURM_JOB_ID" > "$root/parent_allocation_job"
  printf '%s\n' "${SLURM_STEP_ID:-unknown}" > "$root/slurm_step_id"
  hostname -f > "$root/hostname"
  echo STARTING > "$root/status"
  nvidia-smi -L > "$root/gpu.txt" 2>&1
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$root/gpu_at_start.csv" 2>&1
  sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/task58_science_slotB.sh" > "$root/frozen_identity.sha256"
  local cmd=("$PY" -u "$TRAINER" --config "$CONFIG_NAME" --env_name "$env_name" --seed 0 --device 0 --total_timesteps 2000000)
  printf '%q ' "${cmd[@]}" > "$root/command.txt"
  printf '\n' >> "$root/command.txt"
  touch "$root/scientific_started.marker"
  echo RUNNING > "$root/status"
  (cd "$root/runtime"; env PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" PROCGEN_METRIC_TRACE_PATH="$root/metric_trace.jsonl" OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 PYTHONHASHSEED=0 "${cmd[@]}" > "$root/stdout.log" 2> "$root/stderr.log") &
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
  if [ "$rc" -eq 0 ] && [ -s "$root/progress.csv" ] && [ -s "$root/metric_trace.jsonl" ]; then
    echo PASS > "$root/status"
  else
    echo FAIL > "$root/status"
  fi
  return "$rc"
}

run_cell bigfish-easy-0-10 & P1=$!
run_cell bossfight-easy-0-10 & P2=$!
run_cell caveflyer-easy-0-10 & P3=$!
run_cell coinrun-easy-0-10 & P4=$!
printf '%s\n' "$P1" "$P2" "$P3" "$P4" > "$CAMPAIGN/controller/worker_pids"
echo RUNNING > "$CAMPAIGN/controller/status"

wait "$P1"; R1=$?
wait "$P2"; R2=$?
wait "$P3"; R3=$?
wait "$P4"; R4=$?
printf 'bigfish=%s\nbossfight=%s\ncaveflyer=%s\ncoinrun=%s\n' "$R1" "$R2" "$R3" "$R4" > "$CAMPAIGN/controller/rcs"
echo TERMINAL > "$CAMPAIGN/controller/status"
if [ "$R1" -eq 0 ] && [ "$R2" -eq 0 ] && [ "$R3" -eq 0 ] && [ "$R4" -eq 0 ]; then exit 0; else exit 1; fi

