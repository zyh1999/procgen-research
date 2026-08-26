#!/bin/bash
#SBATCH --job-name=pg50-bede-gate
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_rat_rollout_sched_6m_s0_20260826_50/scheduler/%x_%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_rat_rollout_sched_6m_s0_20260826_50/scheduler/%x_%j.err

set -uo pipefail
CAMPAIGN=/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_ppo500k_rat_rollout_sched_6m_s0_20260826_50
ROOT="$CAMPAIGN/gate/production"
CODE="$CAMPAIGN/bundle/code"
PY=/nobackup/projects/bdman37/yihe/ppc64le/envs/procgen_author/bin/python
TRAINER="$CODE/train_full_shared_joint2b_ppo500k_rat_rollout_sched_v1.py"
CONFIG="$CODE/configs/adv_resnet_full_shared_joint2b_ppo500k_rat_rollout_sched_v1_6m.yaml"
if [ -e "$ROOT" ]; then exit 90; fi
mkdir -p "$ROOT" "$CODE/logs"
echo PRECHECK_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/task50_gate_bede.sbatch" > "$ROOT/frozen.sha256"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 PYTHONHASHSEED=0
(cd "$CODE"; "$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name bigfish-easy-0-10 --seed 0 --device 0 --gate_mode) > "$ROOT/stdout.log" 2> "$ROOT/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$ROOT/rc"
LOGDIR=$(find "$CODE/logs" -mindepth 2 -maxdepth 2 -type d -name 'bigfish-easy-0-10.*_0' -print 2>/dev/null | sort | tail -1)
if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
  printf '%s\n' "$LOGDIR" > "$ROOT/source_log_dir"
  cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
  cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
  cp "$LOGDIR/phase_switch.jsonl" "$ROOT/phase_switch.jsonl" 2>/dev/null || true
  cp "$LOGDIR/rollout_scheduler.jsonl" "$ROOT/rollout_scheduler.jsonl" 2>/dev/null || true
fi
if [ "$rc" -eq 0 ] \
  && grep -q '"training_phase": "ppo_warmup"' "$ROOT/metric_trace.jsonl" \
  && grep -q '"training_phase": "joint2b"' "$ROOT/metric_trace.jsonl" \
  && grep -q '"switch_count": 1' "$ROOT/phase_switch.jsonl" \
  && grep -q '"joint_cholesky_info_max": 0' "$ROOT/metric_trace.jsonl" \
  && grep -q '"joint_finite_scan_pass": 1.0' "$ROOT/metric_trace.jsonl" \
  && grep -q '"lr_before": 0.004' "$ROOT/rollout_scheduler.jsonl" \
  && grep -q '"minibatch_lr_change_count": 0' "$ROOT/rollout_scheduler.jsonl" \
  && grep -q '"scheduler_update_count_this_rollout": 1' "$ROOT/rollout_scheduler.jsonl"; then
  echo PRECHECK_PASS > "$ROOT/status"
  touch "$ROOT/precheck_pass.marker"
  exit 0
fi
echo PRECHECK_FAIL > "$ROOT/status"
if [ "$rc" -eq 0 ]; then exit 1; fi
exit "$rc"
