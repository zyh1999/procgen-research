#!/bin/bash
#SBATCH --job-name=pg61-entgrad-gate
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/nobackup/projects/bdman37/yihe/procgen_task55_quick_nowarmup_postsolve_entropy_grad001_beta1_boss_cave_2m_s0_20260827_61/scheduler/%x_%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/procgen_task55_quick_nowarmup_postsolve_entropy_grad001_beta1_boss_cave_2m_s0_20260827_61/scheduler/%x_%j.err

set -uo pipefail
CAMPAIGN=/nobackup/projects/bdman37/yihe/procgen_task55_quick_nowarmup_postsolve_entropy_grad001_beta1_boss_cave_2m_s0_20260827_61
CODE="$CAMPAIGN/bundle/code"
PY=/nobackup/projects/bdman37/yihe/ppc64le/envs/procgen_author/bin/python
TRAINER="$CODE/train_full_shared_joint2b_nowarmup_postsolve_entropy_grad001_beta1_v1.py"
CONFIG="$CODE/configs/adv_resnet_full_shared_joint2b_nowarmup_postsolve_entropy_grad001_beta1_v1_2m.yaml"
ROOT="$CAMPAIGN/gate/production"
[ ! -e "$ROOT" ] || exit 90
mkdir -p "$ROOT/runtime"
ln -s "$CODE/configs" "$ROOT/runtime/configs"
echo PRECHECK_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/task61_gate_bede.sbatch" > "$ROOT/frozen.sha256"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 PYTHONHASHSEED=0
(cd "$ROOT/runtime"; "$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name bigfish-easy-0-10 --seed 0 --device 0 --gate_mode) > "$ROOT/stdout.log" 2> "$ROOT/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$ROOT/rc"
LOGDIR=$(find "$ROOT/runtime/logs" -mindepth 2 -maxdepth 2 -type d -name 'bigfish-easy-0-10.*_0' -print 2>/dev/null | sort | tail -1)
if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
  printf '%s\n' "$LOGDIR" > "$ROOT/source_log_dir"
  cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
  cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
  cp "$LOGDIR/rollout_scheduler.jsonl" "$ROOT/rollout_scheduler.jsonl" 2>/dev/null || true
fi
if [ "$rc" -eq 0 ] \
  && grep -q '"joint_system_rows": 1024' "$ROOT/metric_trace.jsonl" \
  && grep -q '"joint_parameter_columns": 938976' "$ROOT/metric_trace.jsonl" \
  && grep -q '"joint_cholesky_info_max": 0' "$ROOT/metric_trace.jsonl" \
  && grep -q '"joint_finite_scan_pass": 1.0' "$ROOT/metric_trace.jsonl" \
  && grep -q '"postsolve_entropy_coef": 0.01' "$ROOT/metric_trace.jsonl" \
  && grep -q '"postsolve_entropy_value_l2": 0.0' "$ROOT/metric_trace.jsonl" \
  && grep -q '"postsolve_entropy_absent_from_system": 1.0' "$ROOT/metric_trace.jsonl" \
  && grep -q '"postsolve_entropy_absent_from_rhs": 1.0' "$ROOT/metric_trace.jsonl" \
  && grep -q '"postsolve_entropy_absent_from_history": 1.0' "$ROOT/metric_trace.jsonl" \
  && grep -q '"postsolve_entropy_finite_scan_pass": 1.0' "$ROOT/metric_trace.jsonl" \
  && grep -q '"fixed_parameter_lr": 0.004' "$ROOT/rollout_scheduler.jsonl"; then
  echo PRECHECK_PASS > "$ROOT/status"
  touch "$ROOT/precheck_pass.marker"
  exit 0
fi
echo PRECHECK_FAIL > "$ROOT/status"
exit 1
