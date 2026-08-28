#!/bin/bash
#SBATCH --job-name=pg63-direction-gate
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/nobackup/projects/bdman37/yihe/procgen_deterministic_joint2b_actor_critic_direction_telemetry_2m_s0_20260828_63/scheduler/%x_%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/procgen_deterministic_joint2b_actor_critic_direction_telemetry_2m_s0_20260828_63/scheduler/%x_%j.err

set -uo pipefail
CAMPAIGN=/nobackup/projects/bdman37/yihe/procgen_deterministic_joint2b_actor_critic_direction_telemetry_2m_s0_20260828_63
CODE="$CAMPAIGN/bundle/code"
PY=/nobackup/projects/bdman37/yihe/ppc64le/envs/procgen_author/bin/python
TRAINER="$CODE/train_shared_paper_matched_deterministic_ggn_direction_telemetry_only_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_paper_matched_deterministic_ggn_direction_telemetry_gate.yaml"
ROOT="$CAMPAIGN/gate/production"
[ ! -e "$ROOT" ] || exit 90
mkdir -p "$ROOT/runtime"
ln -s "$CODE/configs" "$ROOT/runtime/configs"
echo PRECHECK_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/task63_gate_bede.sbatch" "$CAMPAIGN/frozen/aggregate_task63.py" > "$ROOT/frozen.sha256"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
export TASK63_GATE_MODE=1
(cd "$ROOT/runtime"; "$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name bigfish-easy-0-10 --seed 0 --device 0) > "$ROOT/stdout.log" 2> "$ROOT/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$ROOT/rc"
LOGDIR=$(find "$ROOT/runtime/logs" -mindepth 2 -maxdepth 2 -type d -name 'bigfish-easy-0-10.*_0' -print 2>/dev/null | sort | tail -1)
if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
  printf '%s\n' "$LOGDIR" > "$ROOT/source_log_dir"
  cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
  cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
fi
if [ "$rc" -eq 0 ] && "$PY" "$CAMPAIGN/frozen/aggregate_task63.py" "$ROOT/metric_trace.jsonl" --gate > "$ROOT/gate_validation.json"; then
  echo PRECHECK_PASS > "$ROOT/status"
  touch "$ROOT/precheck_pass.marker"
  exit 0
fi
echo PRECHECK_FAIL > "$ROOT/status"
exit 1
