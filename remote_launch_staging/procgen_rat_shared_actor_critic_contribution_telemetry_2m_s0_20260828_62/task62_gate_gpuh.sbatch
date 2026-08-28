#!/bin/bash --login
#SBATCH --job-name=pg62-rat-telemetry-gate
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_rat_shared_actor_critic_contribution_telemetry_2m_s0_20260828_62/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_rat_shared_actor_critic_contribution_telemetry_2m_s0_20260828_62/scheduler/%x_%j.err

set -uo pipefail
CAMPAIGN=/scratch/h99859yz/procgen_rat_shared_actor_critic_contribution_telemetry_2m_s0_20260828_62
CODE="$CAMPAIGN/bundle/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_rat_telemetry_only_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_rat_telemetry_only_v1_2m.yaml"
ROOT="$CAMPAIGN/gate/production"
[ ! -e "$ROOT" ] || exit 90
mkdir -p "$ROOT/runtime"
ln -s "$CODE/configs" "$ROOT/runtime/configs"
echo PRECHECK_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/task62_gate_gpuh.sbatch" "$CAMPAIGN/frozen/aggregate_task62.py" > "$ROOT/frozen.sha256"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
(cd "$ROOT/runtime"; "$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name bigfish-easy-0-10 --seed 0 --device 0 --gate_mode) > "$ROOT/stdout.log" 2> "$ROOT/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$ROOT/rc"
LOGDIR=$(find "$ROOT/runtime/logs" -mindepth 2 -maxdepth 2 -type d -name 'bigfish-easy-0-10.*_0' -print 2>/dev/null | sort | tail -1)
if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
  printf '%s\n' "$LOGDIR" > "$ROOT/source_log_dir"
  cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
fi
if [ "$rc" -eq 0 ] && "$PY" "$CAMPAIGN/frozen/aggregate_task62.py" --gate "$ROOT/metric_trace.jsonl" > "$ROOT/gate_validation.json"; then
  echo PRECHECK_PASS > "$ROOT/status"
  touch "$ROOT/precheck_pass.marker"
  exit 0
fi
echo PRECHECK_FAIL > "$ROOT/status"
exit 1
