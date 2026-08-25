#!/bin/bash --login
#SBATCH --partition=gpuL
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_gae_ggn_head_widentity_6m_s0_20260825_33/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_gae_ggn_head_widentity_6m_s0_20260825_33/scheduler/%x_%j.err

set -uo pipefail
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
case "$ENV_NAME" in bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;; *) exit 64;; esac
SEED=${PROCGEN_SEED:?PROCGEN_SEED required}
case "$SEED" in 0|1) ;; *) exit 65;; esac
TASK_ID=PROCGEN-GAE-GGN-HEAD-WIDENTITY-6M-S0-20260825-33
CAMPAIGN=/scratch/h99859yz/procgen_gae_ggn_head_widentity_6m_s0_20260825_33
CODE="$CAMPAIGN/code"
FROZEN="$CAMPAIGN/frozen"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_det_gae_ggn_head_widentity_v1.py"
CONFIG="$CODE/configs/adv_resnet_shared_det_gae_ggn_head_widentity_v1_6m.yaml"
LAUNCHER="$FROZEN/gae_ggn_head_widentity_6m_multiseed_gpul.sbatch"
PREFLIGHT="$FROZEN/gpul_compatibility_wrapper_task33.py"
METHOD=DET_GAE_GGN_HEAD_WIDENTITY_V1
ROOT="$CAMPAIGN/runs/$METHOD/$ENV_NAME/seed${SEED}/6m"

if [ -e "$ROOT" ]; then echo "collision: $ROOT" >&2; exit 90; fi
mkdir -p "$ROOT" "$CODE/logs"
printf '%s\n' "$TASK_ID" > "$ROOT/task_id.txt"
printf '%s\n' "$CAMPAIGN" > "$ROOT/campaign_root.txt"
printf '%s\n' "$SEED" > "$ROOT/seed.txt"
echo PREFLIGHT_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$LAUNCHER" "$FROZEN/gpuh_preflight.py" "$PREFLIGHT" > "$ROOT/preflight.sha256"
if ! PROCGEN_ENV="$ENV_NAME" "$PY" "$PREFLIGHT" "$TRAINER" "$CONFIG" "$ROOT/parameter_partition.json" 9c822949a171d7b4c8148ea9be37c1eb5b6aa3ff8d9d3f1603241bc4873c2665 4458567bc61bdb85360d55967c65700992fe60cd812819fa1e77cc8638c202af > "$ROOT/compatibility.out" 2> "$ROOT/compatibility.err"; then
  echo INFRASTRUCTURE_PREFLIGHT_FAIL > "$ROOT/status"; exit 70
fi
grep -q GPUL_TASK33_DEPLOYMENT_COMPATIBILITY_PASS "$ROOT/compatibility.out" || { echo INFRASTRUCTURE_PREFLIGHT_FAIL > "$ROOT/status"; exit 71; }

CMD=("$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name "$ENV_NAME" --seed "$SEED" --device 0)
printf '%q ' "${CMD[@]}" > "$ROOT/command.txt"; printf '\n' >> "$ROOT/command.txt"
touch "$ROOT/start.marker" "$ROOT/scientific_started.marker"
echo RUNNING > "$ROOT/status"

sync_artifacts() {
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    LOGDIR=$(find "$CODE/logs" -mindepth 2 -maxdepth 2 -type d -name "${ENV_NAME}.*_${SEED}" -newer "$ROOT/start.marker" -print 2>/dev/null | sort | tail -1)
    if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
      printf '%s\n' "$LOGDIR" > "$ROOT/source_log_dir"
      cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
      cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
      cp "$LOGDIR/model.ckpt" "$ROOT/model.ckpt" 2>/dev/null || true
      cp "$LOGDIR/parameter_partition.json" "$ROOT/training_parameter_partition.json" 2>/dev/null || true
    fi
    sleep 60
  done
}
(cd "$CODE"; OMP_NUM_THREADS=8 "${CMD[@]}" > "$ROOT/stdout.log" 2> "$ROOT/stderr.log") &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$ROOT/trainer_pid"
sync_artifacts & SYNC_PID=$!
wait "$TRAIN_PID"; RC=$?
wait "$SYNC_PID" 2>/dev/null || true
LOGDIR=$(cat "$ROOT/source_log_dir" 2>/dev/null || true)
if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
  cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
  cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
  cp "$LOGDIR/model.ckpt" "$ROOT/model.ckpt" 2>/dev/null || true
  cp "$LOGDIR/parameter_partition.json" "$ROOT/training_parameter_partition.json" 2>/dev/null || true
fi
printf '%s\n' "$RC" > "$ROOT/rc"
if [ "$RC" -eq 0 ] && [ -s "$ROOT/progress.csv" ] && [ -s "$ROOT/metric_trace.jsonl" ] && [ -s "$ROOT/model.ckpt" ]; then echo PASS > "$ROOT/status"; else echo FAIL > "$ROOT/status"; fi
exit "$RC"
