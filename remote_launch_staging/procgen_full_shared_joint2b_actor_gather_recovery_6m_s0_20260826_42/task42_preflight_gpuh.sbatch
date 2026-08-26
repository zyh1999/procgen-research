#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_full_shared_joint2b_actor_gather_recovery_6m_s0_20260826_42/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_full_shared_joint2b_actor_gather_recovery_6m_s0_20260826_42/scheduler/%x_%j.err

set -uo pipefail
CAMPAIGN=/scratch/h99859yz/procgen_full_shared_joint2b_actor_gather_recovery_6m_s0_20260826_42
TASK41=/scratch/h99859yz/procgen_full_shared_joint2b_manifest_oracle_recovery_6m_s0_20260826_41
ROOT="$CAMPAIGN/preflight/production"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$TASK41/bundle/code/train_full_shared_joint2b_scale_recovery_v1.py"
CONFIG="$TASK41/bundle/code/configs/adv_resnet_full_shared_joint2b_scale_recovery_v1_6m.yaml"
ORACLE="$TASK41/local_gate/production_parameter_oracle.json"
ORACLE_SHA=62b1b10a81fc89ec621f0ceaf60735864cc899fafd6bda074c7414744506d303
TASK41_PREFLIGHT="$TASK41/frozen/preflight_full_shared_joint2b_manifest_oracle_recovery.py"
WRAPPER="$CAMPAIGN/frozen/task42_preflight_wrapper.py"
if [ -e "$ROOT" ]; then exit 90; fi
test "$(cat "$CAMPAIGN/local_gate/status")" = LOCAL_EQUIVALENCE_PASS || exit 70
test "$(cat "$CAMPAIGN/local_gate/rc")" = 0 || exit 71
test "$(sha256sum "$ORACLE" | awk '{print $1}')" = "$ORACLE_SHA" || exit 72
mkdir -p "$ROOT"
echo PRECHECK_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$ORACLE" "$TASK41_PREFLIGHT" "$WRAPPER" > "$ROOT/frozen.sha256"
(cd "$ROOT"; env -u PYTHONPATH "$PY" "$WRAPPER" --task41-preflight "$TASK41_PREFLIGHT" \
  --trainer "$TRAINER" --config "$CONFIG" --env-name bigfish-easy-0-10 \
  --oracle "$ORACLE" --oracle-sha256 "$ORACLE_SHA" --output "$ROOT/preflight.json") \
  > "$ROOT/stdout.log" 2> "$ROOT/stderr.log"
rc=$?
printf '%s\n' "$rc" > "$ROOT/rc"
if [ "$rc" -eq 0 ] && grep -q GPUH_FULL_SHARED_JOINT2B_MANIFEST_ORACLE_RECOVERY_PASS "$ROOT/stdout.log"; then
  echo PRECHECK_PASS > "$ROOT/status"; touch "$ROOT/precheck_pass.marker"; exit 0
fi
echo PRECHECK_FAIL > "$ROOT/status"; exit "$rc"
