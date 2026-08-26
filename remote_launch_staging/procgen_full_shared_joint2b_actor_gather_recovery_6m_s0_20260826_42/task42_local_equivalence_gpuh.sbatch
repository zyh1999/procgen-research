#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_full_shared_joint2b_actor_gather_recovery_6m_s0_20260826_42/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_full_shared_joint2b_actor_gather_recovery_6m_s0_20260826_42/scheduler/%x_%j.err

set -uo pipefail
CAMPAIGN=/scratch/h99859yz/procgen_full_shared_joint2b_actor_gather_recovery_6m_s0_20260826_42
TASK41=/scratch/h99859yz/procgen_full_shared_joint2b_manifest_oracle_recovery_6m_s0_20260826_41
ROOT="$CAMPAIGN/local_gate"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$TASK41/bundle/code/train_full_shared_joint2b_scale_recovery_v1.py"
CONFIG="$TASK41/bundle/code/configs/adv_resnet_full_shared_joint2b_scale_recovery_v1_6m.yaml"
ORACLE="$TASK41/local_gate/production_parameter_oracle.json"
ORACLE_SHA=62b1b10a81fc89ec621f0ceaf60735864cc899fafd6bda074c7414744506d303
TASK41_PREFLIGHT="$TASK41/frozen/preflight_full_shared_joint2b_manifest_oracle_recovery.py"
if [ -e "$ROOT" ]; then exit 90; fi
test "$(cat "$TASK41/local_gate/status")" = LOCAL_GATE_PASS || exit 70
test "$(sha256sum "$ORACLE" | awk '{print $1}')" = "$ORACLE_SHA" || exit 71
mkdir -p "$ROOT"
echo LOCAL_EQUIVALENCE_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$ORACLE" "$TASK41_PREFLIGHT" "$CAMPAIGN/frozen/"*.py > "$ROOT/frozen.sha256"
(cd "$ROOT"; env -u PYTHONPATH "$PY" "$CAMPAIGN/frozen/test_actor_gather_equivalence.py") > "$ROOT/tensor_equivalence.out" 2> "$ROOT/tensor_equivalence.err"
rc1=$?
if [ "$rc1" -ne 0 ] || ! grep -q TASK42_GATHER_VALUE_LOGITS_GRAD_NEGATIVE_PASS "$ROOT/tensor_equivalence.out"; then
  printf '%s\n' "$rc1" > "$ROOT/rc"; echo LOCAL_EQUIVALENCE_FAIL > "$ROOT/status"; exit "$rc1"
fi
(cd "$ROOT"; env -u PYTHONPATH "$PY" "$CAMPAIGN/frozen/production_actor_gather_equivalence.py" \
  --task41-preflight "$TASK41_PREFLIGHT" --trainer "$TRAINER" --config "$CONFIG" \
  --oracle "$ORACLE" --oracle-sha256 "$ORACLE_SHA" --output "$ROOT/production_equivalence.json") \
  > "$ROOT/production_equivalence.out" 2> "$ROOT/production_equivalence.err"
rc=$?
printf '%s\n' "$rc" > "$ROOT/rc"
if [ "$rc" -eq 0 ] && grep -q TASK42_PRODUCTION_ACTOR_GATHER_512ROW_EQUIVALENCE_PASS "$ROOT/production_equivalence.out"; then
  echo LOCAL_EQUIVALENCE_PASS > "$ROOT/status"; touch "$ROOT/local_equivalence_pass.marker"; exit 0
fi
echo LOCAL_EQUIVALENCE_FAIL > "$ROOT/status"; exit "$rc"
