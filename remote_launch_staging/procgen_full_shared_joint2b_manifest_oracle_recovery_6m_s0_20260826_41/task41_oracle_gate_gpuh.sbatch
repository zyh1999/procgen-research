#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_full_shared_joint2b_manifest_oracle_recovery_6m_s0_20260826_41/scheduler/%x_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_full_shared_joint2b_manifest_oracle_recovery_6m_s0_20260826_41/scheduler/%x_%j.err

set -uo pipefail
CAMPAIGN=/scratch/h99859yz/procgen_full_shared_joint2b_manifest_oracle_recovery_6m_s0_20260826_41
ROOT="$CAMPAIGN/local_gate"
CODE="$CAMPAIGN/bundle/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_full_shared_joint2b_scale_recovery_v1.py"
CONFIG="$CODE/configs/adv_resnet_full_shared_joint2b_scale_recovery_v1_6m.yaml"
PREFLIGHT="$CAMPAIGN/frozen/preflight_full_shared_joint2b_manifest_oracle_recovery.py"
TEST="$CAMPAIGN/frozen/test_task41_manifest_oracle.py"
if [ -e "$ROOT" ]; then exit 90; fi
mkdir -p "$ROOT/process1" "$ROOT/process2"
echo LOCAL_GATE_RUNNING > "$ROOT/status"
printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
hostname -f > "$ROOT/hostname"
nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$PREFLIGHT" "$TEST" "$CAMPAIGN/frozen/manifest_oracle.py" > "$ROOT/frozen.sha256"
(cd "$ROOT/process1"; env -u PYTHONPATH "$PY" "$PREFLIGHT" --trainer "$TRAINER" --config "$CONFIG" --env-name bigfish-easy-0-10 --emit-oracle --output "$ROOT/process1/oracle.json") > "$ROOT/process1/stdout.log" 2> "$ROOT/process1/stderr.log"
rc1=$?
(cd "$ROOT/process2"; env -u PYTHONPATH "$PY" "$PREFLIGHT" --trainer "$TRAINER" --config "$CONFIG" --env-name bigfish-easy-0-10 --emit-oracle --output "$ROOT/process2/oracle.json") > "$ROOT/process2/stdout.log" 2> "$ROOT/process2/stderr.log"
rc2=$?
if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ] || ! cmp -s "$ROOT/process1/oracle.json" "$ROOT/process2/oracle.json"; then
  printf '%s %s\n' "$rc1" "$rc2" > "$ROOT/rc"; echo LOCAL_GATE_FAIL > "$ROOT/status"; exit 1
fi
cp "$ROOT/process1/oracle.json" "$ROOT/production_parameter_oracle.json"
ORACLE_SHA=$(sha256sum "$ROOT/production_parameter_oracle.json" | awk '{print $1}')
printf '%s\n' "$ORACLE_SHA" > "$ROOT/production_parameter_oracle.sha256"
(cd "$ROOT"; env -u PYTHONPATH "$PY" "$TEST" --oracle "$ROOT/production_parameter_oracle.json" --oracle-sha256 "$ORACLE_SHA") > "$ROOT/negative_tests.out" 2> "$ROOT/negative_tests.err"
rc=$?
printf '%s\n' "$rc" > "$ROOT/rc"
if [ "$rc" -eq 0 ] && grep -q TASK41_MANIFEST_ORACLE_NEGATIVE_GATES_PASS "$ROOT/negative_tests.out"; then
  echo LOCAL_GATE_PASS > "$ROOT/status"; touch "$ROOT/local_gate_pass.marker"; exit 0
fi
echo LOCAL_GATE_FAIL > "$ROOT/status"; exit "$rc"
