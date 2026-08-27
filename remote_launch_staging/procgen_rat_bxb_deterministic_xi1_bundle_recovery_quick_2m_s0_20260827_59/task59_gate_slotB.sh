#!/bin/bash
set -Eeuo pipefail
umask 027

TASK_ID=PROCGEN-RAT-BXB-DETERMINISTIC-XI1-BUNDLE-RECOVERY-QUICK-2M-S0-20260827-59
CAMPAIGN=/scratch/h99859yz/procgen_rat_bxb_deterministic_xi1_bundle_recovery_quick_2m_s0_20260827_59
CODE="$CAMPAIGN/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_rat_weightedcritic_reference_publicmomentum.py"
CONFIG_NAME=adv_resnet_shared_rat_bxb_deterministic_xi1_v1_2m.yaml
CONFIG="$CODE/configs/$CONFIG_NAME"
TRAINER_SHA=77e005311daa52e9e09352042adedff13c0e587a60a13c92be25a592b4f355cb
CONFIG_SHA=fb4e3787e9e52212bb076744753e57e941a3bf37c82738844083ac6297f208fe
GATE="$CAMPAIGN/gate"

[ "${SLURM_JOB_ID:-}" = "19487252" ] || { echo "wrong parent allocation" >&2; exit 80; }
[ "$(hostname -s)" = "node822" ] || { echo "wrong allocated node" >&2; exit 81; }
[ ! -e "$GATE" ] || { echo "gate root collision" >&2; exit 82; }
mkdir -p "$GATE/runtime"
ln -s "$CODE/configs" "$GATE/runtime/configs"
printf '%s\n' "$TASK_ID" > "$GATE/task_id.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$GATE/parent_allocation_job"
printf '%s\n' "${SLURM_STEP_ID:-unknown}" > "$GATE/slurm_step_id"
hostname -f > "$GATE/hostname"
date -Is > "$GATE/start_time"
nvidia-smi -L > "$GATE/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/task59_gate_slotB.sh" > "$GATE/frozen_identity.sha256"
[ "$(sha256sum "$TRAINER" | awk '{print $1}')" = "$TRAINER_SHA" ] || { echo TRAINER_HASH_FAIL > "$GATE/status"; exit 70; }
[ "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$CONFIG_SHA" ] || { echo CONFIG_HASH_FAIL > "$GATE/status"; exit 71; }
for required in utils/logger.py utils/runners.py utils/utils.py vec_env/__init__.py; do
  [ -s "$CODE/$required" ] || { echo "missing dependency: $required" >&2; echo BUNDLE_FAIL > "$GATE/status"; exit 72; }
done

echo RUNNING > "$GATE/status"
CMD=("$PY" -u "$TRAINER" --config "$CONFIG_NAME" --env_name bigfish-easy-0-10 --seed 0 --device 0 --total_timesteps 4096)
printf '%q ' "${CMD[@]}" > "$GATE/command.txt"; printf '\n' >> "$GATE/command.txt"
set +e
(cd "$GATE/runtime"; env PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}" PROCGEN_METRIC_TRACE_PATH="$GATE/metric_trace.jsonl" OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 PYTHONHASHSEED=0 "${CMD[@]}" > "$GATE/stdout.log" 2> "$GATE/stderr.log")
RC=$?
set -e
printf '%s\n' "$RC" > "$GATE/rc"
[ "$RC" -eq 0 ] || { echo FAILED > "$GATE/status"; exit "$RC"; }

"$PY" -c 'import json, math, pathlib; p=pathlib.Path("'$GATE'/metric_trace.jsonl"); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; assert rows; r=rows[0]; assert r["joint_system_rows"]==512; assert r["joint_rhs_columns"]==2; assert r["joint_kernel_mode"]=="rat_reference_combined_deterministic_xi1_b"; assert r["joint_critic_score_mode"]=="deterministic_unit_value_score"; assert r["critic_score_noise_mean"]==1.0 and r["critic_score_noise_std"]==0.0; assert r["critic_score_noise_second_moment"]==1.0; assert r["critic_score_noise_min"]==1.0 and r["critic_score_noise_max"]==1.0; assert math.isfinite(r["joint_solve_residual"]); assert math.isfinite(r["rat_critic_solve_residual"]); assert r["joint_rhs_mode"]=="combined_score_adv_plus_reference_rat_ones"; print(json.dumps(r, sort_keys=True))' > "$GATE/validated_first_trace.json"
if grep -Eiq 'Traceback|out of memory|CUDA error|NCCL|nonfinite|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)|(^|[^[:alpha:]])Inf([^[:alpha:]]|$)' "$GATE/stdout.log" "$GATE/stderr.log"; then
  echo HARD_ERROR_SCAN_FAIL > "$GATE/status"
  exit 73
fi
echo PRECHECK_PASS > "$GATE/status"
date -Is > "$GATE/end_time"

