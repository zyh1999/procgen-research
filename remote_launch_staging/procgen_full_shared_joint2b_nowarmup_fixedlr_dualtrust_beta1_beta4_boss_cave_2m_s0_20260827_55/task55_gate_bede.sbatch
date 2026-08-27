#!/bin/bash
#SBATCH --job-name=pg55-nowarm-gate
#SBATCH --account=bdman37g
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_nowarmup_fixedlr_dualtrust_beta1_beta4_boss_cave_2m_s0_20260827_55/scheduler/%x_%j.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_nowarmup_fixedlr_dualtrust_beta1_beta4_boss_cave_2m_s0_20260827_55/scheduler/%x_%j.err

set -uo pipefail
CAMPAIGN=/nobackup/projects/bdman37/yihe/procgen_full_shared_joint2b_nowarmup_fixedlr_dualtrust_beta1_beta4_boss_cave_2m_s0_20260827_55
CODE="$CAMPAIGN/bundle/code"
PY=/nobackup/projects/bdman37/yihe/ppc64le/envs/procgen_author/bin/python
TRAINER="$CODE/train_full_shared_joint2b_nowarmup_fixedlr_dualtrust_v1.py"
overall_rc=0
for ARM in beta1 beta4; do
  ROOT="$CAMPAIGN/gate/production/$ARM"
  CONFIG="$CODE/configs/adv_resnet_full_shared_joint2b_nowarmup_fixedlr_dualtrust_${ARM}_v1_2m.yaml"
  if [ -e "$ROOT" ]; then exit 90; fi
  mkdir -p "$ROOT/runtime"; ln -s "$CODE/configs" "$ROOT/runtime/configs"
  echo PRECHECK_RUNNING > "$ROOT/status"
  printf '%s\n' "$SLURM_JOB_ID" > "$ROOT/job_id"
  hostname -f > "$ROOT/hostname"
  nvidia-smi -L > "$ROOT/gpu.txt" 2>&1
  sha256sum "$TRAINER" "$CONFIG" "$CAMPAIGN/frozen/task55_gate_bede.sbatch" > "$ROOT/frozen.sha256"
  export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0 PYTHONHASHSEED=0
  (cd "$ROOT/runtime"; "$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name bigfish-easy-0-10 --seed 0 --device 0 --gate_mode) > "$ROOT/stdout.log" 2> "$ROOT/stderr.log"
  rc=$?; printf '%s\n' "$rc" > "$ROOT/rc"
  LOGDIR=$(find "$ROOT/runtime/logs" -mindepth 2 -maxdepth 2 -type d -name 'bigfish-easy-0-10.*_0' -print 2>/dev/null | sort | tail -1)
  if [ -n "$LOGDIR" ] && [ -d "$LOGDIR" ]; then
    printf '%s\n' "$LOGDIR" > "$ROOT/source_log_dir"
    cp "$LOGDIR/progress.csv" "$ROOT/progress.csv" 2>/dev/null || true
    cp "$LOGDIR/metric_trace.jsonl" "$ROOT/metric_trace.jsonl" 2>/dev/null || true
    cp "$LOGDIR/rollout_scheduler.jsonl" "$ROOT/rollout_scheduler.jsonl" 2>/dev/null || true
  fi
  expected_beta=1.0; [ "$ARM" = beta4 ] && expected_beta=4.0
  phase_switch_count=$(find "$ROOT/runtime/logs" -name phase_switch.jsonl -type f -size +0c 2>/dev/null | wc -l | tr -d ' ')
  if [ "$rc" -eq 0 ] \
    && [ "$phase_switch_count" -eq 0 ] \
    && ! grep -q '"training_phase": "ppo_warmup"' "$ROOT/metric_trace.jsonl" \
    && grep -q '"training_phase": "joint2b"' "$ROOT/metric_trace.jsonl" \
    && grep -q '"joint_cholesky_info_max": 0' "$ROOT/metric_trace.jsonl" \
    && grep -q '"joint_finite_scan_pass": 1.0' "$ROOT/metric_trace.jsonl" \
    && grep -q '"fixed_parameter_lr": 0.004' "$ROOT/rollout_scheduler.jsonl" \
    && grep -q '"minibatch_lr_change_count": 0' "$ROOT/rollout_scheduler.jsonl" \
    && grep -q '"between_rollout_lr_change_count": 0' "$ROOT/rollout_scheduler.jsonl" \
    && grep -q '"actor_trust_update_count_this_rollout": 1' "$ROOT/rollout_scheduler.jsonl" \
    && grep -q '"critic_trust_update_count_this_rollout": 1' "$ROOT/rollout_scheduler.jsonl" \
    && grep -q "\"beta_v\": $expected_beta" "$ROOT/rollout_scheduler.jsonl" \
    && grep -q '"joint_system_rows": 1024' "$ROOT/metric_trace.jsonl" \
    && grep -q '"joint_parameter_columns": 938976' "$ROOT/metric_trace.jsonl" \
    && grep -q '"joint_actor_critic_cross_frobenius"' "$ROOT/metric_trace.jsonl" \
    && grep -q '"joint_critic_actor_cross_frobenius"' "$ROOT/metric_trace.jsonl"; then
    echo PRECHECK_PASS > "$ROOT/status"; touch "$ROOT/precheck_pass.marker"
  else
    echo PRECHECK_FAIL > "$ROOT/status"; overall_rc=1
  fi
done
[ "$overall_rc" -eq 0 ] && touch "$CAMPAIGN/gate/DUAL_ARM_PRECHECK_PASS"
exit "$overall_rc"
