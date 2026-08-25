#!/bin/bash --login
#SBATCH --partition=gpuH
#SBATCH --account=gpu-h200-fse-pgdr
#SBATCH --qos=gpu-h200-fse
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:25:00
#SBATCH --no-requeue
#SBATCH --job-name=pg-normmatch-v2-preflight
#SBATCH --output=/scratch/h99859yz/procgen_paper_hybrid_head_normmatch_detggn_6m_s0_20260825_14/preflight/slurm_%j.out
#SBATCH --error=/scratch/h99859yz/procgen_paper_hybrid_head_normmatch_detggn_6m_s0_20260825_14/preflight/slurm_%j.err

set -euo pipefail
CAMPAIGN=/scratch/h99859yz/procgen_paper_hybrid_head_normmatch_detggn_6m_s0_20260825_14
CODE="$CAMPAIGN/code"
FROZEN="$CAMPAIGN/frozen"
ENV_NAME=${PROCGEN_ENV:?PROCGEN_ENV required}
case "$ENV_NAME" in bigfish-easy-0-10|bossfight-easy-0-10|caveflyer-easy-0-10|coinrun-easy-0-10) ;; *) exit 64;; esac
OUT="$CAMPAIGN/preflight/$ENV_NAME/$SLURM_JOB_ID"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER="$CODE/train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
CONFIG="$CODE/configs/adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml"
TRAINER_SHA=0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b
CONFIG_SHA=9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda

mkdir -p "$OUT"
finalize() {
  RC=$?
  date -Is > "$OUT/end_time.txt"
  printf '%s\n' "$RC" > "$OUT/rc"
  if [ "$RC" -eq 0 ]; then printf 'PRECHECK_PASS\n' > "$OUT/status"; else printf 'PRECHECK_BLOCKED\n' > "$OUT/status"; fi
}
trap finalize EXIT
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export PROCGEN_ENV="$ENV_NAME"
date -Is > "$OUT/start_time.txt"
printf '%s\n' "$SLURM_JOB_ID" > "$OUT/job_id"
hostname -f > "$OUT/hostname"
nvidia-smi -L > "$OUT/gpu.txt" 2>&1
sha256sum "$TRAINER" "$CONFIG" "$FROZEN/reference_hybrid_v1.py" "$FROZEN/gpuh_preflight_normmatch_v2.py" "$FROZEN/test_hybrid_head_normmatch_v2.py" "$FROZEN/normmatch_v2_6m_gpuh.sbatch" "$FROZEN/normmatch_v2_preflight_gpuh.sbatch" > "$OUT/input.sha256"

"$PY" "$FROZEN/test_hybrid_head_normmatch_v2.py" > "$OUT/regression.out" 2> "$OUT/regression.err"
grep -q HYBRID_HEAD_REGRESSION_PASS "$OUT/regression.out"
"$PY" "$FROZEN/gpuh_preflight_normmatch_v2.py" "$TRAINER" "$CONFIG" "$OUT/parameter_partition.json" "$TRAINER_SHA" "$CONFIG_SHA" > "$OUT/compatibility.out" 2> "$OUT/compatibility.err"
grep -q GPUH_HYBRID_HEAD_COMPATIBILITY_PASS "$OUT/compatibility.out"
grep -q actual_network_head_proposal_normmatch=PASS "$OUT/compatibility.out"
grep -q actual_network_global_clip_scale=BIT_IDENTICAL_REUSED "$OUT/compatibility.out"
grep -q actual_network_rng_state_across_normmatch=BIT_IDENTICAL "$OUT/compatibility.out"
grep -q connectivity_semantic_check=PASS "$OUT/compatibility.out"

echo NORMMATCH_V2_PREFLIGHT_PASS
