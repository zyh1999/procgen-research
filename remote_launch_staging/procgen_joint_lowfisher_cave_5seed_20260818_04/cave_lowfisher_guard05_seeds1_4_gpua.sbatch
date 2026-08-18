#!/bin/bash --login
#SBATCH --job-name=pg-jb-cave-g
#SBATCH --partition=gpuA
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --array=1-4%4
#SBATCH --no-requeue
#SBATCH --output=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.out
#SBATCH --error=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/slurm/%x_%A_%a.err

set -Eeuo pipefail
ROOT=/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1
CODE="$ROOT/code"
PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
TRAINER=train_shared_jointb_rhsaligned_deterministic_lowfisherguard.py
CONFIG=adv_resnet_shared_jointb_rhsaligned_actorrelative_criticfloor05_crossguard05_1m.yaml
EXPECTED_TRAINER_SHA=18eea9d75dab6926788673b3bbe9c9ae26468dcbe0688c9a5e9ef150e1751526
EXPECTED_CONFIG_SHA=7c2ad5efbb004ec36d816143b6ae6b8513d05621c92d21a70a938f48358d06cf
EXPECTED_LAUNCHER_SHA=${EXPECTED_LAUNCHER_SHA:?}
FROZEN_COMMIT=${FROZEN_COMMIT:?}
TOTAL_TIMESTEPS=1000000
ENV_NAME=caveflyer-easy-0-10
SEED=${SLURM_ARRAY_TASK_ID:?}
RUN_ROOT="$ROOT/gate_1m_cave5seed_20260818_04/lowfisher_guard05_rhs_aligned_jointb"
RUN="$RUN_ROOT/$ENV_NAME/seed$SEED"

trainer_sha=$(sha256sum "$CODE/$TRAINER" | awk '{print $1}')
config_sha=$(sha256sum "$CODE/configs/$CONFIG" | awk '{print $1}')
launcher_sha=$(sha256sum "$0" | awk '{print $1}')
[[ "$trainer_sha" == "$EXPECTED_TRAINER_SHA" ]]
[[ "$config_sha" == "$EXPECTED_CONFIG_SHA" ]]
[[ "$launcher_sha" == "$EXPECTED_LAUNCHER_SHA" ]]
mkdir -p "$(dirname "$RUN")" "$ROOT/slurm"
if ! mkdir "$RUN"; then echo "run already claimed: $RUN" >&2; exit 3; fi

export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
{
  echo "TASK_ID=PROCGEN-JOINT-LOWFISHER-CAVE-5SEED-GATE-20260818-04"
  echo "FROZEN_COMMIT=$FROZEN_COMMIT"
  echo "HOST=$(hostname)"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  echo "ENV_NAME=$ENV_NAME"
  echo "METHOD=lowfisher_guard05_rhs_aligned_jointb"
  echo "SEED=$SEED"
  echo "TRAINER_SHA256=$trainer_sha"
  echo "CONFIG_SHA256=$config_sha"
  echo "LAUNCHER_SHA256=$launcher_sha"
  echo "ROLLOUT=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=$TOTAL_TIMESTEPS"
  echo "PARENT_ROWS=1024"
  echo "REDUCED_SYSTEM_ROWS=512"
  echo "JOINT_LINEAR_SOLVER=rhs_aligned_rank1_b"
  echo "REDUCTION=deterministic_rhs_aligned_rank1_galerkin"
  echo "RHS_PROJECTION=transformed_sqrt_ratio_rhs"
  echo "FIRST_ORDER_GRADIENT_PRESERVED=1"
  echo "ACTOR_ABSOLUTE_DAMPING=0.003"
  echo "CRITIC_ABSOLUTE_DAMPING=0.5"
  echo "BLOCK_RELATIVE_DAMPING=0.10"
  echo "ACTOR_FROM_CRITIC_FLOOR=0.01"
  echo "LOW_FISHER_ACTOR_CRITIC_GUARD=1"
  echo "LOW_FISHER_GUARD_HIGH=0.50"
  echo "LOW_FISHER_GUARD_LOW=0.20"
  echo "LOW_FISHER_GUARD_MAX=0.05"
  echo "FULL_COMPRESSED_CROSS_TERMS=1"
  echo "CRITIC_SCORE=clean"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "LINEAR_SOLVE_DTYPE=float64"
  echo "CHECKPOINT_STRATEGY=none_by_launcher_design"
} > "$RUN/preflight"

cmd=("$PY" -u "$TRAINER" --config "$CONFIG"
  --env_name "$ENV_NAME" --seed "$SEED" --device 0
  --total_timesteps "$TOTAL_TIMESTEPS" --joint_ablation_mode full_joint
  --joint_critic_score_mode clean --joint_critic_param_scope all
  --joint_critic_reconstruction_scope all
  --joint_critic_curvature_coef 1.0 --joint_critic_objective_coef 1.0)
printf '%q ' "${cmd[@]}" > "$RUN/command.txt"; printf '\n' >> "$RUN/command.txt"
echo RUNNING > "$RUN/status"
cd "$CODE"
export PROCGEN_METRIC_TRACE_PATH="$RUN/metric_trace.jsonl"
set +e
"${cmd[@]}" > "$RUN/stdout" 2> "$RUN/stderr"
rc=$?
set -e
echo "$rc" > "$RUN/rc"
if [[ $rc -eq 0 ]]; then echo PASS > "$RUN/status"; else echo FAILED > "$RUN/status"; fi
exit "$rc"
