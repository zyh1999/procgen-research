#!/bin/bash
#SBATCH --job-name=pg-rat-e500
#SBATCH --account=bdman37
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@60
#SBATCH --output=/nobackup/projects/bdman37/yihe/procgen_rat_expectedcritic_20260816_v1/slurm/%A_%a.out
#SBATCH --error=/nobackup/projects/bdman37/yihe/procgen_rat_expectedcritic_20260816_v1/slurm/%A_%a.err

set -euo pipefail
ROOT=/nobackup/projects/bdman37/yihe/procgen_rat_expectedcritic_20260816_v1
CODE=/nobackup/projects/bdman37/yihe/procgen_joint2b_dualdamping_schur_20260816_v1/code
PY=/nobackup/projects/bdman37/yihe/ppc64le/envs/rat/bin/python
TRAINER=train_shared_rat_expectedcritic_dualdamping.py
CONFIG=adv_resnet_shared_rat_expectedcritic_dualdamping_500k.yaml
EXPECTED_TRAINER_SHA=0514703d9fb6ca17cc68febabb012defb279ab5a54f57cf95365422164848934
EXPECTED_CONFIG_SHA=94c44aa25221316e24f916849221499389b7eba31f26a0d2fe1f365e549690a0
TOTAL_TIMESTEPS=${PROCGEN_TOTAL_TIMESTEPS:-500000}
RUN_ROOT=${PROCGEN_RUN_ROOT:-"$ROOT/gate_500k_seed0/expected_gaussian_score_b_dualdamping"}
ENV_NAMES=(bigfish-easy-0-10 bossfight-easy-0-10 caveflyer-easy-0-10 coinrun-easy-0-10)
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
ENV_NAME=${ENV_NAMES[$TASK_ID]}
RUN="$RUN_ROOT/$ENV_NAME/seed0"

mkdir -p "$(dirname "$RUN")" "$ROOT/slurm"
if ! mkdir "$RUN"; then echo "run already claimed: $RUN" >&2; exit 3; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PROCGEN_METRIC_TRACE_PATH="$RUN/metric_trace.jsonl"

trainer_sha=$(sha256sum "$CODE/$TRAINER" | awk '{print $1}')
config_sha=$(sha256sum "$CODE/configs/$CONFIG" | awk '{print $1}')
[[ "$trainer_sha" == "$EXPECTED_TRAINER_SHA" ]]
[[ "$config_sha" == "$EXPECTED_CONFIG_SHA" ]]
echo RUNNING > "$RUN/status"
{
  date --iso-8601=seconds
  hostname
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "ENV_NAME=$ENV_NAME"
  echo "SEED=0"
  echo "TRAINER_SHA256=$trainer_sha"
  echo "CONFIG_SHA256=$config_sha"
  echo "ROLLOUT=4096"
  echo "MINIBATCH=512"
  echo "EPOCHS=4"
  echo "TRANSITIONS=$TOTAL_TIMESTEPS"
  echo "JOINT_SYSTEM_ROWS=512"
  echo "DETERMINISTIC_KERNEL=Hpi_HpiT_over_B_plus_4_diag_row_norm2_Jv_over_B"
  echo "RANDOM_SCORE_DRAWS=0"
  echo "EXPECTED_CROSS_BLOCK=0"
  echo "ACTOR_DAMPING=0.03"
  echo "CRITIC_DAMPING=0.5"
  echo "ADAPTIVE_LR_INITIAL=0.004"
  echo "ADAPTIVE_LR_MAX=0.02"
  echo "KL_RANGE=0.005,0.04"
  echo "CRITIC_UPDATE=reference_rat_weighted_mse_backward"
  echo "MOMENTUM=0"
  echo "KACZMARZ=false"
  echo "LINEAR_SOLVE_DTYPE=float64"
  "$PY" -V
  "$PY" -c 'import torch, procgen, gym3; print("torch", torch.__version__, "cuda", torch.version.cuda)'
  nvidia-smi
} > "$RUN/preflight" 2>&1

cmd=("$PY" -u "$TRAINER" --config "$CONFIG"
  --env_name "$ENV_NAME" --seed 0 --device 0
  --total_timesteps "$TOTAL_TIMESTEPS"
  --joint_ablation_mode rat_expected_critic
  --joint_critic_score_mode clean --joint_critic_param_scope all
  --joint_critic_curvature_coef 1.0 --joint_critic_objective_coef 1.0)
printf '%q ' "${cmd[@]}" > "$RUN/command.txt"; printf '\n' >> "$RUN/command.txt"
set +e
(cd "$CODE"; exec "${cmd[@]}") > "$RUN/stdout" 2> "$RUN/stderr"
rc=$?
set -e
echo "$rc" > "$RUN/rc"
if [[ $rc -eq 0 ]]; then echo PASS > "$RUN/status"; else echo FAILED > "$RUN/status"; fi
exit "$rc"
