#!/bin/bash
#SBATCH --job-name=pg2b5_paper6m
#SBATCH --partition=gpuA
#SBATCH --account=gpu-aifun
#SBATCH --qos=gpu-aifun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120

set -euo pipefail

: "${BUNDLE_ROOT:?BUNDLE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

PY=/mnt/iusers01/fatpou01/compsci01/h99859yz/.RLvenv/bin/python
CODE="$RUN_ROOT/code"

if [[ -e "$CODE" ]]; then
  echo "Refusing to reuse existing code directory: $CODE" >&2
  exit 3
fi

mkdir -p "$RUN_ROOT"
cp -a "$BUNDLE_ROOT/source" "$CODE"
cp "$BUNDLE_ROOT/SOURCE_COMMIT.txt" "$RUN_ROOT/"
cp "$BUNDLE_ROOT/run_2gpu_csf3.sbatch" "$RUN_ROOT/"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

{
  date --iso-8601=seconds
  hostname
  echo "SOURCE_COMMIT=2b5affd64cbb3c624b4bc1f4767f449df231ffb2"
  echo "SCOPE=procgen_only"
  echo "CONTROLLED_OVERRIDE=timesteps_per_proc_easy:6000000"
  echo "METHODS=rat,ppo"
  echo "ENVS=bigfish,bossfight,caveflyer,coinrun,jumper,maze,miner,starpilot"
  echo "SEEDS=0,1,2,3,4"
  echo "LAYOUT=2_GPUs,5_seeds_concurrent_per_GPU,4_RAT_and_4_PPO_envs_per_GPU"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  sha256sum "$CODE/train_shared.py" "$CODE/configs/adv_resnet_shared.yaml" "$CODE/configs/ppo_resnet_shared.yaml"
  "$PY" -V
  "$PY" -c 'import importlib.metadata as m, torch, procgen, gym3, yaml; print("torch", torch.__version__, "cuda", torch.version.cuda); print("procgen", m.version("procgen"), procgen.__file__); print("gym3", m.version("gym3"), gym3.__file__); print("yaml", yaml.__version__)'
  nvidia-smi
} > "$RUN_ROOT/run_info.txt" 2>&1

(
  while true; do
    date +%s
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
    sleep 15
  done
) > "$RUN_ROOT/gpu.csv" 2>&1 &
MON_PID=$!

ENVS=(
  bigfish-easy-0-10
  bossfight-easy-0-10
  caveflyer-easy-0-10
  coinrun-easy-0-10
  jumper-easy-0-10
  maze-easy-0-10
  miner-easy-0-10
  starpilot-easy-0-10
)

# Adjacent entries form a pair. This gives each GPU four RAT and four PPO
# environments while keeping both cards on the same method in each wave.
TASK_METHODS=(rat rat ppo ppo rat rat ppo ppo rat rat ppo ppo rat rat ppo ppo)
TASK_ENVS=(0 1 0 1 2 3 2 3 4 5 4 5 6 7 6 7)

ALL_SEED_PIDS=()

run_one() {
  local task_index="$1"
  local gpu_id="$2"
  local method="${TASK_METHODS[$task_index]}"
  local env_index="${TASK_ENVS[$task_index]}"
  local env_name="${ENVS[$env_index]}"
  local config
  if [[ "$method" == rat ]]; then
    config=adv_resnet_shared.yaml
  else
    config=ppo_resnet_shared.yaml
  fi

  local run_dir="$RUN_ROOT/runs/$method/$env_name"
  mkdir -p "$run_dir"
  echo RUNNING > "$run_dir/status"
  {
    date --iso-8601=seconds
    echo "task_index=$task_index"
    echo "gpu_id=$gpu_id"
    echo "method=$method"
    echo "env_name=$env_name"
    echo "config=$config"
  } > "$run_dir/task_info.txt"

  local pids=()
  local seed
  for seed in 0 1 2 3 4; do
    echo "CUDA_VISIBLE_DEVICES=$gpu_id $PY -u train_shared.py --config $config --env_name $env_name --seed $seed --device 0" > "$run_dir/seed${seed}.command.txt"
    (
      cd "$CODE"
      CUDA_VISIBLE_DEVICES="$gpu_id" "$PY" -u train_shared.py \
        --config "$config" \
        --env_name "$env_name" \
        --seed "$seed" \
        --device 0
    ) > "$run_dir/seed${seed}.stdout" 2> "$run_dir/seed${seed}.stderr" &
    pids+=("$!")
    ALL_SEED_PIDS+=("$!")
  done

  local overall=0
  local idx
  for idx in 0 1 2 3 4; do
    set +e
    wait "${pids[$idx]}"
    local rc=$?
    set -e
    echo "$rc" > "$run_dir/seed${idx}.rc"
    if [[ "$rc" -ne 0 ]]; then
      overall=1
    fi
  done

  echo "$overall" > "$run_dir/returncode"
  if [[ "$overall" -eq 0 ]]; then
    echo PASS > "$run_dir/status"
  else
    echo FAIL > "$run_dir/status"
  fi
  return "$overall"
}

terminate_children() {
  echo TERMINATING > "$RUN_ROOT/status"
  local pid
  for pid in "${ALL_SEED_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  kill "$MON_PID" 2>/dev/null || true
}
trap terminate_children TERM INT

worker() {
  local worker_index="$1"
  local overall=0
  local task_index
  for ((task_index=worker_index; task_index<16; task_index+=2)); do
    if ! run_one "$task_index" "$worker_index"; then
      overall=1
    fi
  done
  return "$overall"
}

echo RUNNING > "$RUN_ROOT/status"
worker 0 &
WORKER0=$!
worker 1 &
WORKER1=$!

OVERALL=0
if ! wait "$WORKER0"; then OVERALL=1; fi
if ! wait "$WORKER1"; then OVERALL=1; fi

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true

echo "$OVERALL" > "$RUN_ROOT/returncode"
if [[ "$OVERALL" -eq 0 ]]; then
  echo PASS > "$RUN_ROOT/status"
else
  echo FAIL > "$RUN_ROOT/status"
fi
exit "$OVERALL"
