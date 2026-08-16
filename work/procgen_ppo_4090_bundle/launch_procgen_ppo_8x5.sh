#!/usr/bin/env bash
set -u

if [[ -z "${PYTHON_BIN:-}" || -z "${RUN_ROOT:-}" || -z "${GPU_IDS:-}" ]]; then
  echo "Required: PYTHON_BIN=/path/to/python RUN_ROOT=/path/to/new/run GPU_IDS=0,1 [PACK_PER_GPU=1] [TASK_START=0] [TASK_END=39] $0" >&2
  exit 2
fi

PACK_PER_GPU="${PACK_PER_GPU:-1}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-39}"
BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE_ROOT="$BUNDLE_ROOT/source/trust-region-main"
CONFIG_NAME="ppo_resnet_shared.yaml"

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to reuse existing RUN_ROOT: $RUN_ROOT" >&2
  exit 3
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ "${#GPU_ARRAY[@]}" -lt 1 || "$PACK_PER_GPU" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU and PACK_PER_GPU must be >= 1" >&2
  exit 4
fi
if [[ "$TASK_START" -lt 0 || "$TASK_END" -gt 39 || "$TASK_START" -gt "$TASK_END" ]]; then
  echo "Required: 0 <= TASK_START <= TASK_END <= 39" >&2
  exit 5
fi

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

mkdir -p "$RUN_ROOT/runs" "$RUN_ROOT/snapshot"
cp "$CODE_ROOT/train_shared.py" "$RUN_ROOT/snapshot/"
cp "$CODE_ROOT/configs/$CONFIG_NAME" "$RUN_ROOT/snapshot/"
cp "$0" "$RUN_ROOT/snapshot/"

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "method=shared_ppo"
  echo "source_archive=trust-region-main (3).zip"
  echo "source_archive_sha256=6ea93e1285b6fd84c4b74239a18533e2ccd410c4313bb2a333aae4187e473167"
  echo "trainer_sha256=1d20658b154022450b8598949f693b3c04a9bd34eb22ad2f002d59f9573b74d1"
  echo "config_sha256=fdf1538ef199a222ea2caafe9264c5db00319a6f1882d7d86b04506522601807"
  echo "launcher_sha256=$(sha256sum "$0" | awk '{print $1}')"
  echo "pythonpath=${PYTHONPATH:-}"
  echo "envs=${ENVS[*]}"
  echo "seeds=0,1,2,3,4"
  echo "gpu_ids=$GPU_IDS"
  echo "pack_per_gpu=$PACK_PER_GPU"
  echo "task_start=$TASK_START"
  echo "task_end=$TASK_END"
  echo "timesteps_per_seed_easy=6000000"
  echo "num_envs=16"
  echo "nsteps=256"
  echo "epochs=4"
  echo "minibatches=8"
  echo "optimizer=adam"
  echo "lr=0.001"
  echo "cliprange=0.2"
  echo "use_kl_adaptive_lr=false"
  echo "shared_resnet=true"
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" -c 'import importlib.metadata as m, torch, procgen, gym3, yaml; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("procgen_version=" + m.version("procgen")); print("procgen=" + procgen.__file__); print("gym3_version=" + m.version("gym3")); print("gym3=" + gym3.__file__); print("pyyaml=" + yaml.__version__)'
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
} > "$RUN_ROOT/run_info.txt" 2>&1

echo RUNNING > "$RUN_ROOT/status"

run_one() {
  local task_index="$1"
  local gpu_id="$2"
  local env_index=$((task_index / 5))
  local seed=$((task_index % 5))
  local env_name="${ENVS[$env_index]}"
  local run_dir="$RUN_ROOT/runs/$env_name/seed$seed"

  mkdir -p "$run_dir"
  {
    echo "CUDA_VISIBLE_DEVICES=$gpu_id"
    echo "$PYTHON_BIN -u train_shared.py --config $CONFIG_NAME --env_name $env_name --seed $seed --device 0"
  } > "$run_dir/command.txt"

  (
    cd "$CODE_ROOT"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" -u train_shared.py \
      --config "$CONFIG_NAME" \
      --env_name "$env_name" \
      --seed "$seed" \
      --device 0
  ) > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
  local rc=$?
  echo "$rc" > "$run_dir/returncode"
  return "$rc"
}

worker() {
  local worker_index="$1"
  local gpu_id="$2"
  local worker_count="$3"
  local overall=0
  local task_index

  for ((task_index=TASK_START+worker_index; task_index<=TASK_END; task_index+=worker_count)); do
    if ! run_one "$task_index" "$gpu_id"; then
      overall=1
    fi
  done
  return "$overall"
}

TOTAL_WORKERS=$((${#GPU_ARRAY[@]} * PACK_PER_GPU))
WORKER_PIDS=()
for ((worker_index=0; worker_index<TOTAL_WORKERS; worker_index++)); do
  gpu_id="${GPU_ARRAY[$((worker_index % ${#GPU_ARRAY[@]}))]}"
  worker "$worker_index" "$gpu_id" "$TOTAL_WORKERS" &
  WORKER_PIDS+=("$!")
done

OVERALL=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    OVERALL=1
  fi
done

echo "$OVERALL" > "$RUN_ROOT/returncode"
if [[ "$OVERALL" -eq 0 ]]; then
  echo PASS > "$RUN_ROOT/status"
else
  echo FAIL > "$RUN_ROOT/status"
fi
exit "$OVERALL"
