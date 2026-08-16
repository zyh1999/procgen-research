#!/usr/bin/env bash
set -u

if [[ -z "${PYTHON_BIN:-}" || -z "${RUN_ROOT:-}" || -z "${GPU_ID:-}" || -z "${TASK_START:-}" || -z "${TASK_END:-}" ]]; then
  echo "Required: PYTHON_BIN=/path/to/python RUN_ROOT=/path/to/new/run GPU_ID=0 TASK_START=0 TASK_END=39 $0" >&2
  exit 2
fi

if [[ "$TASK_START" -lt 0 || "$TASK_END" -gt 39 || "$TASK_START" -gt "$TASK_END" ]]; then
  echo "Required: 0 <= TASK_START <= TASK_END <= 39" >&2
  exit 3
fi

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE_ROOT="$BUNDLE_ROOT/source/trust-region-main"
TRAINER="train_shared_procgen_maincfg_pklbranch.py"
CONFIG="adv_resnet_shared_procgen_maincfg_pklbranch.yaml"
TRAINER_SHA="f4cfcd3a5dd9ea84e9d7533a5f17c2d897db545a49d352850df89bdc69142369"
CONFIG_SHA="476b210d9da6e1dc973cf293d812a5c1e2f3c6f20654736a9687e397131da1ca"

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

if [[ -e "$RUN_ROOT" ]]; then
  echo "Refusing to reuse existing RUN_ROOT: $RUN_ROOT" >&2
  exit 4
fi

if [[ "$(sha256sum "$CODE_ROOT/$TRAINER" | awk '{print $1}')" != "$TRAINER_SHA" ]]; then
  echo "Trainer hash mismatch" >&2
  exit 5
fi
if [[ "$(sha256sum "$CODE_ROOT/configs/$CONFIG" | awk '{print $1}')" != "$CONFIG_SHA" ]]; then
  echo "Config hash mismatch" >&2
  exit 6
fi

mkdir -p "$RUN_ROOT/runs" "$RUN_ROOT/snapshot"
cp "$CODE_ROOT/$TRAINER" "$RUN_ROOT/snapshot/"
cp "$CODE_ROOT/configs/$CONFIG" "$RUN_ROOT/snapshot/"
cp "$0" "$RUN_ROOT/snapshot/"
cp "$BUNDLE_ROOT/bede_run_info.txt" "$RUN_ROOT/snapshot/"

{
  echo "created_at=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "method=shared_exact_rat"
  echo "bede_formal_job=1062382"
  echo "bede_formal_job_name=pg_rat8_pkl04"
  echo "source_archive=trust-region-main (3).zip"
  echo "trainer_sha256=$TRAINER_SHA"
  echo "config_sha256=$CONFIG_SHA"
  echo "launcher_sha256=$(sha256sum "$0" | awk '{print $1}')"
  echo "pythonpath=${PYTHONPATH:-}"
  echo "gpu_id=$GPU_ID"
  echo "pack_per_gpu=1"
  echo "task_start=$TASK_START"
  echo "task_end=$TASK_END"
  echo "seeds=0,1,2,3,4"
  echo "timesteps_per_seed_easy=6000000"
  echo "num_envs=16"
  echo "nsteps=256"
  echo "epochs=4"
  echo "minibatches=8"
  echo "optimizer=sgd_momentum_0.1"
  echo "initial_lr=0.5"
  echo "cg_damping=0.5"
  echo "use_kl_adaptive_lr=true"
  echo "use_procgen_kl_thresholds=true"
  echo "adaptive_kl_upper=0.04"
  echo "adaptive_kl_lower=0.005"
  "$PYTHON_BIN" -V
  "$PYTHON_BIN" -c 'import importlib.metadata as m, torch, procgen, gym3, yaml; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("procgen=" + m.version("procgen")); print("gym3=" + m.version("gym3")); print("pyyaml=" + yaml.__version__)'
  nvidia-smi -i "$GPU_ID" --query-gpu=index,name,driver_version,memory.total,power.limit --format=csv,noheader
} > "$RUN_ROOT/run_info.txt" 2>&1

echo RUNNING > "$RUN_ROOT/status"

(
  while true; do
    date --iso-8601=seconds
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
    sleep 30
  done
) > "$RUN_ROOT/gpu_monitor.csv" 2>&1 &
MON_PID=$!
CHILD_PID=""

terminate_children() {
  echo TERMINATING > "$RUN_ROOT/status"
  if [[ -n "$CHILD_PID" ]]; then
    kill "$CHILD_PID" 2>/dev/null || true
  fi
  kill "$MON_PID" 2>/dev/null || true
}
trap terminate_children TERM INT

run_one() {
  local task_index="$1"
  local env_index=$((task_index / 5))
  local seed=$((task_index % 5))
  local env_name="${ENVS[$env_index]}"
  local run_dir="$RUN_ROOT/runs/$env_name/seed$seed"

  mkdir -p "$run_dir"
  {
    echo "CUDA_VISIBLE_DEVICES=$GPU_ID"
    echo "$PYTHON_BIN -u $TRAINER --config $CONFIG --env_name $env_name --seed $seed --device 0"
  } > "$run_dir/command.txt"

  (
    cd "$CODE_ROOT"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u "$TRAINER" \
      --config "$CONFIG" \
      --env_name "$env_name" \
      --seed "$seed" \
      --device 0
  ) > "$run_dir/stdout.log" 2> "$run_dir/stderr.log" &
  CHILD_PID=$!
  wait "$CHILD_PID"
  local rc=$?
  CHILD_PID=""
  echo "$rc" > "$run_dir/returncode"
  return "$rc"
}

OVERALL=0
for ((task_index=TASK_START; task_index<=TASK_END; task_index++)); do
  if ! run_one "$task_index"; then
    OVERALL=1
    break
  fi
done

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
echo "$OVERALL" > "$RUN_ROOT/returncode"
if [[ "$OVERALL" -eq 0 ]]; then
  echo PASS > "$RUN_ROOT/status"
else
  echo FAIL > "$RUN_ROOT/status"
fi
exit "$OVERALL"
