#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/procgen_joint2b_dmlp1024_20260811_v1
CODE="$ROOT/code"
PY="$ROOT/.venv/bin/python"
GPU_ID=${1:?physical GPU id required}
shift

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$CODE"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0

TRAINER=train_shared_rat_gaussian_pairedrhs_2b_scale01_dmlp1024.py
CONFIG=adv_resnet_shared_gaussian_pairedrhs_2b_dmlp1024_fp64.yaml
TRAINER_SHA=eb7064c8413067c6797704c20c1d4b7de61fd2002adbe7744c3c791934359bae
CONFIG_SHA=822867a256ce44c69545211ca984bd9861dec5188d571308229a66fa7720cb7b
UTILS_SHA=f72e0affe50b1b09e026deae7547506580b61c649e3aae6957ad7f47aee8949c

run_one() {
    local env_name=$1
    local seed=$2
    local run="$ROOT/runs/$env_name/seed$seed"

    if [[ -e "$run/status" ]]; then
        echo "SKIP existing $run status=$(tr -d '\n' < "$run/status")"
        return 0
    fi
    mkdir -p "$run"
    echo RUNNING > "$run/status"

    cd "$CODE"
    [[ "$(sha256sum "$TRAINER" | awk '{print $1}')" == "$TRAINER_SHA" ]]
    [[ "$(sha256sum "configs/$CONFIG" | awk '{print $1}')" == "$CONFIG_SHA" ]]
    [[ "$(sha256sum utils/utils.py | awk '{print $1}')" == "$UTILS_SHA" ]]

    export PYTHONHASHSEED="$seed"
    export PROCGEN_METRIC_TRACE_PATH="$run/metric.jsonl"

    {
        date --iso-8601=seconds
        hostname
        echo "PHYSICAL_GPU_ID=$GPU_ID"
        echo "ENV_NAME=$env_name"
        echo "SEED=$seed"
        echo "METHOD=strict_gaussian_pairedrhs_joint_2b_dmlp1024_fp64"
        echo "SYSTEM_SHAPE=1024x1024"
        echo "ACTOR_ROWS=512"
        echo "CRITIC_ROWS=512"
        echo "CROSS_BLOCKS=retained"
        echo "CRITIC_SCORE=gaussian_unit"
        echo "CRITIC_RHS=paired_gaussian_residual"
        echo "DECISION_MLP=256-1024-256"
        echo "ROLLOUT_BATCH=4096"
        echo "MINIBATCH=512"
        echo "PPO_EPOCHS=4"
        echo "TOTAL_TRANSITIONS=6000000"
        echo "JOINT_CRITIC_CURVATURE_COEF=0.1"
        echo "JOINT_CRITIC_OBJECTIVE_COEF=1.0"
        echo "DAMPING=0.5"
        echo "OPTIMIZER=sgd"
        echo "OPTIMIZER_MOMENTUM=0.0"
        echo "KACZMARZ=false"
        echo "LINEAR_SOLVE_DTYPE=float64"
        echo "TRAINER_SHA256=$TRAINER_SHA"
        echo "CONFIG_SHA256=$CONFIG_SHA"
        echo "UTILS_SHA256=$UTILS_SHA"
        "$PY" -V
        nvidia-smi -i "$GPU_ID"
    } > "$run/preflight.txt" 2>&1

    (
        while true; do
            date +%s
            nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
            sleep 15
        done
    ) > "$run/gpu.csv" 2>&1 &
    local monitor_pid=$!

    set +e
    "$PY" -u "$TRAINER" \
        --config "$CONFIG" \
        --env_name "$env_name" \
        --seed "$seed" \
        --device 0 \
        > "$run/stdout" 2> "$run/stderr"
    local rc=$?
    set -e

    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    echo "$rc" > "$run/rc"
    if [[ "$rc" -eq 0 ]]; then
        echo PASS > "$run/status"
    else
        echo FAIL > "$run/status"
        return "$rc"
    fi
}

while [[ "$#" -gt 0 ]]; do
    env_name=$1
    seed=$2
    shift 2
    run_one "$env_name" "$seed"
done

echo PASS > "$ROOT/queue_gpu${GPU_ID}.status"
