#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" || -z "${GPU_ID:-}" || -z "${SMOKE_ROOT:-}" ]]; then
  echo "Required: PYTHON_BIN=/path/to/python GPU_ID=0 SMOKE_ROOT=/path/to/new/smoke $0" >&2
  exit 2
fi

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE_ROOT="$BUNDLE_ROOT/source/trust-region-main"

if [[ -e "$SMOKE_ROOT" ]]; then
  echo "Refusing to reuse existing SMOKE_ROOT: $SMOKE_ROOT" >&2
  exit 3
fi
mkdir -p "$SMOKE_ROOT"

"$PYTHON_BIN" -m py_compile "$CODE_ROOT/train_shared.py"
"$PYTHON_BIN" -c 'import torch, procgen, gym3, yaml; print("torch", torch.__version__, "cuda", torch.version.cuda); print("procgen", procgen.__file__); print("gym3", gym3.__file__); print("pyyaml", yaml.__version__); assert torch.cuda.is_available()'
"$PYTHON_BIN" -c "import yaml; c=yaml.safe_load(open('$CODE_ROOT/configs/ppo_resnet_shared.yaml')); assert c['algo']=='ppo'; assert c['algo_config']['optimizer']=='adam'; assert c['algo_config']['lr']==0.001; assert c['algo_config']['epochs']==4; assert c['algo_config']['minibatches']==8; assert c['algo_config']['use_kl_adaptive_lr'] is False; assert c['env_config']['num_envs']==16; assert c['env_config']['nsteps']==256; assert c['env_config']['timesteps_per_proc_easy']==6000000; assert c['nets_config']['type']=='resnet'; print('formal config assertions: PASS')"

test "$(sha256sum "$CODE_ROOT/train_shared.py" | awk '{print $1}')" = "1d20658b154022450b8598949f693b3c04a9bd34eb22ad2f002d59f9573b74d1"
test "$(sha256sum "$CODE_ROOT/configs/ppo_resnet_shared.yaml" | awk '{print $1}')" = "fdf1538ef199a222ea2caafe9264c5db00319a6f1882d7d86b04506522601807"

nvidia-smi -i "$GPU_ID" --query-gpu=index,name,driver_version,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

(
  while true; do
    date +%s
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    sleep 2
  done
) > "$SMOKE_ROOT/gpu.csv" 2>&1 &
MON_PID=$!

set +e
(
  cd "$CODE_ROOT"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u train_shared.py \
    --config ppo_resnet_shared_smoke.yaml \
    --env_name bigfish-easy-0-10 \
    --seed 999 \
    --device 0
) > "$SMOKE_ROOT/stdout.log" 2> "$SMOKE_ROOT/stderr.log"
RC=$?
set -e

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
echo "$RC" > "$SMOKE_ROOT/returncode"

if [[ "$RC" -eq 0 ]]; then
  echo PASS > "$SMOKE_ROOT/status"
else
  echo FAIL > "$SMOKE_ROOT/status"
fi
exit "$RC"
