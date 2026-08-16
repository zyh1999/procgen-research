#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" || -z "${GPU_ID:-}" || -z "${SMOKE_ROOT:-}" ]]; then
  echo "Required: PYTHON_BIN=/path/to/python GPU_ID=0 SMOKE_ROOT=/path/to/new/smoke $0" >&2
  exit 2
fi

BUNDLE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE_ROOT="$BUNDLE_ROOT/source/trust-region-main"
TRAINER="train_shared_procgen_maincfg_pklbranch.py"
CONFIG="adv_resnet_shared_procgen_maincfg_pklbranch.yaml"

if [[ -e "$SMOKE_ROOT" ]]; then
  echo "Refusing to reuse existing SMOKE_ROOT: $SMOKE_ROOT" >&2
  exit 3
fi
mkdir -p "$SMOKE_ROOT"

test "$(sha256sum "$CODE_ROOT/$TRAINER" | awk '{print $1}')" = "f4cfcd3a5dd9ea84e9d7533a5f17c2d897db545a49d352850df89bdc69142369"
test "$(sha256sum "$CODE_ROOT/configs/$CONFIG" | awk '{print $1}')" = "476b210d9da6e1dc973cf293d812a5c1e2f3c6f20654736a9687e397131da1ca"
"$PYTHON_BIN" -m py_compile "$CODE_ROOT/$TRAINER"
"$PYTHON_BIN" -c "import yaml; c=yaml.safe_load(open('$CODE_ROOT/configs/$CONFIG')); a=c['algo_config']; e=c['env_config']; assert c['algo']=='adv'; assert a['optimizer']=='sgd'; assert a['lr']==0.5; assert a['use_kl_adaptive_lr'] is True; assert a['use_procgen_kl_thresholds'] is True; assert a['cg_damping']==0.5; assert a['is_karzmarz'] is False; assert a['epochs']==4; assert a['minibatches']==8; assert e['num_envs']==16; assert e['nsteps']==256; assert e['timesteps_per_proc_easy']==6000000; print('exact RAT config assertions: PASS')"

(
  while true; do
    date --iso-8601=seconds
    nvidia-smi -i "$GPU_ID" --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits
    sleep 2
  done
) > "$SMOKE_ROOT/gpu.csv" 2>&1 &
MON_PID=$!

set +e
(
  cd "$CODE_ROOT"
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u "$TRAINER" \
    --config "$CONFIG" \
    --env_name bigfish-easy-0-10 \
    --seed 999 \
    --device 0 \
    --timesteps_per_proc 40960
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
