#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 GPU COMMAND [ARG ...]" >&2
  exit 2
fi

gpu="$1"
shift
case "${gpu}" in
  0|1) ;;
  *) echo "GPU must be 0 or 1" >&2; exit 2 ;;
esac

stack_root="${HOME}/rlstack5060"
workspace="${PROCGEN_WORKSPACE:-${stack_root}/workspaces/procgen}"

exec docker run --rm \
  --gpus "device=${gpu}" \
  --ipc=host \
  --shm-size=12g \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e OMP_NUM_THREADS=4 \
  -e MKL_NUM_THREADS=4 \
  -v "${workspace}:/workspace/procgen" \
  -w /workspace/procgen \
  rlstack5060/procgen:cu128 "$@"

