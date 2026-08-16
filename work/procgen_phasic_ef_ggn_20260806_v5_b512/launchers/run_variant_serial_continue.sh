#!/usr/bin/env bash
set -u

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 VARIANT CONFIG_BASENAME PHYSICAL_GPU" >&2
  exit 2
fi

variant="$1"
config="$2"
physical_gpu="$3"
project_root="$(cd "$(dirname "$0")/.." && pwd)"
aggregate_status="${project_root}/status/${variant}.status"
failures=0

printf 'RUNNING start_utc=%s physical_gpu=%s seeds=0,1,2\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
  > "${aggregate_status}"

for seed in 0 1 2; do
  if ! bash "${project_root}/launchers/run_variant_once.sh" \
    "${variant}" "${config}" "${physical_gpu}" "${seed}"; then
    failures=$((failures + 1))
  fi
done

if [[ "${failures}" -eq 0 ]]; then
  printf 'COMPLETED end_utc=%s physical_gpu=%s failures=0\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
    > "${aggregate_status}"
  exit 0
fi

printf 'COMPLETED_WITH_FAILURES end_utc=%s physical_gpu=%s failures=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" "${failures}" \
  > "${aggregate_status}"
exit 1
