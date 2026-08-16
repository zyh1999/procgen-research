#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${project_root}/supervisor" "${project_root}/status"

for physical_gpu in 4 5 6 7; do
  active_pids="$(nvidia-smi -i "${physical_gpu}" \
    --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | sed '/^[[:space:]]*$/d')"
  if [[ -n "${active_pids}" ]]; then
    echo "REFUSE: physical GPU${physical_gpu} has compute PID(s): ${active_pids}" >&2
    exit 3
  fi
done

printf 'RUNNING start_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${project_root}/status/V4_SMOKE_SWEEP.status"
: > "${project_root}/status/PIDS_V4_SMOKE.tsv"

launch_one() {
  local variant="$1" config="$2" physical_gpu="$3"
  local log="${project_root}/supervisor/${variant}.log"
  nohup bash "${project_root}/launchers/run_variant_once.sh" \
    "${variant}" "${config}" "${physical_gpu}" 0 > "${log}" 2>&1 &
  printf '%s\t%s\t%s\t%s\n' \
    "${variant}" "${physical_gpu}" "$!" "${log}" \
    >> "${project_root}/status/PIDS_V4_SMOKE.tsv"
}

launch_one V4_SMOKE_C v4_C_same_batch_true_ggn_no_clip_smoke.yaml 4
launch_one V4_SMOKE_D v4_D_same_batch_true_ggn_clip_mu1e2_smoke.yaml 5
launch_one V4_SMOKE_E v4_E_same_batch_true_ggn_clip_mu1e1_smoke.yaml 6
launch_one V4_SMOKE_F v4_F_same_batch_true_ggn_clip_kl5e4_smoke.yaml 7

cat "${project_root}/status/PIDS_V4_SMOKE.tsv"
