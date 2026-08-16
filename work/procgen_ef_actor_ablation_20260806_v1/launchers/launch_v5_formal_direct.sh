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

sha256sum \
  "${project_root}/train_phasic_ef_ggn.py" \
  "${project_root}/phasic_ef_ggn.py" \
  "${project_root}/ppg_auxiliary.py" \
  "${project_root}/tests/test_phasic_ef_ggn.py" \
  "${project_root}/launchers/run_variant_once.sh" \
  "${project_root}/launchers/run_variant_serial_continue.sh" \
  "${project_root}/launchers/watch_v5_formal.sh" \
  "${project_root}/launchers/launch_v5_formal_direct.sh" \
  "${project_root}/configs/v5_C_b512_true_ggn_no_clip_formal.yaml" \
  "${project_root}/configs/v5_D_b512_true_ggn_clip_mu1e2_formal.yaml" \
  "${project_root}/configs/v5_E_b512_true_ggn_clip_mu1e1_formal.yaml" \
  "${project_root}/configs/v5_F_b512_true_ggn_clip_kl5e4_formal.yaml" \
  > "${project_root}/status/PROVENANCE_V5_FORMAL.sha256"

printf 'RUNNING start_utc=%s target_transitions=6000000 seeds=0,1,2 aux_batch=512 fisher_batch=512\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${project_root}/status/V5_FORMAL_SWEEP.status"
: > "${project_root}/status/PIDS_V5_FORMAL.tsv"

launch_one() {
  local variant="$1" config="$2" physical_gpu="$3"
  local log="${project_root}/supervisor/${variant}.log"
  nohup bash "${project_root}/launchers/run_variant_serial_continue.sh" \
    "${variant}" "${config}" "${physical_gpu}" > "${log}" 2>&1 &
  printf '%s\t%s\t%s\t%s\n' \
    "${variant}" "${physical_gpu}" "$!" "${log}" \
    >> "${project_root}/status/PIDS_V5_FORMAL.tsv"
}

launch_one V5_FORMAL_C v5_C_b512_true_ggn_no_clip_formal.yaml 4
launch_one V5_FORMAL_D v5_D_b512_true_ggn_clip_mu1e2_formal.yaml 5
launch_one V5_FORMAL_E v5_E_b512_true_ggn_clip_mu1e1_formal.yaml 6
launch_one V5_FORMAL_F v5_F_b512_true_ggn_clip_kl5e4_formal.yaml 7

nohup bash "${project_root}/launchers/watch_v5_formal.sh" \
  > "${project_root}/supervisor/V5_FORMAL_WATCHER.log" 2>&1 &
printf 'WATCHER\tNA\t%s\t%s\n' "$!" \
  "${project_root}/supervisor/V5_FORMAL_WATCHER.log" \
  >> "${project_root}/status/PIDS_V5_FORMAL.tsv"

cat "${project_root}/status/PIDS_V5_FORMAL.tsv"
