#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${project_root}/supervisor" "${project_root}/status"
cd "${project_root}"

preflight_log="${project_root}/status/ACTOR_PREFLIGHT.log"
"${project_root}/.venv/bin/python" -m unittest -q \
  tests.test_phasic_ef_ggn > "${preflight_log}" 2>&1
"${project_root}/.venv/bin/python" \
  "${project_root}/validate_actor_ablation.py" >> "${preflight_log}" 2>&1
printf 'PASS end_utc=%s tests=14 config_assertions=pass\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${project_root}/status/ACTOR_PREFLIGHT.status"

sha256sum \
  "${project_root}/train_phasic_ef_ggn.py" \
  "${project_root}/actor_ef_ablation.py" \
  "${project_root}/phasic_ef_ggn.py" \
  "${project_root}/ppg_auxiliary.py" \
  "${project_root}/tests/test_phasic_ef_ggn.py" \
  "${project_root}/validate_actor_ablation.py" \
  "${project_root}/launchers/run_variant_once.sh" \
  "${project_root}/launchers/run_variant_serial_continue.sh" \
  "${project_root}/launchers/watch_actor_formal.sh" \
  "${project_root}/launchers/launch_when_gpus_free.sh" \
  "${project_root}/launchers/launch_actor_formal_direct.sh" \
  "${project_root}/configs/actor_G_entropy_official_ppg_formal.yaml" \
  "${project_root}/configs/actor_H_policykl_official_ppg_formal.yaml" \
  "${project_root}/configs/actor_I_epoch1_official_ppg_formal.yaml" \
  "${project_root}/configs/actor_J_combined_official_ppg_formal.yaml" \
  > "${project_root}/status/PROVENANCE_ACTOR_FORMAL.sha256"

printf 'RUNNING_STAGGERED start_utc=%s target_transitions=6000000 seeds=0,1,2 critic=official_ppg_adam actor_ablation=entropy,policy_kl,epoch1,combined\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${project_root}/status/ACTOR_FORMAL_SWEEP.status"
: > "${project_root}/status/PIDS_ACTOR_FORMAL.tsv"

launch_one() {
  local variant="$1" config="$2" physical_gpu="$3"
  local log="${project_root}/supervisor/${variant}.log"
  local handoff_status="${project_root}/status/${variant}_HANDOFF.status"
  printf 'WAITING_FOR_GPU start_utc=%s physical_gpu=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
    > "${handoff_status}"
  while true; do
    active_pids="$(nvidia-smi -i "${physical_gpu}" \
      --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
      | sed '/^[[:space:]]*$/d')"
    if [[ -z "${active_pids}" ]]; then
      break
    fi
    sleep 10
  done
  nohup bash "${project_root}/launchers/run_variant_serial_continue.sh" \
    "${variant}" "${config}" "${physical_gpu}" > "${log}" 2>&1 &
  supervisor_pid="$!"
  printf '%s\t%s\t%s\t%s\n' \
    "${variant}" "${physical_gpu}" "${supervisor_pid}" "${log}" \
    >> "${project_root}/status/PIDS_ACTOR_FORMAL.tsv"
  printf 'LAUNCHED end_utc=%s physical_gpu=%s supervisor_pid=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${physical_gpu}" \
    "${supervisor_pid}" > "${handoff_status}"
}

launch_one ACTOR_G actor_G_entropy_official_ppg_formal.yaml 4 &
wait_g="$!"
launch_one ACTOR_H actor_H_policykl_official_ppg_formal.yaml 5 &
wait_h="$!"
launch_one ACTOR_I actor_I_epoch1_official_ppg_formal.yaml 6 &
wait_i="$!"
launch_one ACTOR_J actor_J_combined_official_ppg_formal.yaml 7 &
wait_j="$!"
wait "${wait_g}" "${wait_h}" "${wait_i}" "${wait_j}"

nohup bash "${project_root}/launchers/watch_actor_formal.sh" \
  > "${project_root}/supervisor/ACTOR_FORMAL_WATCHER.log" 2>&1 &
printf 'WATCHER\tNA\t%s\t%s\n' "$!" \
  "${project_root}/supervisor/ACTOR_FORMAL_WATCHER.log" \
  >> "${project_root}/status/PIDS_ACTOR_FORMAL.tsv"

cat "${project_root}/status/PIDS_ACTOR_FORMAL.tsv"
