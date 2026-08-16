#!/usr/bin/env bash
set -euo pipefail

campaign_root="/root/procgen_ejk_multienv_20260811_v1"
e_source="/root/procgen_matched_ppg_20260806_E_v2"
j_source="/root/procgen_ef_actor_ablation_20260806_v1"
k_source="/root/procgen_ef_adaptivekl_exactrat_20260806_v1"

if [[ -e "${campaign_root}" ]]; then
  echo "REFUSE: campaign root already exists: ${campaign_root}" >&2
  exit 3
fi

for source_root in "${e_source}" "${j_source}" "${k_source}"; do
  test -f "${source_root}/train_phasic_ef_ggn.py"
  test -x "${source_root}/.venv/bin/python"
done

mkdir -p \
  "${campaign_root}/E" \
  "${campaign_root}/J" \
  "${campaign_root}/K" \
  "${campaign_root}/launchers" \
  "${campaign_root}/run_logs" \
  "${campaign_root}/status/tasks" \
  "${campaign_root}/status/workers" \
  "${campaign_root}/supervisor"

copy_source() {
  local source_root="$1"
  local destination_root="$2"
  tar -C "${source_root}" \
    --exclude='./.venv' \
    --exclude='./logs' \
    --exclude='./run_logs' \
    --exclude='./status' \
    --exclude='./supervisor' \
    -cf - . | tar -C "${destination_root}" -xf -
  mkdir -p "${destination_root}/logs"
}

copy_source "${e_source}" "${campaign_root}/E"
copy_source "${j_source}" "${campaign_root}/J"
copy_source "${k_source}" "${campaign_root}/K"

make_config() {
  local method="$1"
  local source_config="$2"
  local output_config="$3"
  local variant_name="$4"
  local env_name="$5"
  cp "${source_config}" "${output_config}"
  sed -i \
    -e "s/^  variant_name:.*/  variant_name: ${variant_name}/" \
    -e "s/^  env_name:.*/  env_name: ${env_name}-easy-0-10/" \
    "${output_config}"
  grep -Fx "  variant_name: ${variant_name}" "${output_config}" >/dev/null
  grep -Fx "  env_name: ${env_name}-easy-0-10" "${output_config}" >/dev/null
  printf '%s\t%s\t%s\n' "${method}" "${env_name}" "${output_config}"
}

for env_name in coinrun bossfight caveflyer; do
  env_upper="${env_name^^}"
  make_config E \
    "${campaign_root}/E/configs/matched_ppg_E_v2_bigfish_formal.yaml" \
    "${campaign_root}/E/configs/e_v2_multienv_${env_name}.yaml" \
    "E_V2_MULTIENV_${env_upper}" "${env_name}"
  make_config J \
    "${campaign_root}/J/configs/actor_J_combined_official_ppg_formal.yaml" \
    "${campaign_root}/J/configs/actor_J_multienv_${env_name}.yaml" \
    "ACTOR_J_MULTIENV_${env_upper}" "${env_name}"
  make_config K \
    "${campaign_root}/K/configs/actor_K_exactrat_adaptivekl_official_ppg_formal.yaml" \
    "${campaign_root}/K/configs/actor_K_multienv_${env_name}.yaml" \
    "ACTOR_K_MULTIENV_${env_upper}" "${env_name}"
done > "${campaign_root}/status/GENERATED_CONFIGS.tsv"

tasks_path="${campaign_root}/status/TASKS.tsv"
: > "${tasks_path}"
for seed in 0 1 2; do
  for method in J E K; do
    for env_name in coinrun bossfight caveflyer; do
      case "${method}" in
        E) config="e_v2_multienv_${env_name}.yaml" ;;
        J) config="actor_J_multienv_${env_name}.yaml" ;;
        K) config="actor_K_multienv_${env_name}.yaml" ;;
      esac
      printf '%s_%s_s%s\t%s\t%s\t%s\t%s\n' \
        "${method}" "${env_name^^}" "${seed}" \
        "${method}" "${env_name}" "${seed}" "${config}" \
        >> "${tasks_path}"
    done
  done
done

assert_same_except_identity() {
  local source_config="$1"
  local generated_config="$2"
  diff -u \
    <(sed -E '/^  variant_name:/d; /^  env_name:/d' "${source_config}") \
    <(sed -E '/^  variant_name:/d; /^  env_name:/d' "${generated_config}")
}

assertions_log="${campaign_root}/status/CONFIG_ASSERTIONS.log"
{
  echo "campaign_root=${campaign_root}"
  echo "task_count=$(wc -l < "${tasks_path}")"
  for env_name in coinrun bossfight caveflyer; do
    assert_same_except_identity \
      "${campaign_root}/E/configs/matched_ppg_E_v2_bigfish_formal.yaml" \
      "${campaign_root}/E/configs/e_v2_multienv_${env_name}.yaml"
    assert_same_except_identity \
      "${campaign_root}/J/configs/actor_J_combined_official_ppg_formal.yaml" \
      "${campaign_root}/J/configs/actor_J_multienv_${env_name}.yaml"
    assert_same_except_identity \
      "${campaign_root}/K/configs/actor_K_exactrat_adaptivekl_official_ppg_formal.yaml" \
      "${campaign_root}/K/configs/actor_K_multienv_${env_name}.yaml"
  done
  grep -Fx '  policy_updates_per_cycle: 32' \
    "${campaign_root}/E/configs/e_v2_multienv_coinrun.yaml"
  grep -Fx '  policy_updates_per_cycle: 16' \
    "${campaign_root}/J/configs/actor_J_multienv_coinrun.yaml"
  grep -Fx '  policy_updates_per_cycle: 16' \
    "${campaign_root}/K/configs/actor_K_multienv_coinrun.yaml"
  grep -Fx '  use_actor_entropy_natural_gradient: true' \
    "${campaign_root}/J/configs/actor_J_multienv_coinrun.yaml"
  grep -Fx '  use_actor_policy_fisher_clip: true' \
    "${campaign_root}/J/configs/actor_J_multienv_coinrun.yaml"
  grep -Fx '  lr: 0.5' \
    "${campaign_root}/K/configs/actor_K_multienv_coinrun.yaml"
  echo PASS
} > "${assertions_log}"

printf 'PREPARED utc=%s tasks=27 environments=coinrun,bossfight,caveflyer methods=E_v2,J,K seeds=0,1,2 transitions_per_seed=6000000 physical_gpus=4,5,6,7 architecture=resnet256 no_training_smoke=1\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${campaign_root}/status/CAMPAIGN.status"

echo "${campaign_root}"
