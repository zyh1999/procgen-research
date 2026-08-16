#!/usr/bin/env bash
set -euo pipefail

controller_root=/scratch/h99859yz/codex_three_domain_controller_20260807
cycle_ts=${1:?cycle timestamp required}
actions="$controller_root/actions/next.tsv"
approved="$controller_root/actions/approved_launchers.tsv"
execution_log="$controller_root/logs/execution-$cycle_ts.tsv"
evidence_file="$controller_root/state/EARLY_STOP_EVIDENCE.tsv"
ledger="$controller_root/state/EARLY_STOP_LEDGER.tsv"
confirm_dir="$controller_root/state/stop_confirmations"
touch "$approved" "$execution_log"
mkdir -p "$confirm_dir"
touch "$ledger"
declare -A cycle_remote_gpu_seen=()
declare -A cycle_remote_launcher_seen=()

reject() {
  printf '%s\tREJECTED\t%s\n' "$cycle_ts" "$1" >>"$execution_log"
}

evidence_ok() {
  local evidence_id=$1 expected_target=$2
  [[ -f "$evidence_file" ]] || return 1
  awk -F '\t' -v wanted="$evidence_id" -v target="$expected_target" '
    $1 == wanted && $3 == target {
      if ($14 == "HARD_FAILURE" && $13 ~ /^[0-9]+$/ && $13 >= 1) ok=1
      if ($7 ~ /^-?[0-9]+([.][0-9]+)?$/ && $8 ~ /^-?[0-9]+([.][0-9]+)?$/ && $13 ~ /^[0-9]+$/ && $7 > 0 && $8 <= 0.60 && $13 >= 2) ok=1
    }
    END { exit(ok ? 0 : 1) }
  ' "$evidence_file"
}

is_hard_failure() {
  local evidence_id=$1 expected_target=$2
  awk -F '\t' -v wanted="$evidence_id" -v target="$expected_target" '$1 == wanted && $3 == target && $14 == "HARD_FAILURE" { found=1 } END { exit(found ? 0 : 1) }' "$evidence_file"
}

confirmed_twice() {
  local evidence_id=$1 target=$2
  local safe marker
  safe=$(printf '%s' "$evidence_id" | tr -cd 'A-Za-z0-9._-')
  [[ -n "$safe" ]] || return 1
  marker="$confirm_dir/$safe"
  if [[ ! -f "$marker" ]]; then
    printf '%s\t%s\t%s\n' "$cycle_ts" "$target" "$evidence_id" >"$marker"
    printf '%s\tPENDING_SECOND_CONFIRMATION\t%s\t%s\n' "$cycle_ts" "$target" "$evidence_id" >>"$execution_log"
    return 1
  fi
  grep -Fq $'\t'"$target"$'\t'"$evidence_id" "$marker"
}

record_stop() {
  local target=$1 evidence_id=$2 outcome=$3
  printf '%s\t%s\t%s\t%s\n' "$cycle_ts" "$target" "$evidence_id" "$outcome" >>"$ledger"
}

while IFS=$'\t' read -r kind arg1 arg2 arg3 extra; do
  [[ -n "${kind:-}" ]] || continue
  case "$kind" in
    NOOP|NEEDS_USER)
      [[ -n "${arg1:-}" && -z "${arg2:-}${arg3:-}${extra:-}" ]] || { reject "malformed $kind"; continue; }
      printf '%s\t%s\t%s\n' "$cycle_ts" "$kind" "$arg1" >>"$execution_log"
      ;;
    SBATCH_CSF3)
      [[ -n "${arg1:-}" && -n "${arg2:-}" && -z "${arg3:-}${extra:-}" ]] || { reject 'malformed SBATCH_CSF3'; continue; }
      grep -Fqx $'SBATCH_CSF3\t'"$arg1" "$approved" || { reject "unapproved CSF3 launcher $arg1"; continue; }
      [[ "$arg1" = /scratch/h99859yz/*.sh && -f "$arg1" && -O "$arg1" ]] || { reject "unsafe CSF3 launcher $arg1"; continue; }
      job_id=$(sbatch --parsable "$arg1")
      printf '%s\tSTARTED_CSF3\t%s\t%s\n' "$cycle_ts" "$job_id" "$arg1" >>"$execution_log"
      ;;
    SBATCH_BEDE)
      [[ -n "${arg1:-}" && -n "${arg2:-}" && -z "${arg3:-}${extra:-}" ]] || { reject 'malformed SBATCH_BEDE'; continue; }
      grep -Fqx $'SBATCH_BEDE\t'"$arg1" "$approved" || { reject "unapproved Bede launcher $arg1"; continue; }
      job_id=$(timeout 40 ssh -n -T -o BatchMode=yes bede "test -f '$arg1' && sbatch --parsable '$arg1'")
      printf '%s\tSTARTED_BEDE\t%s\t%s\n' "$cycle_ts" "$job_id" "$arg1" >>"$execution_log"
      ;;
    START_REMOTE)
      [[ -n "${arg1:-}" && "${arg2:-}" =~ ^[0-7]$ && -n "${arg3:-}" && -n "${extra:-}" ]] || { reject 'malformed START_REMOTE'; continue; }
      case "$arg1" in procgen-3090|ws4090-92|ws4090-76) ;; ws4090-31) reject 'ws4090-31 entire host is quarantined'; continue ;; *) reject "unknown host $arg1"; continue ;; esac
      gpu_key="${arg1}:${arg2}"
      launcher_key="${arg1}:${arg3}"
      [[ -z "${cycle_remote_gpu_seen[$gpu_key]+x}" ]] || { reject "duplicate same-cycle GPU target $gpu_key"; continue; }
      [[ -z "${cycle_remote_launcher_seen[$launcher_key]+x}" ]] || { reject "duplicate same-cycle launcher $launcher_key"; continue; }
      cycle_remote_gpu_seen[$gpu_key]=1
      cycle_remote_launcher_seen[$launcher_key]=1
      grep -Fqx $'START_REMOTE\t'"$arg1"$'\t'"$arg2"$'\t'"$arg3" "$approved" || { reject "unapproved remote launcher $arg1 GPU$arg2 $arg3"; continue; }
      session="codex-${cycle_ts,,}-gpu${arg2}"
      if launch_result=$(timeout 45 ssh -n -T -o BatchMode=yes "$arg1" '$HOME/.local/bin/codex_gpu_launch' "$arg2" "$arg3" "$session" 2>&1); then
        printf '%s\tSTARTED_REMOTE\t%s\t%s\t%s\t%s\n' "$cycle_ts" "$arg1" "$arg2" "$session" "$arg3" >>"$execution_log"
      else
        reject "remote preflight refused $arg1 GPU$arg2 $arg3: $launch_result"
      fi
      ;;
    EARLY_STOP_CSF3)
      [[ "${arg1:-}" =~ ^[0-9]+(_[0-9]+)?$ && -n "${arg2:-}" && -n "${arg3:-}" && -z "${extra:-}" ]] || { reject 'malformed EARLY_STOP_CSF3'; continue; }
      evidence_ok "$arg2" "$arg1" || { reject "invalid early-stop evidence $arg2 for $arg1"; continue; }
      is_hard_failure "$arg2" "$arg1" || confirmed_twice "$arg2" "$arg1" || continue
      [[ -f "$controller_root/state/early_stop_enabled" ]] || { reject 'early stop is in shadow mode'; continue; }
      squeue -h -u "$USER" -j "$arg1" | grep -q . || { reject "CSF3 job not running or owned $arg1"; continue; }
      if scancel "$arg1"; then record_stop "$arg1" "$arg2" EARLY_STOPPED_FAILED; else record_stop "$arg1" "$arg2" STOP_FAILED; fi
      ;;
    EARLY_STOP_BEDE)
      [[ "${arg1:-}" =~ ^[0-9]+(_[0-9]+)?$ && -n "${arg2:-}" && -n "${arg3:-}" && -z "${extra:-}" ]] || { reject 'malformed EARLY_STOP_BEDE'; continue; }
      evidence_ok "$arg2" "$arg1" || { reject "invalid early-stop evidence $arg2 for Bede $arg1"; continue; }
      is_hard_failure "$arg2" "$arg1" || confirmed_twice "$arg2" "$arg1" || continue
      [[ -f "$controller_root/state/early_stop_enabled" ]] || { reject 'early stop is in shadow mode'; continue; }
      if timeout 40 ssh -n -T -o BatchMode=yes bede "squeue -h -u \"\$USER\" -j '$arg1' | grep -q . && scancel '$arg1'"; then record_stop "bede:$arg1" "$arg2" EARLY_STOPPED_FAILED; else record_stop "bede:$arg1" "$arg2" STOP_FAILED; fi
      ;;
    EARLY_STOP_REMOTE)
      [[ -n "${arg1:-}" && "${arg2:-}" =~ ^[0-9]+$ && "${arg3:-}" = /* && -n "${extra:-}" ]] || { reject 'malformed EARLY_STOP_REMOTE'; continue; }
      case "$arg1" in procgen-3090|procgen-5060|ws4090-92|ws4090-76|ws4090-31) ;; *) reject "unknown early-stop host $arg1"; continue ;; esac
      evidence_ok "$extra" "$arg1:$arg2" || { reject "invalid early-stop evidence $extra for $arg1:$arg2"; continue; }
      is_hard_failure "$extra" "$arg1:$arg2" || confirmed_twice "$extra" "$arg1:$arg2" || continue
      [[ -f "$controller_root/state/early_stop_enabled" ]] || { reject 'early stop is in shadow mode'; continue; }
      if timeout 45 ssh -n -T -o BatchMode=yes "$arg1" '$HOME/.local/bin/codex_run_stop' "$arg2" "$arg3"; then record_stop "$arg1:$arg2" "$extra" EARLY_STOPPED_FAILED; else record_stop "$arg1:$arg2" "$extra" STOP_FAILED; fi
      ;;
    *) reject "unknown action $kind" ;;
  esac
done <"$actions"
