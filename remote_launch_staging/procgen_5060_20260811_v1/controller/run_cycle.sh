#!/usr/bin/env bash
set -euo pipefail

controller_root=/scratch/h99859yz/codex_three_domain_controller_20260807
mkdir -p "$controller_root"/{actions,logs,state,prompts,bin}
exec 9>"$controller_root/state/cycle.lock"
if ! flock -n 9; then
  exit 0
fi

cycle_ts=$(date -u +%Y%m%dT%H%M%SZ)
snapshot_tmp="$controller_root/state/live_snapshot.$cycle_ts.tmp"
snapshot="$controller_root/state/live_snapshot.txt"

remote_snapshot() {
  local target=$1
  timeout 35 ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$target" '
    hostname
    date -u +%Y-%m-%dT%H:%M:%SZ
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1
    printf "%s\n" "--owned-processes--"
    ps -u "$(id -un)" -o pid,etimes,stat,args --sort=-etimes | grep -E "[p]ython|[b]ash.*run_|[s]creen|[t]mux" | tail -60
  ' || true
  printf '%s\n' '--recent-metrics--'
  timeout 45 ssh -T -o BatchMode=yes -o ConnectTimeout=15 "$target" '$HOME/.local/bin/codex_metric_snapshot' || true
}

{
  printf 'snapshot_utc=%s\n' "$cycle_ts"
  printf '\n[CSF3_IDENTITY]\n'
  hostname
  id -un
  printf '\n[CSF3_QUEUE]\n'
  timeout 30 squeue -u "$USER" -o '%.18i %.10P %.28j %.9T %.10M %.19S %.24R' || true
  printf '\n[CSF3_ACCOUNTING_48H]\n'
  timeout 30 sacct -u "$USER" -S "$(date -d '2 days ago' +%F)" -X -o JobID,JobName%28,State,Elapsed,Start,End -n | tail -80 || true
  printf '\n[CSF3_USER_PROCESSES]\n'
  ps -u "$USER" -o pid,etimes,stat,args --sort=-etimes | grep -E '[c]odex|[p]ython|[s]batch|[s]run|[s]creen|[t]mux' | tail -80 || true
  printf '\n[CSF3_RECENT_METRICS]\n'
  timeout 45 "$controller_root/bin/codex_metric_snapshot" || true
  printf '\n[BEDE]\n'
  timeout 40 ssh -T -o BatchMode=yes -o ConnectTimeout=15 bede '
    hostname
    id -un
    squeue -u "$USER" -o "%.18i %.10P %.28j %.9T %.10M %.19S %.24R"
    sacct -u "$USER" -S "$(date -d "2 days ago" +%F)" -X -o JobID,JobName%28,State,Elapsed,Start,End -n | tail -60
  ' || true
  for target in procgen-3090 ws4090-92 ws4090-76; do
    printf '\n[%s]\n' "$target"
    remote_snapshot "$target"
  done
  printf '\n[procgen-5060]\n'
  timeout 35 ssh -T -o BatchMode=yes -o ConnectTimeout=15 \
    -i "$HOME/.ssh/id_ed25519_codex_csf3" \
    -p 60023 zzz@47.114.81.212 || true
} >"$snapshot_tmp"
mv "$snapshot_tmp" "$snapshot"

: >"$controller_root/actions/next.tsv"
if timeout 900 env PATH="$controller_root/bin:$PATH" codex -m gpt-5.6-sol -c 'model_reasoning_effort="medium"' -a never exec \
  --sandbox workspace-write --skip-git-repo-check -C "$controller_root" \
  -o "$controller_root/logs/final-$cycle_ts.txt" - \
  <"$controller_root/prompts/controller.md" \
  >"$controller_root/logs/codex-$cycle_ts.log" 2>&1; then
  cp "$controller_root/actions/next.tsv" "$controller_root/logs/actions-$cycle_ts.tsv"
  printf '%s\tPLAN_COMPLETE\t%s\n' "$cycle_ts" "$controller_root/logs/actions-$cycle_ts.tsv" >>"$controller_root/state/history.tsv"
  bash "$controller_root/apply_actions.sh" "$cycle_ts" || true
else
  rc=$?
  printf '%s\tPLAN_FAILED_%s\n' "$cycle_ts" "$rc" >>"$controller_root/state/history.tsv"
fi
