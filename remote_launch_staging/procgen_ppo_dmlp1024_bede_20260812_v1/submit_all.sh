#!/usr/bin/env bash
set -euo pipefail

ROOT=/nobackup/projects/bdman37/yihe/procgen_ppo_dmlp1024_bede_20260812_v1
CAMPAIGN="$ROOT/formal_4env_x3seed_6m_20260812_v1"
SBATCH="$ROOT/formal_bundle3.sbatch"
SUBMITTED="$CAMPAIGN/SUBMITTED.tsv"

for tag in bigfish bossfight caveflyer coinrun; do
  [[ ! -e "$CAMPAIGN/$tag" ]] || {
    echo "refusing duplicate existing root $CAMPAIGN/$tag" >&2
    exit 6
  }
done

printf 'submit_utc\tjob_id\tjob_name\tenvironment\ttag\tchildren\tseeds\n' > "$SUBMITTED"
while IFS='|' read -r tag env job_name; do
  job_id=$(sbatch --parsable \
    --job-name="$job_name" \
    --export=ALL,ENV_NAME="$env",ENV_TAG="$tag" \
    "$SBATCH")
  printf '%s\t%s\t%s\t%s\t%s\t3\t0,1,2\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$job_id" "$job_name" "$env" "$tag" >> "$SUBMITTED"
done <<'MATRIX'
bigfish|bigfish-easy-0-10|pg-pd-bf
bossfight|bossfight-easy-0-10|pg-pd-bo
caveflyer|caveflyer-easy-0-10|pg-pd-cf
coinrun|coinrun-easy-0-10|pg-pd-cr
MATRIX

cat "$SUBMITTED"
