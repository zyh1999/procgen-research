# Explicit user override: gpuL versus gpuH scheduling race

This matrix expansion was explicitly directed by the user and was not authored
by the ChatGPT Planner. It overrides the earlier no-extra-cell/no-duplicate
restriction only for this scheduling race.

- gpuL side: the already frozen four-environment seed0 1M array path, enabled
  only after corrected preflight `19202370` passed.
- gpuH side: four independent sbatch bundles, one environment each. Every
  bundle requests one H200 and at most eight CPUs, runs an in-allocation
  non-scientific aggregate compatibility preflight, then launches seeds 0--7
  concurrently into 32 unique roots total.
- Winner: the side that first begins actual trainer processes/scientific child
  execution. A preflight start does not win. Never cancel a running scientific
  job. Cancel only opposite-side jobs still unstarted, recording scheduler
  Start/Elapsed/Node/root evidence as `cancelled-race-loser-unstarted`.
- No Jupyter, Paper rerun, second method, algorithm/config/trainer change,
  early stop, automatic retry, or unrelated gpuH/Isaac mutation is authorized.

The first gpuH submission command produced no job ID because gpuH enforces at
most 193,392 MB host memory per requested H200 and the initial scheduler-only
request was 256G. The corrected request is 188G (192,512 MiB), below that cap.
No gpuH allocation, root, process, or scientific attempt resulted from the
rejected command.

Frozen scientific hashes remain trainer
`41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a`
and config
`69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d`.
