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

## Submitted race and winner

- gpuL scientific array: `19203054`.
- gpuH bundles: BigFish `19203172`, BossFight `19203173`, CaveFlyer
  `19203174`, CoinRun `19203175`.
- BigFish/BossFight/CaveFlyer won the race by passing their in-allocation
  H200 checks and creating 24 scientific child roots/markers before any gpuL
  target started. Each H200 reports `150,111,977,472` bytes; the eight-child
  reservation was `124,000,000,000` bytes with `26,111,977,472` bytes
  headroom. FP64 residuals are `6.767e-16`, `6.964e-16`, and `6.680e-16`.
- gpuL loser `19203054` was cancelled only after this scientific-start proof.
  Every array task is `CANCELLED by 778916`, `Start=None`, elapsed `00:00:00`,
  node `None assigned`; `runs_gpul` never existed. Classification:
  `cancelled-race-loser-unstarted`.
- BigFish `19203172`, BossFight `19203173`, and CaveFlyer `19203174` each
  completed `0:0` with bundle PASS/rc0 and eight child PASS/rc0 artifact sets.
  Every completed child has 7,872 trace rows ending at 1,007,616 transitions,
  a progress row at 983,040, and a 3,766,013-byte checkpoint.
- CoinRun `19203175` began on node821 and launched all eight children. The user
  later explicitly ordered a scientific-futility early stop after the other
  three seed0 ratios had made the 3-of-4 gate mathematically impossible.
  Scheduler authority records parent `CANCELLED by 778916`, elapsed `00:58:17`,
  node821, and batch `CANCELLED/0:15`. Child markers remain stale `RUNNING`, rc
  and copied terminal artifacts are absent, and immutable source logs end at
  progress 573,440 with traces at 589,824--593,920. Classification:
  `user-authorized scientific-futility early stop`, not PASS and not an
  infrastructure failure. No CoinRun artifact was deleted or overwritten.
