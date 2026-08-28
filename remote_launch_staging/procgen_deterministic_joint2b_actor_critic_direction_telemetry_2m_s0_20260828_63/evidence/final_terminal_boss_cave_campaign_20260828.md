# Task63 final terminal evidence

All four Task63 jobs are scheduler-authoritatively terminal. BigFish and
CoinRun evidence is preserved separately in
`partial_terminal_bigfish_coinrun_20260828.md`.

| Environment | Job | Scheduler | Elapsed | Node | Root/rc | Endpoint reward | Paper reward | Frozen aggregate |
|---|---:|---|---:|---|---|---:|---:|---|
| BigFish | 1078181 | COMPLETED/0:0 | 02:44:51 | gpu025 | PASS/0 | 5.08 | 9.28 | PASS |
| BossFight | 1078182 | COMPLETED/0:0 | 02:51:51 | gpu025 | PASS/0 | .04 | 2.92 | FAIL record5952 |
| CaveFlyer | 1078183 | COMPLETED/0:0 | 02:49:28 | gpu006 | PASS/0 | 0 | 4.45 | PASS |
| CoinRun | 1078184 | COMPLETED/0:0 | 02:45:51 | gpu007 | PASS/0 | 10.00 | 3.70 | FAIL record3808 |

Every root has exact transition 2,007,040, 49 progress rows, 15,680 complete
records, a regular non-symlink remote checkpoint of 3,766,013 bytes mode 664,
and zero focused hard-error matches. No model bytes or model content hashes are
included.

BossFight model-free hashes: progress
`ecd45671475fb42385fb261d50f1d943f21977e1e55ccb3201bef4532ec7109e`,
trace `b412fcd8f7890ac7d4239da6d9cd0591e1c8cb0e5137d3ad0a7f1edf47bf8241`,
stdout `7832f37f4218fd8aaae0f7ed006daa642f73d651ae9f4038c663216f4dbefa68`,
stderr `bf04de874e19842406b05ad6d67636d7cef1b00a8619998503e535751e620c14`.

CaveFlyer model-free hashes: progress
`8712a7f0175bb4407b7087e6976bbdd6cf90cf83641596e1353eb109c9469ec3`,
trace `cfac803913553376737d82fbedfbb7ec8a1abe3be384d5ef823d49256bbb3650`,
stdout `0e23abba6def65e51b928676d7af0887dbb3d9b4d5e59f55463c88caf1bf0f71`,
stderr `a1729e5da186437974225c3a1b5bbeae5006c90cb509c153c614d71194d1f736`.

CaveFlyer frozen aggregation passes all records. Overall medians are full
actor norm/projection `.6203239262/.7368913889`, shared actor norm/projection
`.6319409311/.7556186616`, actor metric norm/energy
`.6402442455/.7600309253`, full cosine `-.0306813885`, cancellation `.726885438`
and clip rate `.6322704082`. Early/middle/late full actor norm medians are
`.5619818568/.6049734950/.7074852586`; projection medians are
`.6266603172/.7113480568/.8663445115`.

BossFight frozen aggregation stops at record5952 because the policy direction
is identically zero after categorical saturation: transition765,952,
entropy3.390e-39, actor raw scale0, actor Fisher quadratic0, policy total norm0
and policy projection sum0. Full/shared/value projection sums remain1;
Cholesky0, finite1, relative residual1.866e-12, RHS reconstruction0, alpha
reconstruction4.173e-15 and direction reconstruction1.914e-11. This is an
instrumentation validator edge-case failure, not a training/solver/GPU failure.
No repair or rerun occurred.

Final classification:
`TERMINAL_TRAINING_COMPLETE_TELEMETRY_AGGREGATION_PARTIAL_FAILURE`. BigFish
and CaveFlyer have valid frozen aggregates; BossFight and CoinRun have clean
training terminals but invalid frozen aggregate conclusions due zero/near-zero
policy subspaces.
