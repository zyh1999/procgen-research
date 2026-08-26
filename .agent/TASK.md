# Task-ID: PROCGEN-FULL-SHARED-JOINT2B-ACTOR-SCALE-FLOOR-6M-S0-20260826-46

Status: READY

Implement exactly one scientific delta from Task45 while preserving strict
full-shared Joint-2B: use
`s_pi_eff=max(s_pi_raw, 0.01*s_v_raw)` for actor row/RHS normalization and keep
critic normalization at positive finite `s_v_raw`. Preserve all rows, RHS,
natural cross blocks, reconstruction, relative damping `.5`, network,
rollout/GAE/PopArt/optimizer/schedule/evaluation and checkpoint semantics.

Run only exact parent-to-child diff/SHA, syntax/import, normalized command,
fresh scheduler/capacity/duplicate/root and non-overlap checks. Run no micro or
negative tests, oracle/Jacobian reference, production preflight or audit chain.
Then submit exactly one fresh seed0 6M job for BigFish, BossFight, CaveFlyer and
CoinRun, with no retry/requeue/resubmit/sweep/extra seed. Do not cancel, modify
or overwrite any Task45 job/root/ledger.

Monitor Task45 and Task46 within the existing sole Procgen automation. Compare
only exact matching Paper seed0 rows at first common >=2M, first common >=4M
and 5,980,160; cancel only the individual cell for exact Target/Paper <.60.
