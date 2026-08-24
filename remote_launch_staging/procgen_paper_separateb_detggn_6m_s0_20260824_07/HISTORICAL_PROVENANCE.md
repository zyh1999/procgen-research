# Historical no-cross provenance gate

Conclusion: `DISTINCT_FORMULA_PASS`. These historical rewards are negative
provenance only, never Paper baselines.

| Evidence | Exact trainer SHA256 | Historical formula | Difference from this candidate |
|---|---|---|---|
| CSF3 `18669377`, deterministic block trace | `1881bf7c3fe3f8d29ded23e25976810ab9127d9bc125d9c89332aa39c1ab61dc` | one 512-row system with `K_actor + 4 J_v J_v^T/B`, expected cross zero | not two independent solves; lambda/objective and actor path differ |
| CSF3 `18669454`, expected-relative | `c976c0e563eb3aedb2d306c450d60b44af0c595d0f4a499cf32c65bcec9933d3` | one 512-row analytic expected-Gaussian system; critic diagonal/off-diagonal expectation, expected cross zero; relative damping | not an independent critic B solve; damping differs |
| CSF3 `18669615`, actor-relative expected | `c976c0e563eb3aedb2d306c450d60b44af0c595d0f4a499cf32c65bcec9933d3` | same expected trainer with actor-relative/critic-floor configuration | no floor exists here; actor system and schedule differ |
| Bede `1072337`, `1072344/46/49/50` | `0514703d9fb6ca17cc68febabb012defb279ab5a54f57cf95365422164848934` | one 512-row expected-Gaussian `HpiHpiT/B + 4 J_vJ_vT/B` system, zero expected cross | not separate actor and critic systems; critic lambda/RHS/damping differ |

The full CSF3 hashes above were recomputed from the referenced files under
`/scratch/h99859yz/procgen_joint2b_causal_ablation_20260812_v1/code`. The Bede
full hash is frozen in the exact launch artifacts
`bede_rat_expectedcritic_dualdamping_smoke100k.sbatch` and
`bede_rat_expectedcritic_dualdamping_gate500k.sbatch`; current Bede login
authentication was unavailable, so no prefix was promoted into a new claim.

| Field | Historical expected/no-cross attempts | `PAPER_MATCHED_SEPARATE_B_DET_GGN_V1` |
|---|---|---|
| critic curvature | analytic expected Gaussian kernel, commonly `4 J_vJ_vT/B`, added to actor kernel | `lambda=.1`, rows `sqrt(.1) J_v`, independent `rows rowsT/B` |
| critic RHS | expected-score formulation / reference weighted-MSE update | residual divided by `sqrt(.1)`, objective coefficient 1 |
| damping | dual actor/critic fixed or relative floors (`.03/.5` in Bede expected) | Paper actor damping `.5`; separate critic damping `.5`; no relative floor |
| actor matrix/path | altered expected/block-trace shared B system | literal Paper sampled-score actor B system and inverse |
| adaptive KL | rollout-level, initial LR `.004` | every minibatch, thresholds `.005/.04`, initial/maximum LR `.5` |
| momentum/history | momentum 0, history disabled | SGD momentum `1e-6`, original `rhs-H@buffer` on each independent system |
| rows/cross | one B system, expected cross zero | actor B plus a second critic B; never 2B; cross blocks absent by construction |

Interpretation constraints: a clean FP64 residual does not establish update
usefulness. Joint-2B V1 ratios `.2583/0/.2188` with LR `.0001` show a
geometry/calibration failure. Low-Fisher was `GUARD_NOT_HELPFUL` and is not
reintroduced. A 100k/250k/1M structural PASS cannot establish competitiveness.
ACTOR_J BossFight ratio `.5465` at 4,096,000 supports exact checks at >=2M and
>=4M. Infrastructure failures remain separate from algorithm evidence.
