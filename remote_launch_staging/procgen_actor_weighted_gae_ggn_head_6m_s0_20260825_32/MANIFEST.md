# Frozen Task32 implementation manifest

- Task-ID: `PROCGEN-ACTOR-WEIGHTED-GAE-GGN-HEAD-6M-S0-20260825-32`
- Method: `DET_ACTOR_WEIGHTED_GAE_GGN_HEAD_V1`
- Assignment: `adbc5396a615bb8855384d1edca4e0ebe76ee432`
- Trainer SHA256: `5d070556a7215b15a5715a4d6100eb1c1b440e5f3283949e94e3ee97ae89aa05`
- Config SHA256: `4458567bc61bdb85360d55967c65700992fe60cd812819fa1e77cc8638c202af`
- Functional preflight SHA256: `fe5908afb6c0e829212eb8a19cda222deffd18139a5885cf9cbaa5f18cbacb6e`
- Scientific launcher SHA256: `e7755fd50df57d3aa719638ba32412082c611bff018dc530792c48f79bac5714`
- Preflight launcher SHA256: `a7784d48d40e453a56ac800a300fb49ca2934d747d0a7cec656f3a77efe437cf`
- Stage monitor SHA256: `344c0654c6c785f87f0a0ff625372fac6b7b15a28c8cfeff8307d1fa9350b84f`
- Static audit SHA256: `b6ee037ae8eda3de1759d4aa18b4dc16ec738d508bdbc44e1927978a2fc777da`

The only scientific delta from strict Paper/Hybrid control is the
actor-weighted exact-GAE deterministic GGN proposal for the 257 critic-only
value-head parameters. Paper actor, sampled shared-critic path, update order,
adaptive KL, momentum/history, global clipping, rollout, PopArt, and evaluation
semantics remain in the copied production path.

Task14--31 origin/closure/observer mechanisms are absent and are not a
scientific gate. Paper proposal norm/RHS/inverse matching is absent.
