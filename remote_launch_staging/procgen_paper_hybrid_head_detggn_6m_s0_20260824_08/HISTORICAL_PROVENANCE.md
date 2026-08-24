# Hybrid-head historical provenance gate

The candidate must remain distinct from every expected/no-cross/block-trace,
joint-2B, and separate-B deterministic implementation. Historical rewards are
negative provenance only, never Paper baselines.

| Evidence | Exact trainer SHA256 | Formula distinction |
|---|---|---|
| CSF3 `18669377` block trace | `1881bf7c3fe3f8d29ded23e25976810ab9127d9bc125d9c89332aa39c1ab61dc` | one shared actor-plus-critic B system with expected-zero cross; deterministic curvature reaches shared trunk |
| CSF3 `18669454/18669615` expected-relative | `c976c0e563eb3aedb2d306c450d60b44af0c595d0f4a499cf32c65bcec9933d3` | analytic expected-Gaussian shared system and relative damping |
| Bede `1072337`, `1072344/46/49/50` | `0514703d9fb6ca17cc68febabb012defb279ab5a54f57cf95365422164848934` | expected actor-plus-critic B system, dual damping, LR `.004`, rollout KL, momentum0/history off |
| joint-2B V1 | `41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a` | deterministic critic rows include shared parameters and are coupled in joint 2B geometry |
| separate-B V1 | `b0dad110c36dbab4c601aa9128ba51eb437bfc6a3e9cadf87be8fd2172f3729a` | deterministic critic B is separate but still includes every shared trunk parameter |

`PAPER_HYBRID_SHARED_PAPER_HEAD_DETGGN_V1` keeps exact Paper sampled critic
direction on the shared trunk and applies deterministic residual GGN only to
the 257 trainable PopArt value-head weight/bias parameters. Policy-head
parameters receive no critic direction. PopArt running mean/variance/debiasing
state remains nontrainable Paper state and is not a curvature parameter.

Conclusion is valid only after source/hash audit and actual-network Jacobian
partition regression return PASS.
