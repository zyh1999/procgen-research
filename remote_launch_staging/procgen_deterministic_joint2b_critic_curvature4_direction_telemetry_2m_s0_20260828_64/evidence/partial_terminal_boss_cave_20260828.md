# Task64 BossFight/CaveFlyer partial terminal evidence

Bounded read-only refresh: 2026-08-28 09:57Z monitor turn. Scheduler state is authoritative. No cancellation, retry, requeue, resubmit, tuning, or model-content access occurred.

## Scheduler and artifacts

| Environment | Job | Scheduler | Exit | Elapsed | Node | Root | rc | Endpoint reward | Trace records | Checkpoint metadata | Hard-error scan |
|---|---:|---|---|---|---|---|---:|---:|---:|---|---:|
| BossFight | 19531930 | COMPLETED | 0:0 | 00:47:15 | node820 | PASS | 0 | 0.01 | 15,680 | regular non-symlink, 3,766,013 bytes, mode 0644 | 0 |
| CaveFlyer | 19531931 | COMPLETED | 0:0 | 00:46:19 | node821 | PASS | 0 | 1.49 | 15,680 | regular non-symlink, 3,766,013 bytes, mode 0644 | 0 |

Both roots contain exact transition 2,007,040. Checkpoint bytes and content hashes were not read, copied, or committed.

Model-free artifact SHA256:

- BossFight: progress `c7d3c0eb30d13499cd31aee51632f737612cb55087e785d1cbc53c7f8bcdf7fc`; trace `35162893f3cbd289281a0855e57c04e20032a44e442df9b07e8dbcba72317986`; stdout `df13a8943ed49af5838cb5a863dc2f2787bb8d8f6dd7e43364d16063d115f3d9`; stderr `7652399df8c088bb9dd20b4c4e78f1f270073d60c15fe6c4bed0d5a62a7bdf51`.
- CaveFlyer: progress `d76964757ea381b8ba0dcb55f924c3e159e3577164fd95318b890508541d3ed6`; trace `9718e9fb67e69b07a32fa979ac10ad88f2168c41d5e30881c6e456dd7edf3b41`; stdout `7c1a258a3205fc8c6fb5ab6dbab630a780f43cd044c1e33d2a9198814790ece0`; stderr `20dabbefcdc9c7075ec1ed64359fe4367da96b569d3922a599cc884d486cdcd1`.

## Frozen complete-trace aggregation

| Environment | Metric actor norm share | Metric actor energy share | Full actor norm share | Full actor signed projection | Shared actor norm share | Shared actor signed projection | A/C cosine | Cancellation ratio | Clip rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BossFight | 0.2668965906 | 0.1170310304 | 0.7293795347 | 0.9426795542 | 0.7316334844 | 0.9431086183 | -0.1584415140 | 0.7375108300 | 0.614923 |
| CaveFlyer | 0.2284597233 | 0.0806122199 | 0.7332879305 | 0.9262807667 | 0.7312285602 | 0.9210293591 | -0.1278955120 | 0.7510870000 | 0.327168 |

Early/Middle/Late full actor norm medians are `0.752406/0.735117/0.701264` for BossFight and `0.731316/0.743265/0.724598` for CaveFlyer. Signed projection medians are `0.929649/0.959513/0.932092` and `0.924204/0.933160/0.921803`, respectively. Frozen aggregation passes all complete records, including RHS/alpha/direction reconstruction, installed-direction identity, structural zeros, Cholesky, residual, and finite checks.

## Bounded interpretation

For both completed cells, curvature 4 makes the pre-inverse metric critic-heavy (actor energy only 8--12%) but the full coupled inverse yields a strongly actor-dominant applied direction (actor signed projection about 92--94%). This does not support the hypothesis that raising critic curvature from 0.1 to 4 increases the post-inverse critic contribution. Reward is mixed and still weak: BossFight falls from Task63 0.04 to 0.01, while CaveFlyer rises from 0.00 to 1.49 but remains below immutable Paper 4.45. CoinRun remains live and is not included in a campaign-final conclusion.
