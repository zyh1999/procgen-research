# Task63 partial terminal evidence: BigFish and CoinRun

This is a model-free bounded archive. No model/checkpoint bytes or content
hashes are included. Scheduler state is authoritative.

## Scheduler and endpoint

| Environment | Job | Scheduler | Elapsed | Node | Root | rc | Exact transition | Reward | Immutable Paper |
|---|---:|---|---:|---|---|---:|---:|---:|---:|
| BigFish | 1078181 | COMPLETED/0:0 | 02:44:51 | gpu025 | PASS | 0 | 2,007,040 | 5.08 | 9.28 |
| CoinRun | 1078184 | COMPLETED/0:0 | 02:45:51 | gpu007 | PASS | 0 | 2,007,040 | 10.00 | 3.70 |

BigFish ended at Paper ratio `5.08/9.28 = 0.5474137931`; CoinRun ended at
`10.00/3.70 = 2.7027027027`. Task63 is read-only telemetry and did not invoke
reward cancellation.

Both roots contain 49 progress rows and 15,680 complete minibatch records.
The remote checkpoint in each source log is a regular non-symlink file of
3,766,013 bytes with mode 664; only this stat metadata is retained.

## Model-free artifact identities

| Environment | progress.csv SHA256 | metric_trace.jsonl SHA256 | stdout SHA256 | stderr SHA256 |
|---|---|---|---|---|
| BigFish | `11c75a1cd47c2e559b4d78ed1652e32b8afb4ce7a8872cdbc6b8e02ec5a766b4` | `93d4259e8d83ca78cd21519c23a16ec05d8cea4a30d704727c8e8badeb63edbe` | `18fcbc860e4e5ab0905380fed29a1c37002595c468687ff577beacdf7ae11d8c` | `beb01289e947fb0c4d226f655915ad768d4b9a240e21181ec8f4523e2f2c9944` |
| CoinRun | `49638ea6c3cb7200f5322cfa5eda460938fe5f51f58a0bb69d328eabaed0fadc` | `31436e836a02ec0b789c124a1310a18397e06fed98a1dc75663a02339fc123fd` | `fff7ddcbe79591da83b36575c9e0c3ec39d27d14e38630627bedbb2308c190c7` | `fe2459ecb4928dd1890c74f93964b7548872c8c88f0aae685e71f4eb08833922` |

Focused Traceback/OOM/CUDA/NCCL/disk/quota/NaN/nonfinite and explicit
reconstruction/identity/structural failure scans returned zero matches.

## Frozen aggregation result

BigFish passed the unmodified frozen aggregator over all 15,680 records:
`TASK63_AGGREGATION_PASS`. Overall medians were:

- full actor norm/projection share `.4404799193/.3784165233`;
- shared actor norm/projection share `.4497030973/.3964104950`;
- metric actor norm/energy share `.7089808881/.8558053970`;
- full actor/critic cosine `-.0394019987`;
- cancellation/amplification `.7053162158`;
- clip rate `.9894770408`.

Early/middle/late full actor norm medians were
`.4582246542/.4230997860/.4432718307`; signed projection medians were
`.4151375592/.3437452614/.3819947690`. Shared actor norm medians were
`.4623425603/.4336541146/.4552244842`; projection medians were
`.4233497232/.3635070920/.4062507600`.

CoinRun training itself completed cleanly, but the unmodified frozen
aggregator stopped at record index 3808 with `policy projection drift`.
The immutable record is finite and exposes a near-zero policy-total edge case:

- transition `491,520`, entropy `3.681075735e-18`;
- actor raw scale `7.786322353e-32`, actor Fisher quadratic `3.332481715e-35`;
- policy total direction norm `5.152824349e-17`;
- actor/critic policy norms `5.152824349e-17/0`;
- policy signed projections `.002648128429/0`, sum `.002648128429`;
- full/shared/value projection sums remain exactly 1;
- Cholesky info 0, relative residual `8.810887588e-14`, finite scan 1;
- RHS reconstruction 0, alpha reconstruction `1.365421640e-15`, direction
  reconstruction 0, structural zeros exact.

This is classified as a frozen telemetry-aggregation validator failure under
an actor-saturated near-zero policy subspace, not a training, solver, GPU or
infrastructure failure. It was not repaired or rerun. Descriptive CoinRun
full-trace medians may be inspected read-only, but there is no valid frozen
`TASK63_AGGREGATION_PASS` for CoinRun.

BossFight 1078182 and CaveFlyer 1078183 remain RUNNING and untouched. Task64
and every unrelated job/root were untouched.
