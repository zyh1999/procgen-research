# Executor Report

## Metadata

- Task-ID: `PROCGEN-HYBRID-HEAD-ASSERTION-FIX-AND-6M-S0-20260824-10`
- Assignment: `30fb08b6791c64cf5fde9e1de5355cb3e72a24c2`
- Scientific freeze: `fe4b8a58812e80689705abec11364457cae31e26`
- Canonical preflight freeze: `9a56a6aba6d70ce7a16b9e81cf105dccbd43d638`
- Assertion-fix freeze: `a22f1a51bbcc953881e780f4dc00da16b2fc317f`
- Repository target: `origin/agent-work`

## Result

The single final assertion-fix preflight was job `19225707`. The exact frozen
production invariant passed: total938,979 parameters; policy2/3,855;
shared22/934,864; critic2/257; exact value-head names and exact manifest SHA.
The canonical production path, three identical resolved configs, real model
construction and critic-head zero policy Jacobian also passed.

The next actual-network one-step test called `autograd.grad` with every model
parameter, including non-trainable PopArt state. PyTorch raised `RuntimeError:
One of the differentiated Tensors does not require grad`. Scheduler evidence
is FAILED/1:0 after17 seconds on node820. Classification is
`infrastructure-failure/preflight-design`, not algorithm, numerical, solver,
config, partition/Jacobian or H200 incompatibility.

Actual-network one-step equality, production-scale memory and final head-solve
checks remained unreached. Task10 authorized no field repair or retry after
this run, so no scientific job/root/process/transition/artifact/checkpoint or
model exists. All prior failures remain immutable.

PRECHECK_BLOCKED
