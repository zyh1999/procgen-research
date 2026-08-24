# Task 12 structural-manifest recovery freeze

- Task: `PROCGEN-HYBRID-HEAD-STRUCTURAL-MANIFEST-RECOVERY-6M-MISSING3-20260824-12`
- Assignment: `05fe72ba8d13217217a3039990cdba2ec5432279`
- Parent harness freeze: `26b2252527076df4bfe537a8612446317cbdcf3a`
- Scope: evidence/preflight only. No scientific implementation changed.

## Immutable scientific identity

- Trainer SHA256: `7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54`
- Config SHA256: `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda`
- Scientific launcher SHA256: `ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e`
- Stage monitor SHA256: `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e`

## Authorized correction

The previous `parameter_partition.json` mixed environment-independent
parameter structure with environment/input-dependent connectivity magnitudes.
The corrected preflight writes:

- `structural_manifest.json`: ordered names, partition, shape, dtype,
  `requires_grad`, numel, trainable/optimizer membership and exact counts.
- `connectivity_probe.json`: environment identity, per-parameter policy/value
  connectivity and Jacobian probe norms, partition-name agreement, finite-value
  checks and a semantic PASS field.

When the frozen scientific launcher supplies its legacy
`parameter_partition.json` output argument, that file is a byte-identical copy
of `structural_manifest.json`; it no longer contains probe values. This keeps
the scientific launcher byte-identical while repairing only its preflight
evidence contract.

The old environment-specific full-file SHA assertion is removed. No observed
environment hash is hard-coded or allow-listed. Cross-environment byte identity
must be established from four independent no-training allocations before any
scientific submission.

## Local static verification

- Python syntax compilation: PASS.
- Canonical preflight static test: PASS.
- Git whitespace/error check: PASS.
- Scientific identity hashes: unchanged as listed above.
- Corrected preflight harness SHA256:
  `704278e8b5802498b8e065b9f12945e2cb72a665cdd28845b2401091b2e993ea`.
- Corrected four-environment preflight launcher SHA256:
  `6dcd0757b88b065eef2befbe862bc28a6da269eb86ea733a24c92919f39ad76b`.

Remote four-environment validation evidence and its structural/probe hashes
will be appended only after the four one-shot allocations reach terminal state.
