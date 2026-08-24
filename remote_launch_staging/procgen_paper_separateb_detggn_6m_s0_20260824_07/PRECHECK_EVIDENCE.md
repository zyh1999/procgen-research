# Precheck evidence

Static identity audit: `AUDIT_PASS`; Paper actor path
`STRICT_LITERAL_PRESERVED`. Numerical regression ran in the authorized CSF3
`.RLvenv` and returned:

```
REGRESSION_PASS
actor_direction=BIT_IDENTICAL
adaptive_kl=paper_thresholds_0.005_0.04_per_minibatch
critic_system=independent_BxB_no_cross_blocks
critic=deterministic_Jv_residual_lambda0.1
solver=FP64_Jacobi_Cholesky relative_residual=2.549e-16
illegal_P1_joint_lowfisher_cross_fields=REJECTED
```

Historical provenance: `DISTINCT_FORMULA_PASS`; see
`HISTORICAL_PROVENANCE.md`. New trainer hash `b0dad110...` differs from
CSF3 block-trace `1881bf7c...`, CSF3 expected `c976c0e5...`, and Bede expected
`0514703d...`; the formula table proves this is not merely a byte-different
rewrite.

Live resource snapshot, CSF3 `2026-08-24T17:18:01+01:00`: user queue contained
only unrelated multicore job `19051570`; no user GPU or Procgen target existed.
gpuH nodes were available in mixed states and the association
`gpu-h200-fse-pgdr/gpu-h200-fse` permits up to four H200. gpuL and gpuA were
also live but contended. Bede authentication failed before scheduling and the
two named 4090 aliases did not resolve; neither was mutated. The quarantined
host was not contacted. Placement is therefore four independently auditable
gpuH jobs, one H200 and eight CPUs each, with no Jupyter and no requeue.

Frozen SHA256:

| Artifact | SHA256 |
|---|---|
| trainer | `b0dad110c36dbab4c601aa9128ba51eb437bfc6a3e9cadf87be8fd2172f3729a` |
| config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` |
| launcher | `5d3caf579d5203eef44f54f93af7f8c53567a0e423e27ea1cd604be7a9bd0554` |
| manifest | `225f0d57f94015480a48a448ae8dc1381281033aced9f40232e6321881456351` |
| regression | `ded706f5c848283b73aa3a0924cadb1359055695d88a584341181814de391ec1` |
| monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` |
| gpuH preflight | `72e4c5471e8a7ca9e2c8ada01fd75734ed09806232ff1409bbe164e2e6ad9faa` |
