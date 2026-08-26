# PROCGEN-FULL-SHARED-JOINT2B-SCALE-RECOVERY-6M-S0-20260826-39

## Current conclusion

`PRECHECK_BLOCKED`

The frozen pure FP64 algebra gate passed, but the one authorized production-
network functional preflight failed during model construction. Per the task's
no-repair/no-retry rule, no science cell was submitted.

## Task38 supersession

Task38 is `SUPERSEDED_BEFORE_EXECUTION`. Local Git/worktree searches found no
Task38 ID, D05 implementation, root, job mapping, or scientific artifact.
The bounded CSF3 check found no matching `squeue`, `sacct`, owned process, or
`/scratch/h99859yz` root. No Task38 file or job was created by this task.

## Causal provenance

| Evidence | SHA256 | Use |
|---|---|---|
| `joint2b_diagnosis_20260813.md` | `b157a27e06dfdd1cafa165155053273dace892a88b0fb7a476734b01a2fe453e` | fixed absolute ridge loses relative strength as learned actor/critic Gram scales grow |
| MuJoCo `export_mujoco_perenv_best.py` | `7876f7e0824d3cfa86008d0e21d38d65bb33f4c7705694aef1547a3a72857c73` | full-shared direct-system provenance |
| MuJoCo `final_last10_summary.csv` | `af63b2ffd7ee65637925de07a9d78734b26b5ffcfbe67896654bd80251a4ed95` | full-joint result provenance |
| Task06 strict Joint-2B report | `2a22205933109985c4388cdcfe64f0aeda23e44c2122fa1dec13757acb0f6251` | Paper-matched actor/control and strict 1024-row parent |
| Task07 separate-B report | `2d19e521a5439396ed7c9868a680ffdd34b5e98c3a5863e809c12a97f13b066b` | negative provenance; not reused scientifically |
| Task37 CVLM report | `d511b87bcafc9b30d9029efa81f1ae72ebb57da41c0fd938b7bf96c1e41539d3` | head-only CVLM rejection; does not reject full-shared Joint-2B |

The diagnosis separates accurate solves, cross structure, head/trunk coupling
and GAE timing from the leading scale mechanism. Task39 therefore preserves
all strict Joint-2B blocks and changes only the units of the two row/RHS
blocks and the interpretation of the existing coefficient `.5`.

## Frozen parent to target diff

Actor score rows and actor RHS, standard critic value Jacobian/residual,
full parameter union, AC/CA blocks, rollout/GAE/PopArt, SGD momentum/history,
adaptive KL, global clip, schedule and evaluation remain the strict parent.
For `s_pi=||A||_F^2/B` and `s_v=||C||_F^2/B`, Task39 divides each row block
and its RHS by its positive square-root scale, stacks the exact 1024 rows,
solves `(Hbar Hbar^T+.5I)z=bbar` in symmetric FP64 Jacobi/Cholesky form, and
reconstructs `Hbar^T z`. No floor, cap, extra coefficient, block deletion,
rank reduction or sweep exists.

## Frozen implementation identities

| File | SHA256 |
|---|---|
| trainer | `b13b3eae445f83f0e1d6565cb942c8390b6cd143fcc7f366410406c370aa0815` |
| config | `1b0cf73885dcb7c61078e463f826d6b296abfedb5a6a019f6782c6b899896b52` |
| functional preflight | `210762ad6734a02c0dcb797418589720b7791dead0da385ff9f1c67cfa51d1be` |
| preflight launcher | `6f5f2db5a4e48aa4590235d0988633de17dfcbe6f21960fdad8ace911f5f731d` |
| science launcher | `59dea11cb0fed21974f4985d2eb2016703dd33f06d2e11daa7ca419d5a888fb4` |

Local syntax compilation passed. The local interpreter has no PyTorch, so
the actual numerical and production-network gate is intentionally executed
once in the frozen CSF3 Python 3.9/PyTorch environment.

## Preflight terminal evidence

The pure frozen CSF3 Python 3.9/PyTorch algebra gate emitted
`TASK39_ALGEBRA_PASS`: exact direct reconstruction, nonzero cross block,
overall/actor-only/critic-only/opposite rescaling invariance, normalized means
one, Cholesky info zero, strict residual, and strict rejection of zero scale.

The only production-network preflight was gpuH job `19407505`. Scheduler state
is `FAILED/1:0`, elapsed `00:00:19`, node820; root state/rc are
`PRECHECK_FAIL/1`. Trainer/config/preflight hashes in the root exactly match
the frozen identities above. The H200 was visible and no OOM, CUDA, NCCL,
disk, quota, NaN or Inf signature exists.

The exact failure occurs before parameter construction, Jacobian rows or a
Joint-2B solve. `preflight_full_shared_joint2b_scale_recovery.py:53` passes
`3` as the ResNet image-size argument. The production constructor reaches
`utils/resnet.py:41`, where successive pooling of the resulting 3x3 dummy
image produces `(16x1x1) -> (16x0x0)` and raises:

```text
RuntimeError: Given input size: (16x1x1). Calculated output size: (16x0x0).
Output size is too small
```

This is `precheck-failure/actual-network-construction-harness-image-size`, not
scientific or solver evidence. The no-field-repair contract forbids correcting
the harness or rerunning the gate under Task39.

## Science matrix

No Task39 science job, root, process, transition, trace, checkpoint, model,
Paper comparison, early-stop action, or monitor exists. The four authorized
cells were not submitted because `PRECHECK_PASS` was not obtained. Task38
remains absent and Task37 evidence is unchanged.

## Failure ledger

| Item | State | Classification |
|---|---|---|
| Task38 | no local/remote implementation, job, root or artifact | `SUPERSEDED_BEFORE_EXECUTION` |
| pure Task39 algebra | PASS | functional algebra only, not scientific evidence |
| Task39 job `19407505` | FAILED/1:0; root PRECHECK_FAIL/1 | preflight harness actual-network construction failure |
| Task39 science | absent | correctly blocked by failed mandatory gate |

The unique terminal conclusion is `PRECHECK_BLOCKED`.
