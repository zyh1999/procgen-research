# Task31R final in-path capture report

## Identity and delivery

- Task-ID: `PROCGEN-NORMMATCH-V2-MP-MAIN-INPATH-CAPTURE-READONLY-20260825-31R`
- Assignment/origin commit: `b345ad9e22619c5f2f26fd0c8eca3722c065ad49`
- Frozen implementation commit: `ae93ca3990168c058a2d9b87662a10ca0d9e0511`
- Evidence/delivery commit: recorded by the final pushed commit containing this report
- Branch: `origin/agent-work`
- Method retained but not executed scientifically:
  `PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2`

## Unique conclusion

`OBSERVER_PERTURBED`

The only permitted bounded activity stopped during the first capture-on clean
process. The versioned capture wrapper used nested `runpy` to execute the exact
frozen Task23 probe. That changed the live frozen-probe `spec/package`
semantics before the unchanged Task28R origin validator ran. The validator
raised `RuntimeError: exact probe spec/package mismatch`; the expected
`__mp_main__` origin-policy frame-local record had therefore not yet been
created, and the capture layer then raised
`RuntimeError: Task31R required existing origin/runpy frames unavailable`.

This is affirmative non-perturbation failure. Per the one-activity hard stop,
the implementation was neither repaired nor rerun.

## Activity and process matrix

| Case | Started | Probe rc | Capture | Result |
|---|---:|---:|---|---|
| capture-on 1 | yes | 1 | absent | nested-runpy changed frozen probe `spec/package` |
| capture-on 2 | no | n/a | absent | not started after hard stop |
| capture-off 1 | no | n/a | absent | not started after hard stop |
| capture-off 2 | no | n/a | absent | not started after hard stop |

Because the mechanism itself perturbed the first process, there are no valid
capture-on/off normalized hashes, import-order differences, or module-set
comparisons to report. Inventing those fields or continuing to later controls
would violate the task's hard-stop rule.

## Preserved successful pre-failure evidence

Before the origin-scan failure, the unchanged production path completed:

- exact 938,979-parameter CUDA network construction;
- `GPUH_HYBRID_HEAD_COMPATIBILITY_PASS`;
- three-way resolved-config identity, config ledger SHA prefix `b44c9a`;
- structural manifest SHA prefix `3f91f5`;
- connectivity proof SHA prefix `8f4486`;
- actor/shared direction and one-step policy-logit bit identity;
- NormMatch head proof and Task27 runtime semantic-binding proof;
- deterministic/Paper/target proposal norms
  `0.6050832272 / 0.9192549586 / 0.9192548990`;
- scale `1.519220710`, cosine `0.8612535000`;
- FP64 relative residual `8.627e-16`, zero Cholesky failures.

These are compatibility/preflight-path observations only. No scientific
training, reward, transition, or performance evidence was produced.

## Frozen identity audit

The implementation added no import beyond the frozen Task23 probe import set
and registered no audit, trace, profile, or import hook. It introduced no
classifier, policy, allowlist, manifest acceptance, or `sys.modules` rebinding.
Protected identities remained:

| Artifact | SHA256 |
|---|---|
| trainer | `0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b` |
| config | `9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda` |
| Task14 preflight | `b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc` |
| regression | `f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c` |
| stage monitor | `536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e` |
| bundle archive | `3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f` |
| bundle manifest | `99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa` |
| Task23 hook | `8d9206a6defc4525114398a952d29ffdd4872cd933dc5c9b96fc838bd1273dbe` |
| frozen Task23 probe | `c3529cb171306d7b3b0517974a682ddffb91d65dc015c102b1e84658e9eeb1f5` |
| Task25 classifier | `f80de2abbcbce29e7a57ef456156c86636798c4e1ea37171922b3b466b6790fc` |
| Task26 AST helper | `c753b38c229a65dcecd54eb376aeabbcbd45586426a000970ea905f2982674b6` |
| Task27 preflight | `e43fe7e730e840de07cc467bbc56900591c581fd24d4520abe129b0bad3d2cfb` |
| Task28R extension | `96da9c8ee7497ce01df3230f6fa0875a81d1175c958c1a5d8276390090c118ad` |

Task31R implementation artifacts:

| Artifact | SHA256 |
|---|---|
| capture wrapper | `c222d770079f433dd0372bdb10609eb2f3b53213a034cab4f00cd11bc9e8263e` |
| analyzer | `48fb07dda65ad1ef67dec1562b3d2411f0e83e0e5274258da76772fd58ae7473` |
| Slurm wrapper | `21420e044c46898d606f57ede25ddb7c08f3997449156215ee453f3415b8fd84` |
| frozen-identity test | `ef90a1b059c569bce7312af96c09ed3c28b10de04e1f46255a43aae4bf46077e` |

## CPython source mapping

- `multiprocessing/__init__.py`, SHA256
  `a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b`,
  line 37: import-time `__main__`/`__mp_main__` alias assignment.
- `multiprocessing/spawn.py`, SHA256
  `16ce6d81f8b5ef7228e5500bff04b37bdceb3d7dfc8d6de3ad523598798c43f4`,
  lines 125, 234, 236, 262 and 290: spawn preparation and child-main
  replacement chain.

The Task31R observation did not validly reach the natural origin-scan boundary,
so these frozen source facts cannot repair or substitute for missing natural
runtime evidence.

## Scheduler, process, artifact and error terminal state

- Single authorized Slurm job: `19279429`, user `h99859yz`, gpuH node821.
- Scheduler: `FAILED/1:0`, elapsed `00:00:18`.
- Root status/rc: `READONLY_CAPTURE_FAIL/1`.
- First capture-on probe rc: `1`.
- No live Task31R process remains.
- No capture JSON, decision JSON, later case directory, scientific marker,
  progress, metric trace, checkpoint, or model exists.
- Hard-error review found no actual OOM, CUDA, NCCL, disk/quota, NaN, or Inf
  failure. Matches such as `numpy.nanfunctions` and `torch.cuda.nccl` were
  import-time module names, not runtime errors.
- Model-free evidence archive:
  `remote_launch_staging/procgen_normmatch_v2_mp_main_inpath_capture_readonly_20260825_31r/evidence/task31r_model_free_evidence_19279429.tar.gz`;
  SHA256 `fea85c23140260188668fa77a3ea49150125046b149a021ef1f8735717a9bfbd`.

## Preserved ledgers and prohibited work

Task29's observer-import-timing and natural non-alias failures, and Task30's
read-only provenance ledger, remain unchanged. No classifier, acceptance
category, policy, allowlist, manifest, bundle, formal audit, environment
preflight, scientific job/root/transition, checkpoint/model, cancellation,
second candidate, or monitor was created or modified.
