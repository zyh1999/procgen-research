# Machine-auditable infrastructure-only diff

Scientific files are invariant:

- trainer `41334b59aa98e03920571251da8498a1ecb816ea72afff72d8134fa8fd314f9a`;
- config `69d12937debb8ef8b4531e79b8f9613185b26e4e9056a51e67754129b269391d`;
- target command remains the Paper trainer, config basename, exact environment,
  requested seed, and device 0. No trainer/config field is overridden.

gpuA to gpuL changes are limited to partition/account/QOS/GRES, CPU/memory/time,
no-requeue, scheduler-log and noncolliding root placement, plus two corrections
for the immutable pre-training failure: pass the config basename required by
the unchanged Paper CLI and make optional log discovery safe.

gpuH adds packing only: partition `gpuH`, account `gpu-h200-fse-pgdr`, QOS
`gpu-h200-fse`, one H200, eight CPUs, four independently submitted environment
bundles, and eight independent seed children per bundle. The in-allocation
preflight runs before children and makes no scientific update. Every child
executes the same frozen trainer/config and has an independent root, command,
stdout/stderr, progress, trace, checkpoint, rc, status, hashes, and seed.

Verification commands before submission:

```text
python3 audit_paper_matched_diff.py
bash -n paper_matched_deterministic_ggn_v1_1m_gpul.sbatch
bash -n paper_matched_deterministic_ggn_v1_1m_gpuh_bundle.sbatch
python3 -m py_compile gpul_compatibility_preflight.py gpuh_bundle_compatibility_preflight.py
git diff --no-index paper_matched_deterministic_ggn_v1_1m_gpua.sbatch paper_matched_deterministic_ggn_v1_1m_gpul.sbatch
```
