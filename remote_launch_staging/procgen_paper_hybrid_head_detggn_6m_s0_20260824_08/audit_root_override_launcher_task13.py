#!/usr/bin/env python3
"""Fail closed unless Task 13 changes only artifact-root routing/provenance."""
import difflib
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
base = here / "hybrid_head_detggn_6m_gpuh.sbatch"
variant = here / "hybrid_head_detggn_6m_gpuh_root_override_task13.sbatch"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_marked(lines, begin, end):
    output = []
    inside = False
    for line in lines:
        if line == begin:
            assert not inside
            inside = True
            continue
        if line == end:
            assert inside
            inside = False
            continue
        if not inside:
            output.append(line)
    assert not inside
    return output


assert sha(base) == "ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e"
base_lines = base.read_text().splitlines()
variant_lines = variant.read_text().splitlines()

normalized_variant = strip_marked(
    variant_lines, "# ROOT_OVERRIDE_BEGIN", "# ROOT_OVERRIDE_END"
)
normalized_variant = strip_marked(
    normalized_variant, "# ROOT_PROVENANCE_BEGIN", "# ROOT_PROVENANCE_END"
)
base_campaign = "CAMPAIGN=/scratch/h99859yz/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08"
assert base_campaign in base_lines
base_normalized = [
    "CAMPAIGN=<ARTIFACT_ROOT>" if line == base_campaign else line
    for line in base_lines
]
normalized_variant.insert(base_normalized.index("CAMPAIGN=<ARTIFACT_ROOT>"), "CAMPAIGN=<ARTIFACT_ROOT>")
assert normalized_variant == base_normalized

trainer_cmd = 'CMD=("$PY" -u "$TRAINER" --config "$(basename "$CONFIG")" --env_name "$ENV_NAME" --seed 0 --device 0)'
preflight_cmd = 'if ! "$PY" "$PREFLIGHT" "$TRAINER" "$CONFIG" "$ROOT/parameter_partition.json" "$TRAINER_SHA" "$CONFIG_SHA" > "$ROOT/compatibility.out" 2> "$ROOT/compatibility.err"; then'
for text in (base_lines, variant_lines):
    assert text.count(trainer_cmd) == 1
    assert text.count(preflight_cmd) == 1

assert "CAMPAIGN=${PROCGEN_CAMPAIGN_ROOT:?PROCGEN_CAMPAIGN_ROOT required}" in variant_lines
assert "CAMPAIGN=$(readlink -m -- \"$CAMPAIGN\")" in variant_lines
assert 'if [ -e "$ROOT" ]; then echo "collision: $ROOT" >&2; exit 90; fi' in variant_lines

payload = {
    "result": "ROOT_OVERRIDE_LAUNCHER_EQUIVALENCE_PASS",
    "base_launcher_sha256": sha(base),
    "variant_launcher_sha256": sha(variant),
    "normalized_launcher_sha256": hashlib.sha256(
        ("\n".join(base_normalized) + "\n").encode()
    ).hexdigest(),
    "normalized_non_root_provenance_byte_identical": True,
    "trainer_command": trainer_cmd,
    "preflight_invocation": preflight_cmd,
    "unified_diff": list(difflib.unified_diff(
        base_lines, variant_lines, fromfile=base.name, tofile=variant.name,
        lineterm="",
    )),
}
print(json.dumps(payload, indent=2, sort_keys=True))
