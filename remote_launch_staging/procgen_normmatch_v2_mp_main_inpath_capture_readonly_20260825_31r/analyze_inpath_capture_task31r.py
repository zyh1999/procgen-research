#!/usr/bin/env python3
"""Compare two Task31R capture-on and two capture-off clean processes."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve(strict=True)
CASE_NAMES = ("on1", "on2", "off1", "off2")
EXPECTED_ERROR = "bundle module absent from manifest or hash mismatch: __mp_main__"
EXPECTED_MAIN_SHA = "c3529cb171306d7b3b0517974a682ddffb91d65dc015c102b1e84658e9eeb1f5"
EXPECTED_MP_SHA = "e43fe7e730e840de07cc467bbc56900591c581fd24d4520abe129b0bad3d2cfb"


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def normalize(value):
    if isinstance(value, dict):
        return {
            key: normalize(item)
            for key, item in sorted(value.items())
            if key not in {"object_identity", "data_pointer", "storage_data_pointer"}
        }
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"/[^\s\"']*task31r-deploy-(?:on|off)[12]-[^/\s\"']+", "<DEPLOY>", value)
        value = re.sub(r"/[^\s\"']*task31r-empty-(?:on|off)[12]-[^/\s\"']+", "<EMPTY>", value)
        value = re.sub(r"0x[0-9a-fA-F]+", "<OBJECT_ID>", value)
        return value
    return value


def normalized_json_sha(path):
    payload = normalize(json.loads(path.read_text()))
    return sha_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def import_order(path):
    result = []
    pattern = re.compile(r"^import time:.*\|\s*([^\s].*)$")
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            result.append(match.group(1).strip())
    return result


def marker(path, name):
    prefix = name + "="
    values = [line[len(prefix):] for line in path.read_text().splitlines() if line.startswith(prefix)]
    if len(values) != 1:
        raise RuntimeError("Task31R marker cardinality mismatch: " + str((path, name, values)))
    return values[0]


cases = {name: ROOT / name for name in CASE_NAMES}
relation_hashes = [marker(cases[name] / "probe.out", "TASK31R_INPATH_RELATION_SHA256") for name in CASE_NAMES]
module_set_hashes = [marker(cases[name] / "probe.out", "TASK31R_INPATH_MODULE_SET_SHA256") for name in CASE_NAMES]
imports = [import_order(cases[name] / "importtime.err") for name in CASE_NAMES]
import_hashes = [sha_bytes(json.dumps(item, separators=(",", ":")).encode()) for item in imports]

captures = [json.loads((cases[name] / "inpath_capture.json").read_text()) for name in ("on1", "on2")]
for payload in captures:
    if payload["terminal_relation_sha256"] not in relation_hashes:
        raise RuntimeError("Task31R capture relation hash not emitted by process")
    relation = payload["terminal_relation"]
    if relation["object_identity"]:
        raise RuntimeError("Task31R terminal modules unexpectedly identical")
    if relation["main"]["backing_sha256"] != EXPECTED_MAIN_SHA:
        raise RuntimeError("Task31R terminal __main__ is not exact Task23 probe")
    if relation["mp_main"]["backing_sha256"] != EXPECTED_MP_SHA:
        raise RuntimeError("Task31R terminal __mp_main__ is not deployed Task27 preflight")
    if [item["label"] for item in payload["milestones"]] != [
        "child_entry", "closure_probe_start", "trainer_import_before",
        "trainer_import_after", "production_model_construction_after",
        "origin_scan_before",
    ]:
        raise RuntimeError("Task31R milestone completeness/order mismatch")

artifact_names = (
    "resolved_config_preflight.json",
    "resolved_config_scientific_launcher_dry_run.json",
    "resolved_config_trainer_entry.json",
    "structural_manifest.json", "connectivity_probe.json", "ast_call_ledger.json",
    "runtime_semantic_binding_ledger.json", "parameter_partition.json",
    "trainable_optimizer_popart_manifest.json",
)
artifact_matrix = {}
for artifact in artifact_names:
    hashes = [normalized_json_sha(cases[name] / artifact) for name in CASE_NAMES]
    artifact_matrix[artifact] = {"hashes": hashes, "all_equal": len(set(hashes)) == 1}

critical_prefixes = (
    "GPUH_", "canonical_", "resolved_", "actual_", "head_", "partition_",
    "torch=", "TASK27_", "task27_", "connectivity_", "structural_", "critic_",
    "gpu=", "paper_rows=",
)
critical_outputs = []
for name in CASE_NAMES:
    values = [
        line for line in (cases[name] / "probe.out").read_text().splitlines()
        if line.startswith(critical_prefixes)
    ]
    critical_outputs.append(values)

errors = [(cases[name] / "probe.err").read_text(errors="replace") for name in CASE_NAMES]
probe_rcs = [int((cases[name] / "probe_rc").read_text().strip()) for name in CASE_NAMES]
expected_rejection_equal = all(EXPECTED_ERROR in item for item in errors) and len(set(probe_rcs)) == 1
hard_error_pattern = re.compile(r"(out of memory|cuda error|nccl|no space left|disk quota|nan|\binf\b)", re.I)
hard_error_hits = [hard_error_pattern.findall(item.replace(EXPECTED_ERROR, "")) for item in errors]

rng_summaries = [payload["rng_summary"] for payload in captures]
relation_equal = len(set(relation_hashes)) == 1
module_set_equal = len(set(module_set_hashes)) == 1
import_order_equal = len(set(import_hashes)) == 1
artifacts_equal = all(value["all_equal"] for value in artifact_matrix.values())
critical_equal = all(item == critical_outputs[0] for item in critical_outputs[1:])
rng_capture_equal = rng_summaries[0] == rng_summaries[1]
capture_write_only_difference = all((cases[name] / "inpath_capture.json").exists() for name in ("on1", "on2")) and all(
    not (cases[name] / "inpath_capture.json").exists() for name in ("off1", "off2")
)

if not relation_equal or not module_set_equal:
    conclusion = "NO_SAFE_ALIAS_RELATION"
elif not import_order_equal or not artifacts_equal or not critical_equal:
    conclusion = "OBSERVER_PERTURBED"
elif not rng_capture_equal or not expected_rejection_equal or any(hard_error_hits):
    conclusion = "INSUFFICIENT_EVIDENCE"
elif not capture_write_only_difference:
    conclusion = "OBSERVER_PERTURBED"
else:
    conclusion = "NATURAL_MP_MAIN_RELATIONSHIP_PROVEN"

decision = {
    "result": "TASK31R_INPATH_CAPTURE_ANALYSIS_COMPLETE",
    "unique_conclusion": conclusion,
    "capture_on_count": 2, "capture_off_count": 2,
    "relation_hashes": relation_hashes, "relation_equal": relation_equal,
    "module_set_hashes": module_set_hashes, "module_set_equal": module_set_equal,
    "import_order_hashes": import_hashes, "import_order_lengths": [len(item) for item in imports],
    "import_order_equal": import_order_equal,
    "artifact_matrix": artifact_matrix, "artifacts_equal": artifacts_equal,
    "critical_stdout_sha256": [sha_bytes("\n".join(item).encode()) for item in critical_outputs],
    "critical_stdout_equal": critical_equal,
    "rng_summaries_capture_on": rng_summaries, "rng_capture_on_equal": rng_capture_equal,
    "expected_task28r_rejection_equal": expected_rejection_equal,
    "probe_rcs": probe_rcs, "hard_error_hits": hard_error_hits,
    "capture_write_only_difference": capture_write_only_difference,
    "capture_on_terminal_relation": captures[0]["terminal_relation_normalized"],
    "capture_on_milestones": [payload["milestones"] for payload in captures],
    "cpython_transition_map": {
        "multiprocessing_init": {"line": 37, "sha256": "a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b"},
        "multiprocessing_spawn": {"lines": [125, 234, 236, 262, 290], "sha256": "16ce6d81f8b5ef7228e5500bff04b37bdceb3d7dfc8d6de3ad523598798c43f4"},
    },
}
(ROOT / "task31r_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print("TASK31R_UNIQUE_CONCLUSION=" + conclusion)
