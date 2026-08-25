#!/usr/bin/env python3
"""Analyze Task30 observed processes and no-observer control."""
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve(strict=True)
DEPLOYS = [Path(item).resolve(strict=True) for item in sys.argv[2:6]]
PROBE_SHA = "c3529cb171306d7b3b0517974a682ddffb91d65dc015c102b1e84658e9eeb1f5"
EXPECTED_LABELS = [
    "child_process_entry", "closure_probe_start", "trainer_import_before",
    "trainer_module_entry", "trainer_import_after",
    "production_model_construction_after", "origin_scan_before",
]


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def normalize(value, deploys=DEPLOYS):
    if isinstance(value, dict):
        return {
            key: normalize(item, deploys)
            for key, item in sorted(value.items())
            if key not in {
                "object_id", "dictionary_id", "data_pointer", "storage_data_pointer",
                "pid", "snapshot_sha256",
            }
        }
    if isinstance(value, list):
        return [normalize(item, deploys) for item in value]
    if isinstance(value, str):
        result = value
        for index, deploy in enumerate(deploys):
            result = result.replace(str(deploy), "<DEPLOY>")
        result = re.sub(r"task30-(?:deploy|empty)-(?:obs[123]|control)-[^/\s]+", "<TASK30_TMP>", result)
        return result
    return value


def fd_identity(path):
    raw = Path(path)
    raw_lstat = os.lstat(raw)
    resolved = raw.resolve(strict=True)
    resolved_lstat = os.lstat(resolved)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("O_NOFOLLOW unavailable")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow)
    try:
        opened = os.fstat(descriptor)
        data = b""
        offset = 0
        while len(data) < opened.st_size:
            chunk = os.pread(descriptor, opened.st_size - len(data), offset)
            if not chunk:
                break
            data += chunk
            offset += len(chunk)
    finally:
        os.close(descriptor)
    if len(data) != opened.st_size:
        raise RuntimeError("fd short read: " + str(path))
    def record(value):
        return {
            "device": value.st_dev, "inode": value.st_ino, "uid": value.st_uid,
            "gid": value.st_gid, "mode": oct(stat.S_IMODE(value.st_mode)),
            "size": value.st_size, "regular_file": stat.S_ISREG(value.st_mode),
            "symlink": stat.S_ISLNK(value.st_mode),
        }
    return {
        "raw_path": str(raw), "resolved_path": str(resolved),
        "samefile": os.path.samefile(raw, resolved),
        "raw_lstat": record(raw_lstat), "resolved_lstat": record(resolved_lstat),
        "opened_fd": record(opened), "sha256": sha_bytes(data),
    }


def classify_backing(record, deploy):
    if not record or "raw_path" not in record:
        return {"classification": "absent"}
    identity = fd_identity(record["raw_path"])
    if identity["sha256"] == PROBE_SHA:
        classification = "TASK28R_FROZEN_PROBE"
    else:
        try:
            relative = Path(identity["resolved_path"]).relative_to(deploy).as_posix()
        except ValueError:
            relative = None
        manifest = json.loads((deploy / "BUNDLE_MANIFEST.json").read_text())
        files = {item["bundle_path"]: item["sha256"] for item in manifest["files"]}
        if relative in files and files[relative] == identity["sha256"]:
            classification = "BUNDLE_MANIFEST_EXACT"
        elif relative is not None:
            classification = "DEPLOY_NONMANIFEST"
        else:
            classification = "OTHER_SOURCE"
    return {"classification": classification, "identity": identity}


def import_sequence(path):
    values = []
    pattern = re.compile(r"^import time:.*\|\s*([^\s].*)$")
    for line in Path(path).read_text(errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            values.append(match.group(1).strip())
    return values


ledgers = []
for index in range(1, 4):
    case = ROOT / ("obs" + str(index))
    ledger = json.loads((case / "natural_state_ledger.json").read_text())
    labels = [item["label"] for item in ledger["snapshots"]]
    if labels != EXPECTED_LABELS:
        raise RuntimeError("milestone completeness/order mismatch: " + str((index, labels)))
    for item in ledger["snapshots"]:
        canonical = json.dumps({key: value for key, value in item.items() if key != "snapshot_sha256"}, sort_keys=True, separators=(",", ":")).encode()
        if sha_bytes(canonical) != item["snapshot_sha256"]:
            raise RuntimeError("snapshot SHA mismatch: " + str((index, item["label"])))
        item["main_backing_proof"] = classify_backing(item["main"].get("backing"), DEPLOYS[index - 1])
        item["mp_main_backing_proof"] = classify_backing(item["mp_main"].get("backing"), DEPLOYS[index - 1])
        item["top_level_code_sha256"] = sha_bytes(json.dumps(item["current_top_level_code"], sort_keys=True, separators=(",", ":")).encode())
    ledgers.append(ledger)

normalized = [normalize(item) for item in ledgers]
normalized_hashes = [sha_bytes(json.dumps(item, sort_keys=True, separators=(",", ":")).encode()) for item in normalized]
reproduction_consistent = len(set(normalized_hashes)) == 1

required_artifacts = [
    "resolved_config_preflight.json", "resolved_config_scientific_launcher_dry_run.json",
    "resolved_config_trainer_entry.json", "structural_manifest.json",
    "connectivity_probe.json", "runtime_semantic_binding_ledger.json", "ast_call_ledger.json",
]
artifact_matrix = {}
for name in required_artifacts:
    values = []
    for case in [ROOT / "obs1", ROOT / "obs2", ROOT / "obs3", ROOT / "control"]:
        payload = json.loads((case / name).read_text())
        values.append(sha_bytes(json.dumps(normalize(payload), sort_keys=True, separators=(",", ":")).encode()))
    artifact_matrix[name] = {"normalized_sha256": values, "all_equal": len(set(values)) == 1}

critical_prefixes = (
    "GPUH_", "canonical_", "resolved_", "actual_", "head_", "partition_", "torch=",
    "TASK27_", "task27_", "connectivity_", "structural_", "critic_", "gpu=", "paper_rows=",
)
critical_outputs = []
for case in [ROOT / "obs1", ROOT / "obs2", ROOT / "obs3", ROOT / "control"]:
    lines = [line for line in (case / "probe.out").read_text().splitlines() if line.startswith(critical_prefixes)]
    critical_outputs.append(lines)

imports = [import_sequence(ROOT / name / "importtime.err") for name in ("obs1", "obs2", "obs3", "control")]
observer_imports_without_site = [[item for item in values if item != "sitecustomize"] for values in imports[:3]]
import_order_match_control = all(values == imports[3] for values in observer_imports_without_site)
observer_extra_imports = [item for item in imports[0] if item not in imports[3]]

old_error = "bundle module absent from manifest or hash mismatch: __mp_main__"
terminal_errors = [(ROOT / name / "probe.err").read_text(errors="replace") for name in ("obs1", "obs2", "obs3", "control")]
same_origin_scan_failure = all(old_error in value for value in terminal_errors)

origin_relations = []
for ledger in ledgers:
    origin = next(item for item in ledger["snapshots"] if item["label"] == "origin_scan_before")
    origin_relations.append({
        "object_identity": origin["object_identity"],
        "main_backing": origin["main_backing_proof"]["classification"],
        "mp_main_backing": origin["mp_main_backing_proof"]["classification"],
        "main_name": origin["main"]["name"], "mp_main_name": origin["mp_main"]["name"],
        "main_file": normalize(origin["main"]["file"]),
        "mp_main_file": normalize(origin["mp_main"]["file"]),
        "difference": normalize(origin["dictionary_difference"]),
    })
relation_consistent = len({json.dumps(item, sort_keys=True) for item in origin_relations}) == 1

observer_nonperturb = {
    "all_normalized_artifacts_equal_control": all(item["all_equal"] for item in artifact_matrix.values()),
    "critical_stdout_equal_control": all(item == critical_outputs[3] for item in critical_outputs[:3]),
    "origin_scan_failure_equal_control": same_origin_scan_failure,
    "import_order_equal_after_excluding_observer_module": import_order_match_control,
    "observer_only_imports": observer_extra_imports,
    "observer_removed_before_probe_body": all(
        item["observer"]["observer_module_removed_before_probe_body"] for item in ledgers
    ),
    "task27_rng_outputs_parameters_telemetry_bit_identical": all(
        json.loads((ROOT / name / "runtime_semantic_binding_ledger.json").read_text())["wrapped_unwrapped_rng_outputs_parameters_optimizer_telemetry_bit_identical"]
        for name in ("obs1", "obs2", "obs3", "control")
    ),
}

if not reproduction_consistent or not relation_consistent:
    conclusion = "INSUFFICIENT_EVIDENCE"
elif not all(value for key, value in observer_nonperturb.items() if key != "observer_only_imports"):
    conclusion = "OBSERVER_PERTURBED"
elif not all(not item["object_identity"] for item in origin_relations):
    conclusion = "INSUFFICIENT_EVIDENCE"
elif not all(item["main_backing"] == "TASK28R_FROZEN_PROBE" for item in origin_relations):
    conclusion = "NO_SAFE_ALIAS_RELATION"
else:
    conclusion = "NATURAL_MP_MAIN_RELATIONSHIP_PROVEN"

payload = {
    "result": "TASK30_NATURAL_STATE_ANALYSIS_COMPLETE",
    "unique_conclusion": conclusion,
    "reproduction_count": 3, "control_count": 1,
    "reproduction_consistent": reproduction_consistent,
    "normalized_reproduction_sha256": normalized_hashes,
    "origin_scan_relations": origin_relations,
    "relation_consistent": relation_consistent,
    "artifact_matrix": artifact_matrix,
    "critical_stdout_sha256": [sha_bytes("\n".join(item).encode()) for item in critical_outputs],
    "observer_nonperturbation": observer_nonperturb,
    "import_sequence_lengths": [len(item) for item in imports],
    "cp39_transition_map": {
        "multiprocessing_init": {"line": 37, "sha256": "a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b"},
        "multiprocessing_spawn": {"lines": [125, 234, 236, 262, 290], "sha256": "16ce6d81f8b5ef7228e5500bff04b37bdceb3d7dfc8d6de3ad523598798c43f4"},
    },
    "ledgers": ledgers,
}
(ROOT / "task30_analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("TASK30_UNIQUE_CONCLUSION=" + conclusion)
