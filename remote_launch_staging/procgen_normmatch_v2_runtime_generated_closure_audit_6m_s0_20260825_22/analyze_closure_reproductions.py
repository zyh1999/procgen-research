#!/usr/bin/env python3
"""Compare two independent closures and enforce Task22 per-file requirements."""
import json
import sys
from pathlib import Path


def normalized(payload):
    return {
        "python": payload["process"]["python"].splitlines()[0],
        "torch_distribution": payload["process"]["torch_distribution"],
        "torch_version": payload["process"]["torch_version"],
        "production_construction": payload["production_construction"],
        "prestart_relative_paths": [item["relative_path"] for item in payload["designated"]["prestart"]],
        "post_relative_paths": [item["relative_path"] for item in payload["designated"]["post_model_construction"]],
        "filesystem_event_signature": [
            {"event": item["event"], "paths": item["paths"], "mode": item.get("mode"), "flags": item.get("flags")}
            for item in payload["designated"]["filesystem_events"]
        ],
        "candidate_modules": [
            {key: item[key] for key in (
                "sys_modules_key", "module_name", "module_type_module", "module_type_name",
                "file", "package", "spec_present", "spec_name", "spec_origin",
                "loader_module", "loader_class", "physical_artifact",
            )}
            for item in payload["candidate_modules"]
        ],
        "torch_classes_source_provenance": {
            key: payload["torch_classes_source_provenance"][key]
            for key in (
                "module", "sha256", "size", "ast_parse", "compile", "distribution",
                "version", "distribution_path", "record_hash", "record_size",
                "synthetic_file_assignment_present",
            )
        },
    }


first = json.loads(Path(sys.argv[1]).read_text())
second = json.loads(Path(sys.argv[2]).read_text())
first_normalized = normalized(first)
second_normalized = normalized(second)
if first_normalized != second_normalized:
    raise RuntimeError("independent production generated closures are not normalized-equal")
for payload in (first, second):
    if payload["designated"]["prestart"]:
        raise RuntimeError("designated directory was not empty at prestart")
    module = payload["torch_classes"]
    if module["module_name"] != "torch.classes" or module["module_type_module"] != "torch._classes":
        raise RuntimeError("torch.classes synthetic module identity changed")
    if module["file"] != "_classes.py":
        raise RuntimeError("unexpected torch.classes pseudo-file spelling")

eligible = []
ineligible = []
for module in first["candidate_modules"]:
    artifact = module["physical_artifact"]
    reasons = []
    if artifact is None:
        reasons.append("no physical artifact")
    else:
        if not artifact["regular_file"]:
            reasons.append("not a regular file")
        if artifact["symlink"]:
            reasons.append("symlink file")
    relevant_events = [
        item for item in first["designated"]["filesystem_events"]
        if module["file"] in item["paths"] or module["spec_origin"] in item["paths"]
    ]
    if not relevant_events:
        reasons.append("no create/write/rename/delete lifecycle")
    if module["spec_present"] is False:
        reasons.append("no module spec")
    if module["loader_module"] is None or module["loader_class"] is None:
        reasons.append("no loader identity")
    record = {"module": module["sys_modules_key"], "reasons": reasons}
    (ineligible if reasons else eligible).append(record)

result = {
    "result": "PRECHECK_BLOCKED",
    "reason": "runtime pseudo-origin cannot satisfy per-file generated-artifact contract",
    "normalized_closure_equal": True,
    "independent_processes": 2,
    "eligible_approved_runtime_generated_thirdparty_modules": eligible,
    "ineligible_runtime_pseudo_origins": ineligible,
    "formal_clean_room_audit_permitted": False,
}
Path(sys.argv[3]).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
if not any(item["module"] == "torch.classes" for item in ineligible):
    raise RuntimeError("torch.classes unexpectedly satisfied the artifact contract")
print("TASK22_NORMALIZED_CLOSURE_EQUAL_PASS")
print("TASK22_CLOSURE_PROVENANCE_BLOCKED_SYNTHETIC_ORIGIN_NO_ARTIFACT")
