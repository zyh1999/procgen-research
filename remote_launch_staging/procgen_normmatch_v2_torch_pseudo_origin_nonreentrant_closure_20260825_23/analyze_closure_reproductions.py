#!/usr/bin/env python3
"""Require normalized equality of two independent Task23 full closures."""
import json
import sys
from collections import Counter
from pathlib import Path


def normalize(payload):
    closure = payload["origin_closure"]
    generated = closure["runtime_generated_thirdparty_modules"]
    pseudo = closure["installed_distribution_pseudo_origins"]
    return {
        "python": payload["process"]["python"].splitlines()[0],
        "production_construction": payload["production_construction"],
        "designated_prestart": [item["relative_path"] for item in payload["designated"]["prestart"]],
        "designated_post": [item["relative_path"] for item in payload["designated"]["post"]],
        "hook_event_counts": payload["hook"]["event_counts"],
        "hook_reentrant_total": payload["hook"]["ledger"]["reentrant_total"],
        "module_classification_counts": dict(sorted(Counter(
            item["classification"] for item in closure["modules"]
        ).items())),
        "bundle_origins": sorted((key, value) for key, value in closure["bundle_origins"].items()),
        "generated": [{
            "classification": item["classification"],
            "module": item["module"],
            "loader_module": item["loader_module"],
            "loader_class": item["loader_class"],
            "file_sha256": item["file"]["sha256"],
            "file_size": item["file"]["size"],
            "content_audit": item["content_audit"],
            "generator_provenance": item["generator_provenance"],
        } for item in generated],
        "pseudo": [{
            key: item[key] for key in (
                "classification", "sys_modules_key", "module_type_module", "module_type_name",
                "distribution", "version", "source_sha256", "source_size",
                "distribution_path", "record_hash", "record_size", "static_source_proof",
            )
        } for item in pseudo],
        "generated_post": closure["runtime_generated_post_import_revalidation"],
        "pseudo_post": closure["pseudo_origin_post_audit_revalidation"],
    }


first = json.loads(Path(sys.argv[1]).read_text())
second = json.loads(Path(sys.argv[2]).read_text())
for payload in (first, second):
    if payload["result"] != "TASK23_RUNTIME_CLOSURE_PROBE_PASS":
        raise RuntimeError("closure probe did not pass")
    if payload["hook"]["ledger"]["reentrant_total"] != 0:
        raise RuntimeError("reentrant audit events compromise closure completeness")
    if payload["designated"]["prestart"] or payload["designated"]["post"]:
        raise RuntimeError("designated directory was not empty")
    closure = payload["origin_closure"]
    if len(closure["runtime_generated_thirdparty_modules"]) != 1:
        raise RuntimeError("physical generated closure is incomplete")
    if len(closure["installed_distribution_pseudo_origins"]) != 1:
        raise RuntimeError("synthetic pseudo-origin closure is incomplete")
normalized_first = normalize(first)
normalized_second = normalize(second)
if normalized_first != normalized_second:
    raise RuntimeError("independent full production closures are not normalized-equal")
decision = {
    "result": "CLOSURE_PASS",
    "independent_clean_processes": 2,
    "normalized_closure_equal": True,
    "all_physical_and_synthetic_items_approved": True,
    "normal_reentrant_event_count": 0,
    "normalized_closure": normalized_first,
    "formal_clean_room_audit_permitted": True,
}
Path(sys.argv[3]).write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
print("TASK23_TWO_PROCESS_NORMALIZED_CLOSURE_EQUAL_PASS")
print("TASK23_CLOSURE_PASS")
