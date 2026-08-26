#!/usr/bin/env python3
"""Import the frozen trainer from an extracted bundle and audit local origins."""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


EXPECTED_TRAINER_SHA = "ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc"
EXPECTED_CONFIG_SHA = "52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: hermetic_import_smoke BUNDLE_ROOT OUTPUT_JSON")
    bundle_root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    code_root = bundle_root / "code"
    trainer = code_root / "train_shared_det_standard_mse_ggn_head_cvlm_v1.py"
    config = code_root / "configs/adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml"
    manifest = json.loads((bundle_root / "BUNDLE_MANIFEST.json").read_text())
    entries = {entry["bundle_path"]: entry for entry in manifest["files"]}
    if sha(trainer) != EXPECTED_TRAINER_SHA or sha(config) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("frozen trainer/config hash mismatch")
    if Path.cwd() == bundle_root or inside(Path.cwd(), bundle_root):
        raise RuntimeError("smoke cwd must be outside the extracted bundle")

    original_path = list(sys.path)
    cleaned = []
    for item in original_path:
        if item in ("", "."):
            continue
        candidate = Path(item)
        if candidate.exists() and inside(candidate, bundle_root):
            continue
        cleaned.append(item)
    ambient_candidates = []
    for item in cleaned:
        candidate = Path(item)
        if not candidate.is_dir() or candidate.resolve() == code_root:
            continue
        for top_level in ("utils", "vec_env"):
            if (candidate / top_level).exists():
                ambient_candidates.append(str((candidate / top_level).resolve()))
    if ambient_candidates:
        raise RuntimeError(
            f"ambient repository-local candidate paths forbidden: {sorted(ambient_candidates)}"
        )
    sys.path[:] = [str(code_root), *cleaned]

    import yaml

    spec = importlib.util.spec_from_file_location("standard_mse_cvlm_trainer", trainer)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    parsed = yaml.safe_load(config.read_text())
    if not isinstance(parsed, dict) or "algo_config" not in parsed:
        raise RuntimeError("frozen config did not resolve to expected mapping")

    origins = []
    for name, loaded in sorted(sys.modules.items()):
        if not (name == "standard_mse_cvlm_trainer" or name == "utils" or
                name.startswith("utils.") or name == "vec_env" or name.startswith("vec_env.")):
            continue
        raw_origin = getattr(loaded, "__file__", None)
        if raw_origin is None:
            locations = list(getattr(loaded, "__path__", []))
            if name != "utils" or locations != [str(code_root / "utils")]:
                raise RuntimeError(f"unexpected namespace origin: {name}: {locations}")
            origins.append({
                "module": name,
                "kind": "namespace_package",
                "origin": None,
                "search_locations": locations,
                "manifest_path": None,
            })
            continue
        origin = Path(raw_origin).resolve()
        if not inside(origin, code_root):
            raise RuntimeError(f"ambient repository-local module origin: {name}: {origin}")
        relative = origin.relative_to(bundle_root).as_posix()
        if relative not in entries:
            raise RuntimeError(f"loaded local module absent from manifest: {name}: {relative}")
        if sha(origin) != entries[relative]["sha256"]:
            raise RuntimeError(f"loaded local module hash mismatch: {name}: {relative}")
        origins.append({
            "module": name,
            "kind": "file_module",
            "origin": str(origin),
            "manifest_path": relative,
            "sha256": entries[relative]["sha256"],
            "git_blob": entries[relative]["git_blob"],
        })
    required = {"standard_mse_cvlm_trainer", "utils", "utils.logger", "utils.runners", "utils.utils", "vec_env"}
    observed = {record["module"] for record in origins}
    if not required.issubset(observed):
        raise RuntimeError(f"required imported modules absent: {sorted(required - observed)}")
    payload = {
        "result": "TASK35R_EMPTY_CWD_IMPORT_PASS",
        "cwd": str(Path.cwd()),
        "bundle_root": str(bundle_root),
        "repository_import_root": str(code_root),
        "trainer_sha256": sha(trainer),
        "config_sha256": sha(config),
        "config_top_level_keys": sorted(parsed),
        "module_origins": origins,
        "ambient_repository_fallback": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("TASK35R_EMPTY_CWD_IMPORT_PASS")
    print(f"module_origins={len(origins)} output={output}")


if __name__ == "__main__":
    main()
