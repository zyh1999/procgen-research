#!/usr/bin/env python3
"""Strict origin audit with one explicitly designated empty cwd."""
import hashlib
import json
import os
import site
import stat
import sys
import sysconfig
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_empty_directory(path, phase, expected=None):
    path = Path(path)
    raw = path.lstat()
    if stat.S_ISLNK(raw.st_mode) or not stat.S_ISDIR(raw.st_mode):
        raise RuntimeError(f"designated path is not a real directory: {path}")
    entries = []
    for current, directories, files in os.walk(path, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories + files):
            candidate = current_path / name
            candidate_stat = candidate.lstat()
            entries.append({
                "path": str(candidate.relative_to(path)),
                "mode": stat.S_IMODE(candidate_stat.st_mode),
                "kind": "symlink" if stat.S_ISLNK(candidate_stat.st_mode) else (
                    "directory" if stat.S_ISDIR(candidate_stat.st_mode) else "file"
                ),
            })
    record = {
        "phase": phase,
        "canonical_path": str(path.resolve(strict=True)),
        "device": raw.st_dev,
        "inode": raw.st_ino,
        "uid": raw.st_uid,
        "gid": raw.st_gid,
        "permissions_octal": oct(stat.S_IMODE(raw.st_mode)),
        "ctime_ns": raw.st_ctime_ns,
        "entries": entries,
    }
    if entries:
        raise RuntimeError(f"designated empty directory is contaminated at {phase}: {entries}")
    if expected is not None:
        for key in ("canonical_path", "device", "inode", "uid", "gid"):
            if record[key] != expected[key]:
                raise RuntimeError(f"designated directory identity changed for {key}")
    return record


def approved_origin_roots(bundle_root):
    bundle_root = Path(bundle_root).resolve(strict=True)
    site_roots = set()
    for value in [sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]:
        if value:
            site_roots.add(str(Path(value).resolve()))
    try:
        site_roots.update(str(Path(value).resolve()) for value in site.getsitepackages())
    except AttributeError:
        pass
    stdlib_roots = set()
    for value in [sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")]:
        if value:
            stdlib_roots.add(str(Path(value).resolve()))
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    stdlib_roots.add(str((Path(sys.base_prefix) / "lib" / version).resolve()))
    return {
        "bundle": [str(bundle_root)],
        "site_packages": sorted(site_roots),
        "stdlib": sorted(stdlib_roots),
        "builtin_frozen": ["built-in", "frozen"],
    }


def under(path, roots):
    path = str(Path(path).resolve())
    return any(path == root or path.startswith(root + os.sep) for root in roots)


def classify_origin(origin, approved, designated, forbidden_roots):
    if origin in ("built-in", "frozen"):
        return "builtin/frozen"
    path = str(Path(origin).resolve())
    designated = str(Path(designated).resolve(strict=True))
    if path == designated or path.startswith(designated + os.sep):
        raise RuntimeError(f"module resolved from designated empty directory: {path}")
    forbidden = [str(Path(item).resolve()) for item in forbidden_roots]
    if under(path, forbidden):
        raise RuntimeError(f"module resolved from forbidden source location: {path}")
    if under(path, approved["bundle"]):
        return "verified_bundle"
    if under(path, approved["site_packages"]):
        return "fixed_environment_site_packages"
    if under(path, approved["stdlib"]):
        return "python_standard_library"
    raise RuntimeError(f"module origin is not approved: {path}")


def audit_sys_path(bundle_root, designated, approved, forbidden_roots):
    records = []
    designated = str(Path(designated).resolve(strict=True))
    for raw in sys.path:
        path = str(Path(raw or os.getcwd()).resolve())
        if path == designated:
            classification = "designated_empty_working_directory"
        elif under(path, approved["bundle"]):
            classification = "verified_bundle"
        elif under(path, approved["site_packages"]):
            classification = "fixed_environment_site_packages"
        elif under(path, approved["stdlib"]):
            classification = "python_standard_library"
        elif path.endswith(f"python{sys.version_info.major}{sys.version_info.minor}.zip") and under(str(Path(path).parent), [str(Path(sys.base_prefix) / "lib")]):
            classification = "python_standard_library_zip"
        else:
            forbidden = [str(Path(item).resolve()) for item in forbidden_roots]
            if under(path, forbidden):
                raise RuntimeError(f"forbidden source path on sys.path: {path}")
            raise RuntimeError(f"unapproved sys.path entry: {path}")
        records.append({"raw": raw, "canonical": path, "classification": classification})
    return records


def audit_loaded_modules(bundle_root, manifest, designated, forbidden_roots):
    bundle_root = Path(bundle_root).resolve(strict=True)
    approved = approved_origin_roots(bundle_root)
    files = {record["bundle_path"]: record for record in manifest["files"]}
    origins = []
    bundle_origins = {}
    for name, module in sorted(sys.modules.items()):
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
        if not origin:
            continue
        classification = classify_origin(origin, approved, designated, forbidden_roots)
        record = {"module": name, "origin": origin, "classification": classification}
        if classification == "verified_bundle":
            path = Path(origin).resolve()
            relative = path.relative_to(bundle_root).as_posix()
            if relative not in files:
                raise RuntimeError(f"bundle module absent from manifest: {name} -> {relative}")
            digest = sha256(path)
            if digest != files[relative]["sha256"]:
                raise RuntimeError(f"bundle module hash mismatch: {name} -> {relative}")
            record.update({"bundle_path": relative, "sha256": digest})
            bundle_origins[relative] = name
        origins.append(record)
    expected = set(manifest["repository_local_import_closure"])
    missing = expected - set(bundle_origins)
    if missing:
        raise RuntimeError(f"repository-local closure modules were not imported: {sorted(missing)}")
    return {"approved_roots": approved, "modules": origins, "bundle_origins": bundle_origins}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
