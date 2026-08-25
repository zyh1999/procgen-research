#!/usr/bin/env python3
"""Read-only Task29 proof for the live CPython 3.9 ``__mp_main__`` alias."""
import ast
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

TASK28R_SHA256 = "96da9c8ee7497ce01df3230f6fa0875a81d1175c958c1a5d8276390090c118ad"
MP_INIT_SHA256 = "a5a42976033c7d63ee2740acceef949a3582dcb0e0442845f9717e1be771c68b"
MP_SPAWN_SHA256 = "16ce6d81f8b5ef7228e5500bff04b37bdceb3d7dfc8d6de3ad523598798c43f4"
MP_INIT = Path("/usr/lib64/python3.9/multiprocessing/__init__.py")
MP_SPAWN = Path("/usr/lib64/python3.9/multiprocessing/spawn.py")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stat(value):
    return {
        "device": value.st_dev, "inode": value.st_ino,
        "uid": value.st_uid, "gid": value.st_gid,
        "mode": oct(stat.S_IMODE(value.st_mode)), "size": value.st_size,
        "regular_file": stat.S_ISREG(value.st_mode),
        "symlink": stat.S_ISLNK(value.st_mode),
    }


def _module_snapshot(module):
    values = vars(module)
    spec = values.get("__spec__")
    loader = values.get("__loader__")
    return {
        "object_id": id(module),
        "name": values.get("__name__"),
        "file": values.get("__file__"),
        "package": values.get("__package__"),
        "spec": None if spec is None else {
            "name": getattr(spec, "name", None),
            "origin": getattr(spec, "origin", None),
            "loader_type": type(getattr(spec, "loader", None)).__name__,
        },
        "loader": None if loader is None else {
            "type_module": type(loader).__module__,
            "type_name": type(loader).__name__,
            "name": getattr(loader, "name", None),
            "path": getattr(loader, "path", None),
        },
        "dictionary_keys": sorted(values),
    }


def _source_proof(path, expected_sha):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise RuntimeError("multiprocessing stdlib source SHA mismatch: " + str(path))
    tree = ast.parse(data, filename=str(path))
    compile(tree, str(path), "exec")
    alias_assignments = []
    call_edges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [ast.dump(item, include_attributes=False) for item in node.targets]
            if any("__mp_main__" in value for value in targets):
                alias_assignments.append({
                    "line": node.lineno,
                    "targets": targets,
                    "value": ast.dump(node.value, include_attributes=False),
                })
        if isinstance(node, ast.Call):
            rendered = ast.dump(node.func, include_attributes=False)
            if any(token in rendered for token in ("prepare", "_fixup_main_from_name", "_fixup_main_from_path")):
                call_edges.append({"line": node.lineno, "call": rendered})
    return {
        "path": str(path), "sha256": expected_sha, "size": len(data),
        "lstat": _stat(os.lstat(path)),
        "alias_assignments": alias_assignments, "call_edges": call_edges,
    }


def collect_live_proof(frozen):
    main = sys.modules.get("__main__")
    alias = sys.modules.get("__mp_main__")
    if main is None or alias is None or main is not alias:
        raise RuntimeError("live __main__/__mp_main__ exact object alias not established")
    main_snapshot = _module_snapshot(main)
    alias_snapshot = _module_snapshot(alias)
    if main_snapshot != alias_snapshot:
        raise RuntimeError("same-object module snapshots unexpectedly differ")
    raw = Path(main_snapshot["file"])
    resolved = raw.resolve(strict=True)
    if str(raw) != frozen["raw_path"] or str(resolved) != frozen["resolved_path"]:
        raise RuntimeError("live alias backing is not the exact frozen Task28R probe")
    raw_lstat, resolved_lstat = os.lstat(raw), os.lstat(resolved)
    raw_stat, resolved_stat = os.stat(raw), os.stat(resolved)
    if (stat.S_ISLNK(raw_lstat.st_mode) or stat.S_ISLNK(resolved_lstat.st_mode)
            or not stat.S_ISREG(raw_lstat.st_mode) or not stat.S_ISREG(resolved_lstat.st_mode)
            or not os.path.samefile(raw, resolved)):
        raise RuntimeError("live alias backing raw/resolved file identity mismatch")
    checks = {
        "device": raw_stat.st_dev == resolved_stat.st_dev == frozen["device"],
        "inode": raw_stat.st_ino == resolved_stat.st_ino == frozen["inode"],
        "uid": raw_stat.st_uid == resolved_stat.st_uid == frozen["uid"],
        "gid": raw_stat.st_gid == resolved_stat.st_gid == frozen["gid"],
        "mode": stat.S_IMODE(raw_stat.st_mode) == stat.S_IMODE(resolved_stat.st_mode) == frozen["mode"],
        "size": raw_stat.st_size == resolved_stat.st_size == frozen["size"],
        "sha256": _sha256(raw) == _sha256(resolved) == frozen["sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError("live alias backing frozen identity mismatch: " + str(checks))
    frame = sys._getframe()
    stack = []
    while frame is not None:
        stack.append({"function": frame.f_code.co_name, "file": frame.f_code.co_filename, "line": frame.f_lineno})
        frame = frame.f_back
    init_proof = _source_proof(MP_INIT, MP_INIT_SHA256)
    spawn_proof = _source_proof(MP_SPAWN, MP_SPAWN_SHA256)
    if not any(item["line"] == 37 for item in init_proof["alias_assignments"]):
        raise RuntimeError("CPython import-time alias assignment not found at frozen line")
    spawn_lines = {item["line"] for item in spawn_proof["alias_assignments"]}
    if not {262, 290}.issubset(spawn_lines):
        raise RuntimeError("CPython spawn alias assignments not found at frozen lines")
    multiprocessing_module = sys.modules.get("multiprocessing")
    if multiprocessing_module is None:
        raise RuntimeError("multiprocessing was not naturally imported by frozen production construction")
    return {
        "result": "TASK29_READONLY_LIVE_ALIAS_PROOF_PASS",
        "python": sys.version,
        "implementation": sys.implementation.name,
        "process": {
            "pid": os.getpid(), "name": multiprocessing_module.current_process().name,
            "start_method": multiprocessing_module.get_start_method(allow_none=True),
            "all_start_methods": multiprocessing_module.get_all_start_methods(),
            "argv0": sys.argv[0], "stack_at_observation": stack,
        },
        "module_keys": ["__main__", "__mp_main__"],
        "exact_object_alias": True,
        "main": main_snapshot, "mp_main": alias_snapshot,
        "dictionary_difference": {"only_main": [], "only_mp_main": [], "different_values": []},
        "backing_approved_entry": "APPROVED_EXACT_FROZEN_CLOSURE_PROBE_ALIAS",
        "backing": {
            "raw_path": str(raw), "resolved_path": str(resolved), "samefile": True,
            "raw_lstat": _stat(raw_lstat), "resolved_lstat": _stat(resolved_lstat),
            "raw_stat": _stat(raw_stat), "resolved_stat": _stat(resolved_stat),
            "checks": checks, "sha256": frozen["sha256"],
        },
        "stdlib_provenance": {
            "import_time_alias": init_proof,
            "spawn_bootstrap_alias": spawn_proof,
            "semantic_call_chain": [
                "multiprocessing.spawn.spawn_main", "multiprocessing.spawn._main",
                "multiprocessing.spawn.prepare", "multiprocessing.spawn._fixup_main_from_path",
            ],
        },
    }


def _atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_task28r():
    path = Path(os.environ["TASK29_TASK28R_EXTENSION"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != TASK28R_SHA256:
        raise RuntimeError("Task28R extension identity mismatch")
    namespace = {"__file__": str(path), "__name__": "task28r_exact_probe_frozen"}
    exec(compile(data, str(path), "exec"), namespace)
    return namespace


def install(namespace):
    task28r = _load_task28r()
    task28r["install"](namespace)
    frozen_audit = namespace["audit_loaded_modules"]

    def audit_loaded_modules(bundle_root, manifest, designated, forbidden_roots):
        proof = collect_live_proof(task28r["FROZEN"])
        _atomic_write(os.environ["TASK29_READONLY_ALIAS_LEDGER"], proof)
        return frozen_audit(bundle_root, manifest, designated, forbidden_roots)

    namespace["audit_loaded_modules"] = audit_loaded_modules
    return namespace
