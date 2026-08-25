#!/usr/bin/env python3
"""Task28R exact same-file alias for the frozen closure probe only."""
import hashlib
import importlib.machinery
import json
import os
import stat
import sys
from pathlib import Path

CLASSIFICATION = "APPROVED_EXACT_FROZEN_CLOSURE_PROBE_ALIAS"
BASE_CLASSIFIER_SHA256 = "f80de2abbcbce29e7a57ef456156c86636798c4e1ea37171922b3b466b6790fc"
SCIENCE_LAUNCHER_SHA256 = "ec60864aaa9940fd61eb1391008b50f2f48402eeae8f78c91c0b0c1fc313a398"
FROZEN = {
    "frozen_git_commit": "baab71b243b0913ada24104bcca6788121c0b5ad",
    "repository_relative_path": "remote_launch_staging/procgen_normmatch_v2_torch_pseudo_origin_nonreentrant_closure_20260825_23/runtime_closure_probe_task23.py",
    "git_mode": "100644",
    "git_blob": "e4c63952f24b732f05cb52224e6673883450d659",
    "sha256": "c3529cb171306d7b3b0517974a682ddffb91d65dc015c102b1e84658e9eeb1f5",
    "basename": "runtime_closure_probe_task23.py",
    "raw_path": "/scratch/h99859yz/procgen_normmatch_v2_torch_pseudo_origin_nonreentrant_closure_20260825_23/tools/runtime_closure_probe_task23.py",
    "resolved_path": "/net/scratch/h99859yz/procgen_normmatch_v2_torch_pseudo_origin_nonreentrant_closure_20260825_23/tools/runtime_closure_probe_task23.py",
    "module_key": "__main__",
    "loader_module": "_frozen_importlib_external",
    "loader_name": "SourceFileLoader",
    "package": None,
    "spec": None,
    "device": 3592384858,
    "inode": 144122242274496637,
    "uid": 778916,
    "gid": 10049,
    "mode": 0o644,
    "size": 4558,
}


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _git_blob(data):
    header = ("blob " + str(len(data)) + "\0").encode()
    return hashlib.sha1(header + data).hexdigest()


def _stat_record(value):
    return {
        "device": value.st_dev, "inode": value.st_ino,
        "uid": value.st_uid, "gid": value.st_gid,
        "mode": oct(stat.S_IMODE(value.st_mode)), "size": value.st_size,
        "regular_file": stat.S_ISREG(value.st_mode),
        "symlink": stat.S_ISLNK(value.st_mode),
    }


def _read_fd(descriptor, size):
    chunks, offset = [], 0
    while offset < size:
        chunk = os.pread(descriptor, size - offset, offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    data = b"".join(chunks)
    if len(data) != size:
        raise RuntimeError("exact probe descriptor short read")
    return data


def _assert_identity(value, expected, label):
    checks = {
        "device": value.st_dev == expected["device"],
        "inode": value.st_ino == expected["inode"],
        "uid": value.st_uid == expected["uid"],
        "gid": value.st_gid == expected["gid"],
        "mode": stat.S_IMODE(value.st_mode) == expected["mode"],
        "size": value.st_size == expected["size"],
        "regular_file": stat.S_ISREG(value.st_mode),
        "not_symlink": not stat.S_ISLNK(value.st_mode),
    }
    if not all(checks.values()):
        raise RuntimeError("exact frozen probe " + label + " identity mismatch: " + str(checks))
    return checks


def _load_frozen_base():
    raw = os.environ.get("TASK28R_CLASSIFIER_BASE")
    if raw:
        path = Path(raw)
    else:
        path = (Path(__file__).resolve().parents[1]
                / "procgen_normmatch_v2_torch_class_attribute_pseudo_origin_20260825_25/class_attribute_classifier.py")
    data = path.read_bytes()
    if _sha256_bytes(data) != BASE_CLASSIFIER_SHA256:
        raise RuntimeError("Task25 classifier base identity mismatch")
    namespace = {"__file__": str(path), "__name__": "task25_classifier_frozen"}
    exec(compile(data, str(path), "exec"), namespace)
    return namespace


class ExactFrozenProbeAlias:
    """One-process identity lease for exactly the frozen Task23 probe."""
    def __init__(self, expected=None, process_role=None, science_launcher=None,
                 after_open_hook=None, after_scan_hook=None):
        self.expected = dict(FROZEN if expected is None else expected)
        self.process_role = process_role if process_role is not None else os.environ.get("TASK28R_PROCESS_ROLE")
        launcher = science_launcher if science_launcher is not None else os.environ.get("TASK28R_FROZEN_SCIENCE_LAUNCHER")
        self.science_launcher = None if launcher is None else Path(launcher)
        self.after_open_hook = after_open_hook
        self.after_scan_hook = after_scan_hook
        self.descriptor = None
        self.module = None
        self.record = None
        self.count = 0

    def is_candidate(self, origin):
        return str(origin) == self.expected["raw_path"]

    def _module_for_origin(self, origin):
        candidates = []
        for name, module in sys.modules.items():
            values = vars(module) if hasattr(module, "__dict__") else {}
            if values.get("__file__") == origin:
                candidates.append((name, module))
        if len(candidates) != 1:
            raise RuntimeError("exact probe origin must map to exactly one loaded module")
        return candidates[0]

    def approve(self, origin):
        if self.count != 0:
            raise RuntimeError("exact frozen closure probe alias approved more than once")
        expected = self.expected
        if self.process_role != "closure-audit-entrypoint":
            raise RuntimeError("exact probe is not running as the closure-audit entrypoint")
        if str(origin) != expected["raw_path"] or Path(origin).name != expected["basename"]:
            raise RuntimeError("reported exact probe raw spelling/basename mismatch")
        name, module = self._module_for_origin(origin)
        values = vars(module)
        loader = values.get("__loader__")
        if name != expected["module_key"] or sys.modules.get(name) is not module:
            raise RuntimeError("exact probe module key/object mismatch")
        if values.get("__spec__") is not expected["spec"] or values.get("__package__") is not expected["package"]:
            raise RuntimeError("exact probe spec/package mismatch")
        if (type(loader).__module__ != expected["loader_module"]
                or type(loader).__name__ != expected["loader_name"]
                or getattr(loader, "name", None) != expected["module_key"]
                or getattr(loader, "path", None) != expected["raw_path"]):
            raise RuntimeError("exact probe loader identity mismatch")
        if not sys.argv or sys.argv[0] != expected["raw_path"] or not os.path.samefile(sys.argv[0], origin):
            raise RuntimeError("exact probe was not loaded as the process entrypoint")

        raw_path = Path(origin)
        raw_lstat = os.lstat(raw_path)
        if stat.S_ISLNK(raw_lstat.st_mode) or not stat.S_ISREG(raw_lstat.st_mode):
            raise RuntimeError("exact probe raw final component is not a regular non-symlink")
        resolved = raw_path.resolve(strict=True)
        if str(resolved) != expected["resolved_path"]:
            raise RuntimeError("exact probe resolved spelling mismatch")
        resolved_lstat = os.lstat(resolved)
        if stat.S_ISLNK(resolved_lstat.st_mode) or not stat.S_ISREG(resolved_lstat.st_mode):
            raise RuntimeError("exact probe resolved final component is not a regular non-symlink")
        samefile = os.path.samefile(raw_path, resolved)
        raw_stat, resolved_stat = os.stat(raw_path), os.stat(resolved)
        if not samefile or (raw_stat.st_dev, raw_stat.st_ino) != (resolved_stat.st_dev, resolved_stat.st_ino):
            raise RuntimeError("exact probe raw/resolved same-file identity mismatch")
        raw_checks = _assert_identity(raw_stat, expected, "raw")
        resolved_checks = _assert_identity(resolved_stat, expected, "resolved")

        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise RuntimeError("O_NOFOLLOW unavailable for exact probe identity")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow
        descriptor = os.open(resolved, flags)
        try:
            opened = os.fstat(descriptor)
            opened_checks = _assert_identity(opened, expected, "opened-fd")
            data = _read_fd(descriptor, opened.st_size)
            sha = _sha256_bytes(data)
            blob = _git_blob(data)
            if sha != expected["sha256"] or blob != expected["git_blob"]:
                raise RuntimeError("exact probe SHA256/Git blob mismatch")
            if self.after_open_hook is not None:
                self.after_open_hook(raw_path, resolved, descriptor)
            self.descriptor = descriptor
            descriptor = None
        finally:
            if descriptor is not None:
                os.close(descriptor)

        self.module = module
        self.count = 1
        self.record = {
            "classification": CLASSIFICATION,
            "frozen_git_commit": expected["frozen_git_commit"],
            "repository_relative_path": expected["repository_relative_path"],
            "git_mode": expected["git_mode"],
            "git_blob": expected["git_blob"],
            "module_key": name,
            "module_object_id": id(module),
            "spec": None, "package": None,
            "loader": {
                "module": type(loader).__module__, "name": type(loader).__name__,
                "loader_name": loader.name, "loader_path": loader.path,
            },
            "entrypoint": {"argv0": sys.argv[0], "role": self.process_role},
            "raw_path": str(raw_path), "resolved_path": str(resolved),
            "samefile": samefile,
            "raw_lstat": _stat_record(raw_lstat), "raw_stat": _stat_record(raw_stat),
            "resolved_lstat": _stat_record(resolved_lstat), "resolved_stat": _stat_record(resolved_stat),
            "raw_checks": raw_checks, "resolved_checks": resolved_checks,
            "o_nofollow": {"available": True, "applied": True, "value": nofollow},
            "opened_fd": _stat_record(opened), "opened_fd_checks": opened_checks,
            "pre_scan_sha256": sha, "pre_scan_git_blob": blob,
            "loaded_by_closure_audit_entrypoint_not_science": True,
        }
        return CLASSIFICATION

    def finalize(self, bundle_root, manifest):
        if self.count != 1 or self.descriptor is None or self.record is None:
            raise RuntimeError("exact frozen closure probe was not approved exactly once")
        if self.after_scan_hook is not None:
            self.after_scan_hook(Path(self.record["raw_path"]), Path(self.record["resolved_path"]), self.descriptor)
        expected = self.expected
        fd_after = os.fstat(self.descriptor)
        fd_checks = _assert_identity(fd_after, expected, "post-scan-fd")
        fd_data = _read_fd(self.descriptor, fd_after.st_size)
        raw_path, resolved = Path(self.record["raw_path"]), Path(self.record["resolved_path"])
        raw_lstat, resolved_lstat = os.lstat(raw_path), os.lstat(resolved)
        if stat.S_ISLNK(raw_lstat.st_mode) or stat.S_ISLNK(resolved_lstat.st_mode):
            raise RuntimeError("exact probe became a symlink during origin scan")
        if not os.path.samefile(raw_path, resolved):
            raise RuntimeError("exact probe raw/resolved samefile changed during origin scan")
        path_stat = os.stat(resolved)
        path_checks = _assert_identity(path_stat, expected, "post-scan-path")
        path_data = resolved.read_bytes()
        fd_sha, path_sha = _sha256_bytes(fd_data), _sha256_bytes(path_data)
        if fd_sha != expected["sha256"] or path_sha != expected["sha256"] or _git_blob(path_data) != expected["git_blob"]:
            raise RuntimeError("exact probe post-scan SHA/Git blob changed")
        values = vars(self.module)
        if (sys.modules.get(expected["module_key"]) is not self.module
                or values.get("__file__") != expected["raw_path"]
                or values.get("__spec__") is not None or values.get("__package__") is not None):
            raise RuntimeError("exact probe module identity changed during scan")

        bundle_root = Path(bundle_root).resolve(strict=True)
        manifest_text = json.dumps(manifest, sort_keys=True)
        forbidden_tokens = (expected["basename"], expected["repository_relative_path"],
                            expected["sha256"], expected["git_blob"])
        if any(token in manifest_text for token in forbidden_tokens):
            raise RuntimeError("exact closure probe leaked into bundle manifest/import closure")
        if resolved == bundle_root or bundle_root in resolved.parents:
            raise RuntimeError("exact closure probe is inside the scientific bundle")
        probe_modules = [
            name for name, module in sys.modules.items()
            if hasattr(module, "__dict__") and vars(module).get("__file__") == expected["raw_path"]
        ]
        if probe_modules != [expected["module_key"]]:
            raise RuntimeError("exact closure probe leaked beyond the audit entrypoint module")

        if self.science_launcher is None:
            raise RuntimeError("missing frozen formal science launcher for exclusion proof")
        launcher_data = self.science_launcher.read_bytes()
        if _sha256_bytes(launcher_data) != SCIENCE_LAUNCHER_SHA256:
            raise RuntimeError("frozen formal science launcher identity mismatch")
        launcher_text = launcher_data.decode()
        if any(token in launcher_text for token in forbidden_tokens + (expected["raw_path"], expected["resolved_path"])):
            raise RuntimeError("exact closure probe leaked into formal scientific launcher")

        self.record.update({
            "post_scan_fd": _stat_record(fd_after), "post_scan_fd_checks": fd_checks,
            "post_scan_path": _stat_record(path_stat), "post_scan_path_checks": path_checks,
            "post_scan_fd_sha256": fd_sha, "post_scan_path_sha256": path_sha,
            "post_scan_git_blob": _git_blob(path_data),
            "fd_path_identity_and_sha_stable": True,
            "bundle_manifest_probe_absence": True,
            "bundle_import_closure_probe_absence": True,
            "formal_science_launcher_sha256": SCIENCE_LAUNCHER_SHA256,
            "formal_scientific_process_probe_absence": True,
        })
        return dict(self.record)

    def close(self):
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


_base_ns = _load_frozen_base()
_frozen_install = _base_ns["install"]


def install(namespace):
    frozen_classify = namespace["_base_classify_origin"]
    session = ExactFrozenProbeAlias()

    def exact_classify(origin, approved, designated, forbidden_roots):
        try:
            return frozen_classify(origin, approved, designated, forbidden_roots)
        except RuntimeError:
            if not session.is_candidate(origin):
                raise
            return session.approve(origin)

    namespace["_base_classify_origin"] = exact_classify
    _frozen_install(namespace)
    frozen_audit = namespace["audit_loaded_modules"]

    def audit_loaded_modules(bundle_root, manifest, designated, forbidden_roots):
        try:
            result = frozen_audit(bundle_root, manifest, designated, forbidden_roots)
            result["exact_frozen_closure_probe_aliases"] = [session.finalize(bundle_root, manifest)]
            return result
        finally:
            session.close()

    namespace["audit_loaded_modules"] = audit_loaded_modules
    namespace["validate_exact_frozen_closure_probe_alias"] = session.approve
    return namespace
