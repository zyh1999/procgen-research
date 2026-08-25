#!/usr/bin/env python3
"""Load the frozen Task18 policy through a same-file verified storage alias."""
import hashlib
import json
import os
import stat
from pathlib import Path

POLICY_PATH_ENV = "NORMMATCH_V2_ORIGIN_POLICY_PATH"
EXPECTED_POLICY_SHA256 = "889b914a792132358f90572d5c4f16b561c60bf0615eb8492c9ecf781f601fc1"
FROZEN_REMOTE_IDENTITY = {
    "device": 3592384858,
    "inode": 144122242006038476,
    "uid": 778916,
    "gid": 10049,
    "mode": 0o644,
    "size": 13605,
    "sha256": EXPECTED_POLICY_SHA256,
}


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _stat_record(value):
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "mode": oct(stat.S_IMODE(value.st_mode)),
        "size": value.st_size,
        "regular_file": stat.S_ISREG(value.st_mode),
        "symlink": stat.S_ISLNK(value.st_mode),
    }


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_descriptor(descriptor, size):
    chunks = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, size - offset, offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    data = b"".join(chunks)
    if len(data) != size:
        raise RuntimeError("origin-policy fd produced a short read")
    return data


def _assert_frozen_identity(value, expected, label):
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
        raise RuntimeError(f"origin-policy {label} identity mismatch: {checks}")
    return checks


def _resolve_and_validate(designated_empty, expected_identity):
    raw_policy_path = os.environ.get(POLICY_PATH_ENV)
    if not raw_policy_path:
        raise RuntimeError("missing explicit origin-policy path")
    raw_path = Path(raw_policy_path)
    if not raw_path.is_absolute():
        raise RuntimeError("origin-policy path must be absolute")
    raw_lstat = raw_path.lstat()
    if stat.S_ISLNK(raw_lstat.st_mode) or not stat.S_ISREG(raw_lstat.st_mode):
        raise RuntimeError("origin-policy final path must be a regular non-symlink file")
    resolved_path = raw_path.resolve(strict=True)
    resolved_lstat = resolved_path.lstat()
    if stat.S_ISLNK(resolved_lstat.st_mode) or not stat.S_ISREG(resolved_lstat.st_mode):
        raise RuntimeError("resolved origin-policy target must be a regular non-symlink file")
    samefile = os.path.samefile(raw_path, resolved_path)
    if not samefile:
        raise RuntimeError("raw and resolved origin-policy paths are not the same file")
    raw_stat = raw_path.stat(follow_symlinks=False)
    resolved_stat = resolved_path.stat(follow_symlinks=False)
    if (raw_stat.st_dev, raw_stat.st_ino) != (resolved_stat.st_dev, resolved_stat.st_ino):
        raise RuntimeError("raw and resolved origin-policy device/inode differ")
    raw_checks = _assert_frozen_identity(raw_stat, expected_identity, "raw")
    resolved_checks = _assert_frozen_identity(resolved_stat, expected_identity, "resolved")
    designated = Path(designated_empty).resolve(strict=True)
    if resolved_path == designated or designated in resolved_path.parents:
        raise RuntimeError("origin-policy path is inside designated empty cwd")
    return raw_policy_path, raw_path, resolved_path, designated, {
        "raw_lstat": _stat_record(raw_lstat),
        "raw_stat": _stat_record(raw_stat),
        "resolved_lstat": _stat_record(resolved_lstat),
        "resolved_stat": _stat_record(resolved_stat),
        "raw_checks": raw_checks,
        "resolved_checks": resolved_checks,
        "samefile": samefile,
    }


def load_explicit_policy(designated_empty, ledger_path=None, expected_ledger_path=None,
                         expected_identity=None, _after_resolve_hook=None,
                         _after_exec_hook=None):
    expected = dict(FROZEN_REMOTE_IDENTITY if expected_identity is None else expected_identity)
    raw_spelling, raw_path, resolved_path, designated, pre = _resolve_and_validate(
        designated_empty, expected
    )
    if _after_resolve_hook is not None:
        _after_resolve_hook(raw_path, resolved_path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved_path, flags)
    try:
        opened_stat = os.fstat(descriptor)
        opened_checks = _assert_frozen_identity(opened_stat, expected, "opened-fd")
        policy_bytes = _read_descriptor(descriptor, opened_stat.st_size)
        pre_sha = _sha256_bytes(policy_bytes)
        if pre_sha != expected["sha256"] or pre_sha != EXPECTED_POLICY_SHA256:
            raise RuntimeError("origin-policy pre-exec SHA256 mismatch")
        namespace = {"__name__": "task18_origin_policy", "__file__": str(resolved_path)}
        exec(compile(policy_bytes, str(resolved_path), "exec"), namespace)
        if _after_exec_hook is not None:
            _after_exec_hook(raw_path, resolved_path)
        fd_after = os.fstat(descriptor)
        fd_after_checks = _assert_frozen_identity(fd_after, expected, "post-exec-fd")
        post_fd_sha = _sha256_bytes(_read_descriptor(descriptor, fd_after.st_size))
        post_raw_spelling, post_raw, post_resolved, _, post = _resolve_and_validate(
            designated_empty, expected
        )
        if post_raw_spelling != raw_spelling or post_resolved != resolved_path:
            raise RuntimeError("origin-policy path resolution changed after execution")
        with open(post_resolved, "rb") as handle:
            post_path_sha = _sha256_bytes(handle.read())
        if pre_sha != post_fd_sha or pre_sha != post_path_sha:
            raise RuntimeError("origin-policy SHA256 changed after execution")
        ledger = {
            "result": "ORIGIN_POLICY_PATH_IDENTITY_PASS",
            "environment_variable": POLICY_PATH_ENV,
            "raw_path": raw_spelling,
            "resolved_path": str(resolved_path),
            "designated_empty_cwd": str(designated),
            "pre": pre,
            "opened_fd": _stat_record(opened_stat),
            "opened_fd_checks": opened_checks,
            "pre_sha256": pre_sha,
            "post": post,
            "post_exec_fd": _stat_record(fd_after),
            "post_exec_fd_checks": fd_after_checks,
            "post_exec_fd_sha256": post_fd_sha,
            "post_exec_path_sha256": post_path_sha,
            "identity_and_sha_revalidated_after_exec": True,
        }
        if expected_ledger_path is not None:
            previous = json.loads(Path(expected_ledger_path).read_text())
            for key in ("raw_path", "resolved_path", "pre_sha256"):
                if ledger[key] != previous[key]:
                    raise RuntimeError(f"origin-policy identity changed after prestart: {key}")
            for key in ("device", "inode", "uid", "gid", "mode", "size"):
                if ledger["opened_fd"][key] != previous["opened_fd"][key]:
                    raise RuntimeError(f"origin-policy opened identity changed after prestart: {key}")
            ledger["prestart_identity_revalidated"] = True
        if ledger_path is not None:
            _write_json(ledger_path, ledger)
        return namespace, ledger
    finally:
        os.close(descriptor)
