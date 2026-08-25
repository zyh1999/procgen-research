#!/usr/bin/env python3
"""Load the frozen Task18 policy from one explicit, identity-checked path."""
import hashlib
import json
import os
import stat
from pathlib import Path

POLICY_PATH_ENV = "NORMMATCH_V2_ORIGIN_POLICY_PATH"
EXPECTED_POLICY_SHA256 = "889b914a792132358f90572d5c4f16b561c60bf0615eb8492c9ecf781f601fc1"
EXPECTED_POLICY_MODE = 0o644


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_explicit_policy(designated_empty):
    raw_policy_path = os.environ.get(POLICY_PATH_ENV)
    if not raw_policy_path:
        raise RuntimeError("missing explicit origin-policy path")
    raw_path = Path(raw_policy_path)
    if not raw_path.is_absolute():
        raise RuntimeError("origin-policy path must be absolute")
    raw = raw_path.lstat()
    if stat.S_ISLNK(raw.st_mode) or not stat.S_ISREG(raw.st_mode):
        raise RuntimeError("origin-policy path must be a regular non-symlink file")
    policy_path = raw_path.resolve(strict=True)
    if policy_path != raw_path:
        raise RuntimeError("origin-policy path must already be canonical")
    if raw.st_uid != os.geteuid():
        raise RuntimeError("origin-policy owner mismatch")
    mode = stat.S_IMODE(raw.st_mode)
    if mode != EXPECTED_POLICY_MODE:
        raise RuntimeError(f"origin-policy mode mismatch: {oct(mode)}")
    designated = Path(designated_empty).resolve(strict=True)
    if policy_path == designated or designated in policy_path.parents:
        raise RuntimeError("origin-policy path is inside designated empty cwd")
    actual_hash = _sha256(policy_path)
    if actual_hash != EXPECTED_POLICY_SHA256:
        raise RuntimeError("origin-policy SHA256 mismatch")
    return policy_path, {
        "result": "EXPLICIT_ORIGIN_POLICY_PATH_PASS",
        "environment_variable": POLICY_PATH_ENV,
        "raw_path": raw_policy_path,
        "canonical_path": str(policy_path),
        "device": raw.st_dev,
        "inode": raw.st_ino,
        "uid": raw.st_uid,
        "gid": raw.st_gid,
        "mode": oct(mode),
        "size": raw.st_size,
        "sha256": actual_hash,
        "designated_empty_cwd": str(designated),
        "outside_designated_empty_cwd": True,
    }


def load_explicit_policy(designated_empty, ledger_path=None, expected_ledger_path=None):
    policy_path, ledger = resolve_explicit_policy(designated_empty)
    if expected_ledger_path is not None:
        expected = json.loads(Path(expected_ledger_path).read_text())
        for key in ("canonical_path", "device", "inode", "uid", "gid", "mode", "size", "sha256"):
            if ledger[key] != expected[key]:
                raise RuntimeError(f"origin-policy identity changed after prestart: {key}")
        ledger["prestart_identity_revalidated"] = True
    if ledger_path is not None:
        _write_json(ledger_path, ledger)
    namespace = {"__name__": "task18_origin_policy", "__file__": str(policy_path)}
    exec(compile(policy_path.read_bytes(), str(policy_path), "exec"), namespace)
    return namespace, ledger
