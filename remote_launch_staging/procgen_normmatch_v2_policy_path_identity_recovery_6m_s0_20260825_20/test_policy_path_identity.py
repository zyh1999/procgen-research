#!/usr/bin/env python3
"""Positive and negative identity gates for Task20 storage aliases."""
import importlib.util
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK18 = HERE.parent / "procgen_normmatch_v2_torch_generated_origin_audit_6m_s0_20260825_18"
TASK17 = HERE.parent / "procgen_normmatch_v2_interpreter_path_audit_6m_s0_20260825_17"
SOURCE_POLICY = TASK18 / "origin_safety.py"
BASELINE = TASK18 / "provenance/reproduction_1.json"

spec = importlib.util.spec_from_file_location("task20_policy_support", HERE / "policy_namespace_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def identity(path):
    value = path.stat(follow_symlinks=False)
    return {
        "device": value.st_dev,
        "inode": value.st_ino,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "mode": value.st_mode & 0o777,
        "size": value.st_size,
        "sha256": support.EXPECTED_POLICY_SHA256,
    }


old_env = dict(os.environ)
try:
    os.environ["TASK17_ORIGIN_SAFETY_BASE"] = str((TASK17 / "origin_safety.py").resolve())
    os.environ["TORCH_GENERATED_PROVENANCE_BASELINE"] = str(BASELINE.resolve())
    with tempfile.TemporaryDirectory(prefix="task20_identity_") as temporary:
        root = Path(temporary)
        real_parent = root / "real"
        real_parent.mkdir()
        policy = real_parent / "origin_safety.py"
        policy.write_bytes(SOURCE_POLICY.read_bytes())
        policy.chmod(0o644)
        expected = identity(policy)
        alias_parent = root / "storage_alias"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        alias = alias_parent / policy.name
        empty = root / "empty"
        empty.mkdir()

        # Positive: parent spelling differs, final target is the exact same inode.
        os.environ[support.POLICY_PATH_ENV] = str(alias)
        namespace, ledger = support.load_explicit_policy(empty, expected_identity=expected)
        assert ledger["raw_path"] != ledger["resolved_path"]
        assert ledger["pre"]["samefile"] and ledger["post"]["samefile"]
        assert ledger["opened_fd"]["device"] == expected["device"]
        assert ledger["opened_fd"]["inode"] == expected["inode"]
        assert ledger["pre_sha256"] == ledger["post_exec_fd_sha256"] == ledger["post_exec_path_sha256"]
        assert namespace["snapshot_empty_directory"](empty, "task20")["entries"] == []

        # Same bytes in another inode must fail even though size/mode/SHA match.
        duplicate = root / "duplicate.py"
        duplicate.write_bytes(policy.read_bytes())
        duplicate.chmod(0o644)
        os.environ[support.POLICY_PATH_ENV] = str(duplicate)
        try:
            support.load_explicit_policy(empty, expected_identity=expected)
        except RuntimeError:
            pass
        else:
            raise AssertionError("byte-identical different file was accepted")

        # Final-component symlink must fail; parent alias remains the only allowed alias.
        final_link = root / "final_link.py"
        final_link.symlink_to(policy)
        os.environ[support.POLICY_PATH_ENV] = str(final_link)
        try:
            support.load_explicit_policy(empty, expected_identity=expected)
        except RuntimeError:
            pass
        else:
            raise AssertionError("final file symlink was accepted")

        # Missing path and mismatched identity fields fail.
        os.environ[support.POLICY_PATH_ENV] = str(root / "missing.py")
        try:
            support.load_explicit_policy(empty, expected_identity=expected)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing path was accepted")
        os.environ[support.POLICY_PATH_ENV] = str(alias)
        for key, wrong in (
            ("device", expected["device"] + 1), ("inode", expected["inode"] + 1),
            ("uid", expected["uid"] + 1), ("mode", 0o600),
            ("size", expected["size"] + 1), ("sha256", "0" * 64),
        ):
            changed = dict(expected)
            changed[key] = wrong
            try:
                support.load_explicit_policy(empty, expected_identity=changed)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"mismatched {key} was accepted")

        # Replacement after resolve but before open, and after execution, must fail.
        replacement = root / "replacement.py"
        replacement.write_bytes(policy.read_bytes())
        replacement.chmod(0o644)

        def replace_after_resolve(raw, resolved):
            os.replace(replacement, resolved)

        try:
            support.load_explicit_policy(
                empty, expected_identity=expected, _after_resolve_hook=replace_after_resolve
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("post-resolve replacement was accepted")

        # Restore the expected inode identity for the independent post-exec case.
        policy.unlink()
        policy.write_bytes(SOURCE_POLICY.read_bytes())
        policy.chmod(0o644)
        expected = identity(policy)
        post_replacement = root / "post_replacement.py"
        post_replacement.write_bytes(policy.read_bytes())
        post_replacement.chmod(0o644)

        def replace_after_exec(raw, resolved):
            os.replace(post_replacement, resolved)

        try:
            support.load_explicit_policy(
                empty, expected_identity=expected, _after_exec_hook=replace_after_exec
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("post-exec replacement was accepted")
finally:
    os.environ.clear()
    os.environ.update(old_env)

print("TASK20_POLICY_PATH_IDENTITY_TESTS_PASS")
