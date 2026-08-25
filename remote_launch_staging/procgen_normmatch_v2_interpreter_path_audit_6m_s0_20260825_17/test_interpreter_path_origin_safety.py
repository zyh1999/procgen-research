#!/usr/bin/env python3
"""Required positive and negative tests for Task17 origin classification."""
import importlib.util
import os
import stat
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("task17_origin_safety", HERE / "origin_safety.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
version = sys.version_info
basename = f"python{version.major}{version.minor}.zip"

with tempfile.TemporaryDirectory(prefix="task17_origin_test_") as temporary:
    root = Path(temporary)
    bundle = root / "bundle"
    bundle.mkdir()
    designated = root / "designated"
    designated.mkdir()
    repo = root / "repo"
    repo.mkdir()

    # Task16 positive and four negative protections remain active.
    before = module.snapshot_empty_directory(designated, "before")
    after = module.snapshot_empty_directory(designated, "after", before)
    assert before["device"] == after["device"] and before["inode"] == after["inode"]
    contaminant = designated / "injected_module.py"
    contaminant.write_text("VALUE = 1\n")
    try:
        module.snapshot_empty_directory(designated, "contaminated", before)
    except RuntimeError:
        pass
    else:
        raise AssertionError("importable designated-directory module was accepted")
    contaminant.unlink()
    linked_designated = root / "linked_designated"
    linked_designated.symlink_to(designated, target_is_directory=True)
    try:
        module.snapshot_empty_directory(linked_designated, "symlink")
    except RuntimeError:
        pass
    else:
        raise AssertionError("symlink designated directory was accepted")
    before = module.snapshot_empty_directory(designated, "before_mutation")
    (designated / "late.pyc").write_bytes(b"late")
    try:
        module.snapshot_empty_directory(designated, "after_mutation", before)
    except RuntimeError:
        pass
    else:
        raise AssertionError("post-scan mutation was accepted")
    (designated / "late.pyc").unlink()
    outside = repo / "utils.py"
    outside.write_text("VALUE = 2\n")
    approved = {"bundle": [str(bundle)], "site_packages": [], "stdlib": [],
                "interpreter_zip_candidates": [], "builtin_frozen": ["built-in", "frozen"]}
    try:
        module.classify_origin(str(outside), approved, designated, [repo])
    except RuntimeError:
        pass
    else:
        raise AssertionError("out-of-bundle repository-local module was accepted")

    # Current interpreter dynamically derives legitimate nonexistent candidates.
    current = module.derive_interpreter_zip_candidates()
    nonexistent = [path for path in current if not Path(path).exists()]
    assert nonexistent, current
    for path in nonexistent:
        record = module.inspect_interpreter_zip_candidate(path, current)
        assert record["classification"] == "NONEXISTENT_INTERPRETER_ZIP_CANDIDATE"

    # A safe real standard zip passes only when derived from a supplied interpreter context.
    prefix = root / "prefix"
    stdlib = prefix / "lib" / f"python{version.major}.{version.minor}"
    stdlib.mkdir(parents=True)
    candidates = module.derive_interpreter_zip_candidates(
        base_prefix=prefix,
        base_exec_prefix=prefix,
        version_info=version,
        config_paths={"stdlib": str(stdlib), "platstdlib": str(stdlib)},
    )
    real_zip = prefix / "lib" / basename
    real_zip.write_bytes(b"safe-interpreter-zip-test")
    real_zip.chmod(0o444)
    record = module.inspect_interpreter_zip_candidate(real_zip, candidates)
    assert record["classification"] == "SAFE_INTERPRETER_STANDARD_LIBRARY_ZIP"
    assert record["sha256"] == module.sha256(real_zip)

    # Same basename in an arbitrary temporary directory is rejected.
    arbitrary = root / "arbitrary" / basename
    arbitrary.parent.mkdir()
    arbitrary.write_bytes(b"not-derived")
    arbitrary.chmod(0o444)
    try:
        module.inspect_interpreter_zip_candidate(arbitrary, candidates)
    except RuntimeError:
        pass
    else:
        raise AssertionError("arbitrary same-name zip was accepted")

    # Wrong Python version is rejected even if injected into the candidate set.
    wrong_version = prefix / "lib" / "python999.zip"
    wrong_version.write_bytes(b"wrong-version")
    wrong_version.chmod(0o444)
    try:
        module.inspect_interpreter_zip_candidate(wrong_version, [str(wrong_version.resolve())])
    except RuntimeError:
        pass
    else:
        raise AssertionError("wrong-version zip was accepted")

    # Current-user-writable and symlink candidates are rejected.
    real_zip.chmod(0o644)
    try:
        module.inspect_interpreter_zip_candidate(real_zip, candidates)
    except RuntimeError:
        pass
    else:
        raise AssertionError("current-user-writable zip was accepted")
    real_zip.unlink()
    target = root / "zip_target"
    target.write_bytes(b"target")
    target.chmod(0o444)
    real_zip.symlink_to(target)
    symlink_candidates = [str(real_zip.resolve())]
    try:
        module.inspect_interpreter_zip_candidate(real_zip, symlink_candidates)
    except RuntimeError:
        pass
    else:
        raise AssertionError("symlink zip was accepted")

    # A repository-local module can never resolve from an interpreter zip.
    manifest = {"repository_local_import_closure": ["code/utils/utils.py"]}
    try:
        module.reject_repository_local_zip_origin(
            "utils.utils", "python_standard_library_zip", manifest,
            f"{prefix / 'lib' / basename}/utils/utils.py",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("repository-local module from interpreter zip was accepted")

print("TASK17_INTERPRETER_PATH_ORIGIN_TESTS_PASS")
