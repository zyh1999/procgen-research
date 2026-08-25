#!/usr/bin/env python3
"""Positive and negative tests for the bounded designated-cwd exception."""
import importlib.util
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("origin_safety", HERE / "origin_safety.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory(prefix="task16_origin_test_") as temporary:
    root = Path(temporary)
    bundle = root / "bundle"
    bundle.mkdir()
    designated = root / "designated"
    designated.mkdir()
    repo = root / "repo"
    repo.mkdir()

    # Positive: the single designated directory is empty and supplies no module.
    before = module.snapshot_empty_directory(designated, "before")
    after = module.snapshot_empty_directory(designated, "after", before)
    assert before["device"] == after["device"] and before["inode"] == after["inode"]

    # Negative: an importable module in the designated directory is contamination.
    contaminant = designated / "injected_module.py"
    contaminant.write_text("VALUE = 1\n")
    try:
        module.snapshot_empty_directory(designated, "contaminated", before)
    except RuntimeError:
        pass
    else:
        raise AssertionError("importable designated-directory module was accepted")
    contaminant.unlink()

    # Negative: a symlink cannot be the designated directory.
    linked = root / "linked"
    linked.symlink_to(designated, target_is_directory=True)
    try:
        module.snapshot_empty_directory(linked, "symlink")
    except RuntimeError:
        pass
    else:
        raise AssertionError("symlink designated directory was accepted")

    # Negative: a post-scan file addition is detected.
    before = module.snapshot_empty_directory(designated, "before_mutation")
    (designated / "late.pyc").write_bytes(b"late")
    try:
        module.snapshot_empty_directory(designated, "after_mutation", before)
    except RuntimeError:
        pass
    else:
        raise AssertionError("post-scan mutation was accepted")
    (designated / "late.pyc").unlink()

    # Negative: repository-local origin outside the verified bundle is rejected.
    outside = repo / "utils.py"
    outside.write_text("VALUE = 2\n")
    approved = {"bundle": [str(bundle)], "site_packages": [], "stdlib": [], "builtin_frozen": ["built-in", "frozen"]}
    try:
        module.classify_origin(str(outside), approved, designated, [repo])
    except RuntimeError:
        pass
    else:
        raise AssertionError("out-of-bundle repository-local module was accepted")

print("TASK16_ORIGIN_SAFETY_TESTS_PASS")
