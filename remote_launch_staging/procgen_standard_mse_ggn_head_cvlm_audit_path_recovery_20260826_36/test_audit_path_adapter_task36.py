#!/usr/bin/env python3
"""Negative and positive path-identity regressions for the Task36 adapter."""

import copy
import hashlib
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("adapter", HERE / "audit_path_adapter_task36.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class AdapterIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "bundle"
        self.target = self.root / "code" / "trainer.py"
        self.target.parent.mkdir(parents=True)
        self.data = b"print('frozen')\n"
        self.target.write_bytes(self.data)
        os.chmod(self.target, 0o755)
        self.repo = "repo/frozen/trainer.py"
        self.sha = hashlib.sha256(self.data).hexdigest()
        self.blob = adapter.git_blob_bytes(self.data)
        self.entry = {
            "repository_path": self.repo,
            "bundle_path": "code/trainer.py",
            "sha256": self.sha,
            "git_blob": self.blob,
            "size": len(self.data),
            "mode": "100755",
        }

    def tearDown(self):
        self.temp.cleanup()

    def resolve(self, manifest=None, **overrides):
        values = {
            "bundle_root": self.root,
            "manifest": manifest if manifest is not None else {"files": [self.entry]},
            "repository_path": self.repo,
            "bundle_path": "code/trainer.py",
            "expected_sha": self.sha,
            "expected_blob": self.blob,
            "expected_size": len(self.data),
            "expected_mode": "100755",
        }
        values.update(overrides)
        return adapter.resolve_manifest_member(**values)

    def assert_rejected(self, manifest=None, **overrides):
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            self.resolve(manifest=manifest, **overrides)

    def test_positive_exact_manifest_identity(self):
        path, entry, identity = self.resolve()
        self.assertEqual(path, self.target)
        self.assertEqual(entry, self.entry)
        self.assertEqual(identity["sha256"], self.sha)

    def test_old_frozen_false_path_rejected(self):
        self.assert_rejected(bundle_path="frozen/trainer.py")

    def test_symlink_and_escape_rejected(self):
        real = self.root / "code" / "real.py"
        self.target.rename(real)
        self.target.symlink_to(real)
        self.assert_rejected()
        escaped = copy.deepcopy(self.entry)
        escaped["bundle_path"] = "../outside.py"
        self.assert_rejected(manifest={"files": [escaped]}, bundle_path="../outside.py")

    def test_same_bytes_wrong_manifest_identity_rejected(self):
        wrong = copy.deepcopy(self.entry)
        wrong["repository_path"] = "repo/other/trainer.py"
        self.assert_rejected(manifest={"files": [wrong]})

    def test_wrong_blob_hash_size_mode_rejected(self):
        for key, value in (
            ("git_blob", "0" * 40),
            ("sha256", "0" * 64),
            ("size", len(self.data) + 1),
            ("mode", "100644"),
        ):
            wrong = copy.deepcopy(self.entry)
            wrong[key] = value
            self.assert_rejected(manifest={"files": [wrong]})

    def test_missing_duplicate_and_ambient_fallback_rejected(self):
        self.assert_rejected(manifest={"files": []})
        self.assert_rejected(manifest={"files": [self.entry, copy.deepcopy(self.entry)]})
        # The verified bytes remain present on disk, but absence from the
        # manifest must still reject rather than falling back to that file.
        self.assertTrue(self.target.is_file())
        self.assert_rejected(manifest={"files": []})

    def test_audit_math_mutation_rejected(self):
        audit = Path(self.temp.name) / "audit.py"
        audit.write_bytes(b"modified numerical logic\n")
        with self.assertRaises(RuntimeError):
            adapter.verify_audit_source(audit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
