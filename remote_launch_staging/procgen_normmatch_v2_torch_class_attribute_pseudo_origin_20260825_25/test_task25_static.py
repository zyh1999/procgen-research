#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from class_attribute_classifier import _exact_static_source_proof


class StaticProofTests(unittest.TestCase):
    def test_exact_frozen_shape(self):
        lines = ["#"] * 18 + [
            "class _Classes(types.ModuleType):",
            "    __file__ = \"_classes.py\"",
            "",
            "    def __init__(self) -> None:",
            "        super().__init__(\"torch.classes\")",
            "",
            "    def __getattr__(self, name):",
            "        namespace = _ClassNamespace(name)",
            "        setattr(self, name, namespace)",
            "        return namespace",
        ]
        proof = _exact_static_source_proof("\n".join(lines) + "\n")
        self.assertEqual(proof["class_file_assignment_line"], 20)
        self.assertTrue(proof["getattr_only_handles_missing_names"])

    def test_wrong_line_rejected(self):
        with self.assertRaises(RuntimeError):
            _exact_static_source_proof('class _Classes:\n    __file__ = "_classes.py"\n')


if __name__ == "__main__":
    unittest.main()
