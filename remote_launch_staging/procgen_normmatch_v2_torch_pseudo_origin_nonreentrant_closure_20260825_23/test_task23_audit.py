#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nonreentrant_audit_hook import NonReentrantAuditRecorder
from pseudo_origin_extension import _source_assignment_proof


class AuditHookTests(unittest.TestCase):
    def test_first_level_filesystem_events_and_zero_reentry(self):
        recorder = NonReentrantAuditRecorder()
        sys.addaudithook(recorder)
        with tempfile.TemporaryDirectory() as root:
            first = os.path.join(root, "first")
            second = os.path.join(root, "second")
            descriptor = os.open(first, os.O_CREAT | os.O_WRONLY, 0o600)
            os.write(descriptor, b"x")
            os.close(descriptor)
            os.rename(first, second)
            os.remove(second)
        names = [item["event"] for item in recorder.events]
        self.assertIn("open", names)
        self.assertIn("os.rename", names)
        self.assertIn("os.remove", names)
        self.assertEqual(recorder.reentrant_total, 0)

    def test_deliberate_nested_callback_is_counted(self):
        recorder = NonReentrantAuditRecorder()
        recorder._local.active = True
        recorder("open", ("x", "r", 0))
        recorder._local.active = False
        self.assertEqual(recorder.reentrant_total, 1)
        self.assertEqual(recorder.reentrant_by_event, {"open": 1})


class StaticSourceTests(unittest.TestCase):
    def test_exact_assignment(self):
        lines = ["#"] * 18 + ["class _Classes:", "    __file__ = \"_classes.py\""]
        proof = _source_assignment_proof("\n".join(lines) + "\n")
        self.assertEqual(proof["assignment_line"], 20)

    def test_wrong_assignment_position_rejected(self):
        with self.assertRaises(RuntimeError):
            _source_assignment_proof('class _Classes:\n    __file__ = "_classes.py"\n')


if __name__ == "__main__":
    unittest.main()
