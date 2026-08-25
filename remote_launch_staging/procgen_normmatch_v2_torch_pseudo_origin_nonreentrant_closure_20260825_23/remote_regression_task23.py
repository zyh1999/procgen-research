#!/usr/bin/env python3
"""Actual Python3.9/Torch positive and negative Task23 regression suite."""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nonreentrant_audit_hook import NonReentrantAuditRecorder
from pseudo_origin_extension import (
    _source_assignment_proof,
    revalidate_torch_classes_pseudo_origin,
    validate_torch_classes_pseudo_origin,
)

import torch


def expect_reject(label, function):
    try:
        function()
    except RuntimeError:
        return
    raise RuntimeError("negative case was not rejected: " + label)


module = sys.modules["torch.classes"]
with tempfile.TemporaryDirectory() as designated, tempfile.TemporaryDirectory() as bundle:
    approved = {
        "bundle": bundle,
        "site_packages": str(Path(torch.__file__).resolve().parents[1]),
        "stdlib": str(Path(os.__file__).resolve().parent),
    }
    first = validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated)
    revalidate_torch_classes_pseudo_origin(module, first)
    expect_reject("wrong key", lambda: validate_torch_classes_pseudo_origin("torch.not_classes", module, approved, designated))
    expect_reject("wrong type", lambda: validate_torch_classes_pseudo_origin("torch.classes", object(), approved, designated))
    for key, value in (("__spec__", object()), ("__loader__", object()), ("__package__", "torch"), ("__origin__", "x")):
        old = module.__dict__.get(key)
        module.__dict__[key] = value
        try:
            expect_reject(key, lambda: validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated))
        finally:
            module.__dict__[key] = old
    physical = Path(designated) / "_classes.py"
    physical.write_text("x")
    try:
        expect_reject("physical pseudo file", lambda: validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated))
    finally:
        physical.unlink()
    expect_reject("version", lambda: validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated, expected_version="0"))
    expect_reject("source sha", lambda: validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated, expected_sha256="0" * 64))
    expect_reject("source size", lambda: validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated, expected_size=1))
    expect_reject("RECORD", lambda: validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated, expected_record_hash="sha256=wrong"))
    expect_reject("assignment position", lambda: _source_assignment_proof('class _Classes:\n    __file__ = "_classes.py"\n'))
    sys.modules["torch.classes"] = object()
    try:
        expect_reject("module replacement", lambda: revalidate_torch_classes_pseudo_origin(module, first))
    finally:
        sys.modules["torch.classes"] = module

recorder = NonReentrantAuditRecorder()
recorder._local.active = True
recorder("open", ("x", "r", 0))
recorder._local.active = False
if recorder.reentrant_total != 1 or recorder.reentrant_by_event != {"open": 1}:
    raise RuntimeError("hook-induced recursion was not counted")
print(sys.version)
print("torch_version=" + torch.__version__)
print("TASK23_ACTUAL_PY39_TORCH_POSITIVE_NEGATIVE_PASS")
