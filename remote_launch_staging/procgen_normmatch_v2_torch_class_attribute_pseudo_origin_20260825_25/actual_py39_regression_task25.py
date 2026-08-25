#!/usr/bin/env python3
"""Actual Python3.9/Torch Task25 positive and negative classifier tests."""
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import class_attribute_classifier as classifier
import torch


def expect_reject(label, function):
    try:
        function()
    except RuntimeError:
        return
    raise RuntimeError("negative case was not rejected: " + label)


module = sys.modules["torch.classes"]
cls = type(module)
with tempfile.TemporaryDirectory() as designated, tempfile.TemporaryDirectory() as bundle:
    approved = {
        "bundle": [bundle],
        "site_packages": [str(Path(torch.__file__).resolve().parents[1])],
        "stdlib": [str(Path(os.__file__).resolve().parent)],
    }
    first = classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated)
    classifier.revalidate_torch_classes_pseudo_origin(module, first)
    if first["lookup_ledger"]["provider"] != "class attribute":
        raise RuntimeError("positive ledger did not identify class-attribute provider")
    if first["lookup_ledger"]["getattr_provider_call_count"] != 0:
        raise RuntimeError("positive public lookup invoked __getattr__")

    expect_reject("wrong key", lambda: classifier.validate_torch_classes_pseudo_origin("torch.not_classes", module, approved, designated))
    expect_reject("wrong type", lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", object(), approved, designated))
    module.__dict__["__file__"] = "_classes.py"
    try:
        expect_reject("instance file injection", lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated))
    finally:
        module.__dict__.pop("__file__", None)
    for key, value in (("__spec__", object()), ("__loader__", object()), ("__package__", "torch"), ("__origin__", "x")):
        old_present = key in module.__dict__
        old = module.__dict__.get(key)
        module.__dict__[key] = value
        try:
            expect_reject(key, lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated))
        finally:
            if old_present:
                module.__dict__[key] = old
            else:
                module.__dict__.pop(key, None)
    old_class_file = cls.__dict__["__file__"]
    setattr(cls, "__file__", "wrong.py")
    try:
        expect_reject("class/static/public inconsistency", lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated))
    finally:
        setattr(cls, "__file__", old_class_file)
    old_getattr = cls.__dict__["__getattr__"]
    def wrong_getattr(self, name):
        return "_classes.py"
    setattr(cls, "__getattr__", wrong_getattr)
    try:
        expect_reject("getattr monkeypatch", lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated))
    finally:
        setattr(cls, "__getattr__", old_getattr)
    side_effect = Path(designated) / "lookup_side_effect"
    def malicious_getattribute(self, name):
        side_effect.write_text(name)
        return types.ModuleType.__getattribute__(self, name)
    import types
    setattr(cls, "__getattribute__", malicious_getattribute)
    try:
        expect_reject("side-effect lookup injection", lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated))
        if side_effect.exists():
            raise RuntimeError("rejected side-effect lookup was executed")
    finally:
        delattr(cls, "__getattribute__")
    physical = Path(designated) / "_classes.py"
    physical.write_text("x")
    try:
        expect_reject("physical pseudo file", lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated))
    finally:
        physical.unlink()
    expect_reject("version", lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated, expected_version="0"))
    expect_reject("source sha", lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated, expected_sha256="0" * 64))
    expect_reject("source size", lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated, expected_size=1))
    expect_reject("RECORD", lambda: classifier.validate_torch_classes_pseudo_origin("torch.classes", module, approved, designated, expected_record_hash="sha256=wrong"))
    expect_reject("source lines", lambda: classifier._exact_static_source_proof('class _Classes:\n    __file__ = "_classes.py"\n'))
    sys.modules["torch.classes"] = object()
    try:
        expect_reject("module replacement", lambda: classifier.revalidate_torch_classes_pseudo_origin(module, first))
    finally:
        sys.modules["torch.classes"] = module

print(sys.version)
print("torch_version=" + torch.__version__)
print("TASK25_ACTUAL_PY39_TORCH_CLASS_ATTRIBUTE_POSITIVE_NEGATIVE_PASS")
