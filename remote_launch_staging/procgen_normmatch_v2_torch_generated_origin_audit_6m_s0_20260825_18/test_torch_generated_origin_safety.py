#!/usr/bin/env python3
"""Positive and mandatory negative tests for the single Task18 category."""
import importlib.machinery
import importlib.util
import os
import tempfile
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("task18_origin_safety", HERE / "origin_safety.py")
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)

if safety._generated_preexisted:
    raise AssertionError("generated module unexpectedly preexisted test policy")
from torch.distributed.nn.api import remote_module  # noqa: E402,F401
import sys  # noqa: E402

name = "_remote_module_non_scriptable"
module = sys.modules[name]

# Positive: exact installed generator, lifecycle, loader, template, content and origin.
valid = safety.validate_runtime_generated_thirdparty_module(name, module, [])
assert valid["classification"] == "APPROVED_RUNTIME_GENERATED_THIRDPARTY_MODULE"
assert valid["file"]["sha256"] == "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
assert safety.revalidate_runtime_generated_module(module, valid)["result"].endswith("PASS")

# Negative: same-name module/file preexisting before policy is rejected.
try:
    safety.validate_runtime_generated_thirdparty_module(name, module, [], generated_preexisted=True)
except RuntimeError:
    pass
else:
    raise AssertionError("preexisting generated module was accepted")

# Negative: invalid AST and repository/network references are rejected.
for text, forbidden in [
    ("def broken(:\n", []),
    ("X = '/Users/user/Documents/procgen/secret.py'\n", ["/Users/user/Documents/procgen"]),
    ("X = 'https://example.invalid/download'\n", []),
]:
    try:
        safety.validate_generated_content_safety(text, forbidden)
    except (RuntimeError, SyntaxError):
        pass
    else:
        raise AssertionError("unsafe generated content was accepted")

expected = safety._expected_generated_content()


def fake_module(path, loader=None):
    loader = loader or importlib.machinery.SourceFileLoader(name, str(path))
    fake_spec = types.SimpleNamespace(name=name, origin=str(path), loader=loader)
    return types.SimpleNamespace(__name__=name, __package__="", __spec__=fake_spec, __file__=str(path))


with tempfile.TemporaryDirectory(prefix="task18_negative_") as temporary:
    root = Path(temporary)
    root.chmod(0o700)

    # Negative: content/hash mismatch.
    mismatch = root / (name + ".py")
    mismatch.write_text(expected + "\n# altered\n")
    try:
        safety.validate_runtime_generated_thirdparty_module(
            name, fake_module(mismatch), [], policy_load_ns=time.time_ns() - 10**9,
            generated_preexisted=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("content/hash mismatch was accepted")

    # Negative: non-PyTorch/nonstandard loader.
    mismatch.write_text(expected)
    class FakeLoader:
        pass
    try:
        safety.validate_runtime_generated_thirdparty_module(
            name, fake_module(mismatch, FakeLoader()), [], policy_load_ns=time.time_ns() - 10**9,
            generated_preexisted=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-PyTorch generator/loader was accepted")

    # Negative: symlink parent and symlink file.
    real_parent = root / "real_parent"
    real_parent.mkdir(mode=0o700)
    real_file = real_parent / (name + ".py")
    real_file.write_text(expected)
    linked_parent = root / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    try:
        safety.validate_runtime_generated_thirdparty_module(
            name, fake_module(linked_parent / real_file.name), [],
            policy_load_ns=time.time_ns() - 10**9, generated_preexisted=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("symlink parent was accepted")
    mismatch.unlink()
    linked_file = root / (name + ".py")
    linked_file.symlink_to(real_file)
    try:
        safety.validate_runtime_generated_thirdparty_module(
            name, fake_module(linked_file), [], policy_load_ns=time.time_ns() - 10**9,
            generated_preexisted=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("symlink file was accepted")

# Negative: post-import replacement is detected, then restore the isolated temp source.
origin = Path(module.__spec__.origin)
original = origin.read_text()
origin.write_text(original + "\n# replacement-test\n")
try:
    safety.revalidate_runtime_generated_module(module, valid)
except RuntimeError:
    pass
else:
    raise AssertionError("post-import file replacement was accepted")
finally:
    origin.write_text(original)
restored = safety.validate_runtime_generated_thirdparty_module(name, module, [])
assert safety.revalidate_runtime_generated_module(module, restored)["result"].endswith("PASS")

print("TASK18_TORCH_GENERATED_ORIGIN_TESTS_PASS")
