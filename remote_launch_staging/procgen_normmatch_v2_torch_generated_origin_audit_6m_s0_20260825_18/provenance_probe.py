#!/usr/bin/env python3
"""Reproduce and strictly attest PyTorch's non-scriptable RemoteModule source."""
import ast
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
import time
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


started_ns = time.time_ns()
module_name = "_remote_module_non_scriptable"
if module_name in sys.modules:
    raise RuntimeError("generated module existed before provenance process start")

from torch.distributed.nn.api import remote_module  # noqa: E402
from torch.distributed.nn.jit import instantiator  # noqa: E402
from torch.distributed.nn.jit.templates import remote_module_template  # noqa: E402

module = sys.modules.get(module_name)
if module is None:
    raise RuntimeError("expected generated module was not loaded")
spec = module.__spec__
origin = Path(spec.origin)
parent = origin.parent
file_stat = origin.lstat()
parent_stat = parent.lstat()
if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
    raise RuntimeError("generated origin is not a regular non-symlink file")
if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
    raise RuntimeError("generated parent is not a real directory")
if parent_stat.st_uid != os.geteuid() or file_stat.st_uid != os.geteuid():
    raise RuntimeError("generated parent/file ownership mismatch")
if stat.S_IMODE(parent_stat.st_mode) & 0o077:
    raise RuntimeError("generated parent is not restricted to current UID")
if parent_stat.st_ctime_ns < started_ns or file_stat.st_ctime_ns < started_ns:
    raise RuntimeError("generated parent/file predates provenance process")

substitutions = {
    "assign_module_interface_cls": "module_interface_cls = None",
    "args": "*args",
    "kwargs": "**kwargs",
    "arg_types": "*args, **kwargs",
    "arrow_and_return_type": "",
    "arrow_and_future_return_type": "",
    "jit_script_decorator": "",
}
expected = remote_module_template.get_remote_module_template(True).format(**substitutions)
actual = origin.read_text()
if actual != expected:
    raise RuntimeError("generated content differs from deterministic installed template")
tree = ast.parse(actual, filename=str(origin))
compile(tree, str(origin), "exec")
for forbidden in (
    "/Users/user/Documents/procgen",
    "/Users/user/.codex/worktrees/7edd/procgen",
    "http://",
    "https://",
    "download(",
):
    if forbidden in actual:
        raise RuntimeError(f"forbidden generated content reference: {forbidden}")

dist = importlib.metadata.distribution("torch")
source_paths = {
    "generator_loader": Path(instantiator.__file__).resolve(),
    "template": Path(remote_module_template.__file__).resolve(),
    "import_trigger": Path(remote_module.__file__).resolve(),
}
distribution_files = {}
for item in dist.files or []:
    relative = str(item)
    for label, source in source_paths.items():
        try:
            if source == Path(dist.locate_file(item)).resolve():
                distribution_files[label] = {
                    "distribution_path": relative,
                    "record_hash": str(item.hash) if item.hash else None,
                    "record_size": item.size,
                }
        except OSError:
            continue
if set(distribution_files) != set(source_paths):
    raise RuntimeError("generator source not fully represented by installed distribution")

payload = {
    "result": "TORCH_GENERATED_ORIGIN_PROVENANCE_PASS",
    "process_start_ns": started_ns,
    "distribution": {
        "name": dist.metadata["Name"],
        "version": dist.version,
        "installer": dist.read_text("INSTALLER"),
        "direct_url": dist.read_text("direct_url.json"),
    },
    "generator": {
        "module": instantiator.__name__,
        "function": "instantiate_non_scriptable_remote_module_template -> _do_instantiate_remote_module_template -> _write -> importlib.import_module",
        "template_module": remote_module_template.__name__,
        "template_function": "get_remote_module_template(True)",
        "trigger_module": remote_module.__name__,
        "trigger_expression": "instantiator.instantiate_non_scriptable_remote_module_template()",
        "source_files": {
            label: {"origin": str(path), "sha256": digest(path), **distribution_files[label]}
            for label, path in source_paths.items()
        },
    },
    "module": {
        "name": module.__name__,
        "package": module.__package__,
        "origin": str(origin.resolve(strict=True)),
        "spec_name": spec.name,
        "spec_origin": spec.origin,
        "loader_module": type(spec.loader).__module__,
        "loader_class": type(spec.loader).__name__,
        "file": {
            "uid": file_stat.st_uid,
            "gid": file_stat.st_gid,
            "mode": oct(stat.S_IMODE(file_stat.st_mode)),
            "size": file_stat.st_size,
            "ctime_ns": file_stat.st_ctime_ns,
            "mtime_ns": file_stat.st_mtime_ns,
            "sha256": digest(origin),
        },
        "parent": {
            "path": str(parent.resolve(strict=True)),
            "uid": parent_stat.st_uid,
            "gid": parent_stat.st_gid,
            "mode": oct(stat.S_IMODE(parent_stat.st_mode)),
            "ctime_ns": parent_stat.st_ctime_ns,
        },
        "content_match": "byte-identical deterministic installed template",
        "ast_parse": "PASS",
        "compile": "PASS",
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
