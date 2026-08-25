#!/usr/bin/env python3
"""Observe the complete designated-dir closure during production model construction."""
import ast
import hashlib
import importlib.metadata
import json
import os
import runpy
import stat
import sys
import time
import traceback
from pathlib import Path

DEPLOY = Path(sys.argv[1]).resolve(strict=True)
OUTPUT = Path(sys.argv[2])
DESIGNATED = Path(sys.argv[3]).resolve(strict=True)
CODE = DEPLOY / "code"
FROZEN = DEPLOY / "frozen"
TRAINER = CODE / "train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
CONFIG = CODE / "configs/adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml"
PREFLIGHT = FROZEN / "gpuh_preflight_normmatch_v2.py"
TRAINER_SHA = "0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b"
CONFIG_SHA = "9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    records = []
    for path in sorted(DESIGNATED.rglob("*")):
        value = os.lstat(path)
        records.append({
            "relative_path": path.relative_to(DESIGNATED).as_posix(),
            "device": value.st_dev,
            "inode": value.st_ino,
            "uid": value.st_uid,
            "gid": value.st_gid,
            "mode": oct(stat.S_IMODE(value.st_mode)),
            "size": value.st_size,
            "regular_file": stat.S_ISREG(value.st_mode),
            "directory": stat.S_ISDIR(value.st_mode),
            "symlink": stat.S_ISLNK(value.st_mode),
            "sha256": sha256(path) if stat.S_ISREG(value.st_mode) else None,
        })
    return records


def under_designated(value):
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    try:
        text = os.fsdecode(value)
        absolute = Path(text) if os.path.isabs(text) else DESIGNATED / text
        normalized = Path(os.path.abspath(str(absolute)))
        normalized.relative_to(DESIGNATED)
        return normalized
    except (OSError, TypeError, ValueError):
        return None


events = []
started_ns = time.time_ns()
prestart = snapshot()
if prestart:
    raise RuntimeError("designated directory was not empty before process start")


def audit_hook(event, args):
    if event not in {"open", "os.mkdir", "os.rename", "os.remove", "os.rmdir", "os.listdir", "os.scandir"}:
        return
    paths = []
    for value in args[:2]:
        normalized = under_designated(value)
        if normalized is not None:
            paths.append(normalized.relative_to(DESIGNATED).as_posix() or ".")
    if not paths:
        return
    stack = []
    for frame in traceback.extract_stack(limit=24)[:-1]:
        stack.append({"file": frame.filename, "line": frame.lineno, "name": frame.name})
    record = {"event": event, "paths": paths, "stack": stack}
    if event == "open":
        record["mode"] = args[1] if len(args) > 1 else None
        record["flags"] = args[2] if len(args) > 2 else None
    events.append(record)


sys.addaudithook(audit_hook)
old_argv = sys.argv[:]
old_cwd = Path.cwd()
old_env = dict(os.environ)
try:
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONPATH"] = str(CODE)
    os.environ["PROCGEN_ENV"] = "bigfish-easy-0-10"
    sys.argv = [
        str(PREFLIGHT), str(TRAINER), str(CONFIG),
        str(OUTPUT.parent / "parameter_partition.json"), TRAINER_SHA, CONFIG_SHA,
    ]
    os.chdir(DESIGNATED)
    runpy.run_path(str(PREFLIGHT), run_name="__main__")
finally:
    sys.argv = old_argv
    os.chdir(old_cwd)
    os.environ.clear()
    os.environ.update(old_env)

post = snapshot()
forbidden_tokens = [
    "/Users/user/Documents/procgen",
    "/Users/user/.codex/worktrees/7edd/procgen",
    "http://",
    "https://",
    "download(",
]
physical_artifacts = []
for item in post:
    if not item["regular_file"]:
        continue
    path = DESIGNATED / item["relative_path"]
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    compile(tree, str(path), "exec")
    found = [token for token in forbidden_tokens if token in text]
    if found:
        raise RuntimeError(f"runtime artifact contains forbidden references: {found}")
    physical_artifacts.append({
        **item,
        "ast_parse": "PASS",
        "compile": "PASS",
        "forbidden_reference_scan": "PASS",
    })
candidate_modules = []
for key, module in sorted(sys.modules.items()):
    module_file = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None) if spec is not None else None
    relative_pseudo = module_file and not os.path.isabs(str(module_file))
    file_path = under_designated(module_file) if module_file else None
    origin_path = under_designated(spec_origin) if spec_origin else None
    if not (relative_pseudo or file_path is not None or origin_path is not None):
        continue
    physical = None
    chosen = file_path or origin_path
    if chosen is not None and chosen.exists():
        value = os.lstat(chosen)
        physical = {
            "path": str(chosen),
            "device": value.st_dev,
            "inode": value.st_ino,
            "uid": value.st_uid,
            "gid": value.st_gid,
            "mode": oct(stat.S_IMODE(value.st_mode)),
            "size": value.st_size,
            "regular_file": stat.S_ISREG(value.st_mode),
            "symlink": stat.S_ISLNK(value.st_mode),
            "sha256": sha256(chosen) if stat.S_ISREG(value.st_mode) else None,
        }
    candidate_modules.append({
        "sys_modules_key": key,
        "module_name": getattr(module, "__name__", None),
        "module_type_module": type(module).__module__,
        "module_type_name": type(module).__name__,
        "file": module_file,
        "package": getattr(module, "__package__", None),
        "spec_present": spec is not None,
        "spec_name": getattr(spec, "name", None) if spec is not None else None,
        "spec_origin": spec_origin,
        "loader_module": type(spec.loader).__module__ if spec is not None and spec.loader is not None else None,
        "loader_class": type(spec.loader).__name__ if spec is not None and spec.loader is not None else None,
        "physical_artifact": physical,
    })

torch_classes = next((item for item in candidate_modules if item["sys_modules_key"] == "torch.classes"), None)
if torch_classes is None:
    raise RuntimeError("production construction did not expose torch.classes pseudo-origin")
from torch import _classes as torch_classes_source
source = Path(torch_classes_source.__file__).resolve(strict=True)
source_text = source.read_text()
tree = ast.parse(source_text, filename=str(source))
compile(tree, str(source), "exec")
dist = importlib.metadata.distribution("torch")
dist_item = None
for item in dist.files or []:
    try:
        if Path(dist.locate_file(item)).resolve(strict=True) == source:
            dist_item = item
            break
    except OSError:
        continue
if dist_item is None:
    raise RuntimeError("torch._classes source absent from installed distribution RECORD")

payload = {
    "result": "RUNTIME_GENERATED_CLOSURE_PROBE_COMPLETE",
    "process": {
        "pid": os.getpid(),
        "python": sys.version,
        "torch_distribution": dist.metadata["Name"],
        "torch_version": dist.version,
        "start_ns": started_ns,
    },
    "production_construction": {
        "environment": "bigfish-easy-0-10",
        "trainer_sha256": sha256(TRAINER),
        "config_sha256": sha256(CONFIG),
        "preflight_sha256": sha256(PREFLIGHT),
        "result": "GPUH_HYBRID_HEAD_COMPATIBILITY_PASS",
    },
    "designated": {
        "path": str(DESIGNATED),
        "prestart": prestart,
        "post_model_construction": post,
        "physical_artifacts": physical_artifacts,
        "filesystem_events": events,
    },
    "candidate_modules": candidate_modules,
    "torch_classes": torch_classes,
    "torch_classes_source_provenance": {
        "module": torch_classes_source.__name__,
        "source": str(source),
        "sha256": sha256(source),
        "size": source.stat().st_size,
        "ast_parse": "PASS",
        "compile": "PASS",
        "distribution": dist.metadata["Name"],
        "version": dist.version,
        "distribution_path": str(dist_item),
        "record_hash": str(dist_item.hash),
        "record_size": dist_item.size,
        "synthetic_file_assignment_present": '__file__ = "_classes.py"' in source_text,
    },
}
OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("TASK22_PRODUCTION_MODEL_CONSTRUCTION_PASS")
print("TASK22_RUNTIME_CLOSURE_PROBE_COMPLETE")
print("torch_classes_physical_artifact=" + str(torch_classes["physical_artifact"]))
