#!/usr/bin/env python3
"""Task17 origin policy plus one provenance-bound PyTorch generated module."""
import ast
import hashlib
import importlib.metadata
import json
import os
import stat
import sys
import time
from pathlib import Path

BASE_SHA256 = "061548959748423c570939f453c0e25c445dd6d1680bae451aca27e38514220e"
BASE_PATH = Path(os.environ.get(
    "TASK17_ORIGIN_SAFETY_BASE",
    Path(__file__).resolve().parents[1]
    / "procgen_normmatch_v2_interpreter_path_audit_6m_s0_20260825_17/origin_safety.py",
))
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("Task17 origin-safety base hash mismatch")
exec(compile(BASE_PATH.read_bytes(), str(BASE_PATH), "exec"), globals())

_base_classify_origin = classify_origin
_policy_load_ns = time.time_ns()
_generated_name = "_remote_module_non_scriptable"
_generated_preexisted = _generated_name in sys.modules
_baseline_path = Path(os.environ.get(
    "TORCH_GENERATED_PROVENANCE_BASELINE",
    Path(__file__).resolve().parent / "provenance/reproduction_1.json",
))
_baseline = json.loads(_baseline_path.read_text())


def _installed_generator_provenance(baseline):
    from torch.distributed.nn.api import remote_module
    from torch.distributed.nn.jit import instantiator
    from torch.distributed.nn.jit.templates import remote_module_template

    dist = importlib.metadata.distribution("torch")
    if dist.metadata["Name"] != baseline["distribution"]["name"]:
        raise RuntimeError("generated-module distribution name mismatch")
    if dist.version != baseline["distribution"]["version"]:
        raise RuntimeError("generated-module distribution version mismatch")
    if dist.read_text("INSTALLER") != baseline["distribution"]["installer"]:
        raise RuntimeError("generated-module installer provenance mismatch")
    if dist.read_text("direct_url.json") != baseline["distribution"]["direct_url"]:
        raise RuntimeError("generated-module direct-url provenance mismatch")
    modules = {
        "generator_loader": instantiator,
        "template": remote_module_template,
        "import_trigger": remote_module,
    }
    records = {}
    distribution_index = {str(item): item for item in (dist.files or [])}
    for label, module in modules.items():
        expected = baseline["generator"]["source_files"][label]
        origin = Path(module.__file__).resolve(strict=True)
        if str(origin) != expected["origin"]:
            raise RuntimeError(f"{label} installed origin mismatch")
        actual_hash = sha256(origin)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(f"{label} installed source hash mismatch")
        item = distribution_index.get(expected["distribution_path"])
        if item is None or str(item.hash) != expected["record_hash"] or item.size != expected["record_size"]:
            raise RuntimeError(f"{label} installed distribution RECORD mismatch")
        records[label] = {
            "module": module.__name__,
            "origin": str(origin),
            "sha256": actual_hash,
            "distribution_path": str(item),
            "record_hash": str(item.hash),
            "record_size": item.size,
        }
    return {
        "distribution": {
            "name": dist.metadata["Name"],
            "version": dist.version,
            "installer": dist.read_text("INSTALLER"),
            "direct_url": dist.read_text("direct_url.json"),
        },
        "generator_function": baseline["generator"]["function"],
        "template_function": baseline["generator"]["template_function"],
        "trigger_expression": baseline["generator"]["trigger_expression"],
        "source_files": records,
    }


def _expected_generated_content():
    from torch.distributed.nn.jit.templates.remote_module_template import get_remote_module_template

    substitutions = {
        "assign_module_interface_cls": "module_interface_cls = None",
        "args": "*args",
        "kwargs": "**kwargs",
        "arg_types": "*args, **kwargs",
        "arrow_and_return_type": "",
        "arrow_and_future_return_type": "",
        "jit_script_decorator": "",
    }
    return get_remote_module_template(True).format(**substitutions)


def validate_generated_content_safety(text, forbidden_roots):
    tree = ast.parse(text, filename=_generated_name + ".py")
    compile(tree, _generated_name + ".py", "exec")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(name.split(".")[0] not in {"typing", "torch"} for name in names):
            raise RuntimeError(f"generated content imports unapproved module: {names}")
    forbidden = [str(Path(item).resolve()) for item in forbidden_roots]
    forbidden += ["/Users/user/Documents/procgen", "/Users/user/.codex/worktrees/7edd/procgen",
                  "http://", "https://", "download("]
    for token in forbidden:
        if token and token in text:
            raise RuntimeError(f"generated content contains forbidden reference: {token}")
    return {"ast_parse": "PASS", "compile": "PASS", "approved_import_roots": ["torch", "typing"]}


def validate_runtime_generated_thirdparty_module(name, module, forbidden_roots,
                                                 baseline=None, policy_load_ns=None,
                                                 generated_preexisted=None):
    baseline = _baseline if baseline is None else baseline
    policy_load_ns = _policy_load_ns if policy_load_ns is None else policy_load_ns
    generated_preexisted = _generated_preexisted if generated_preexisted is None else generated_preexisted
    if generated_preexisted:
        raise RuntimeError("generated module existed before origin policy loaded")
    if name != _generated_name or module.__name__ != _generated_name:
        raise RuntimeError("runtime-generated module name mismatch")
    spec = module.__spec__
    if spec is None or spec.name != _generated_name or module.__package__ != "":
        raise RuntimeError("runtime-generated module spec/package mismatch")
    expected_module = baseline["module"]
    loader_module = type(spec.loader).__module__
    loader_class = type(spec.loader).__name__
    if loader_module != expected_module["loader_module"] or loader_class != expected_module["loader_class"]:
        raise RuntimeError("runtime-generated module loader mismatch")
    if module.__file__ != spec.origin:
        raise RuntimeError("runtime-generated module file/spec origin mismatch")
    origin = Path(spec.origin)
    if origin.name != _generated_name + ".py":
        raise RuntimeError("runtime-generated module basename mismatch")
    parent = origin.parent
    parent_stat = parent.lstat()
    file_stat = origin.lstat()
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise RuntimeError("runtime-generated parent is not a real directory")
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError("runtime-generated origin is not a regular file")
    if parent_stat.st_uid != os.geteuid() or file_stat.st_uid != os.geteuid():
        raise RuntimeError("runtime-generated parent/file owner mismatch")
    if stat.S_IMODE(parent_stat.st_mode) & 0o077:
        raise RuntimeError("runtime-generated parent permissions are not restricted")
    if parent_stat.st_ctime_ns < policy_load_ns or file_stat.st_ctime_ns < policy_load_ns:
        raise RuntimeError("runtime-generated parent/file predates audit origin policy")
    actual = origin.read_text()
    expected = _expected_generated_content()
    content_hash = hashlib.sha256(actual.encode()).hexdigest()
    if actual != expected or content_hash != expected_module["file"]["sha256"]:
        raise RuntimeError("runtime-generated content/template/hash mismatch")
    content_audit = validate_generated_content_safety(actual, forbidden_roots)
    provenance = _installed_generator_provenance(baseline)
    return {
        "classification": "APPROVED_RUNTIME_GENERATED_THIRDPARTY_MODULE",
        "module": name,
        "package": module.__package__,
        "spec_name": spec.name,
        "origin": str(origin.resolve(strict=True)),
        "loader_module": loader_module,
        "loader_class": loader_class,
        "process_origin_policy_load_ns": policy_load_ns,
        "file": {
            "device": file_stat.st_dev,
            "inode": file_stat.st_ino,
            "uid": file_stat.st_uid,
            "gid": file_stat.st_gid,
            "mode": oct(stat.S_IMODE(file_stat.st_mode)),
            "size": file_stat.st_size,
            "ctime_ns": file_stat.st_ctime_ns,
            "mtime_ns": file_stat.st_mtime_ns,
            "sha256": content_hash,
        },
        "parent": {
            "path": str(parent.resolve(strict=True)),
            "device": parent_stat.st_dev,
            "inode": parent_stat.st_ino,
            "uid": parent_stat.st_uid,
            "gid": parent_stat.st_gid,
            "mode": oct(stat.S_IMODE(parent_stat.st_mode)),
            "ctime_ns": parent_stat.st_ctime_ns,
        },
        "content_audit": content_audit,
        "content_match": "byte-identical independent reproduction and installed deterministic template",
        "generator_provenance": provenance,
        "independent_reproduction": {
            "result": baseline["result"],
            "content_sha256": baseline["module"]["file"]["sha256"],
            "size": baseline["module"]["file"]["size"],
        },
    }


def revalidate_runtime_generated_module(module, first):
    spec = module.__spec__
    origin = Path(spec.origin)
    parent_stat = origin.parent.lstat()
    file_stat = origin.lstat()
    actual = origin.read_bytes()
    checks = {
        "origin": str(origin.resolve(strict=True)) == first["origin"],
        "spec_origin": module.__file__ == spec.origin == first["origin"],
        "loader": type(spec.loader).__module__ == first["loader_module"] and type(spec.loader).__name__ == first["loader_class"],
        "file_identity": (file_stat.st_dev, file_stat.st_ino) == (first["file"]["device"], first["file"]["inode"]),
        "parent_identity": (parent_stat.st_dev, parent_stat.st_ino) == (first["parent"]["device"], first["parent"]["inode"]),
        "file_hash": hashlib.sha256(actual).hexdigest() == first["file"]["sha256"],
        "file_metadata": (file_stat.st_size, file_stat.st_ctime_ns, file_stat.st_mtime_ns) == (first["file"]["size"], first["file"]["ctime_ns"], first["file"]["mtime_ns"]),
    }
    if not all(checks.values()):
        raise RuntimeError(f"runtime-generated module replaced after import: {checks}")
    return {"result": "RUNTIME_GENERATED_MODULE_POST_IMPORT_REVALIDATION_PASS", "checks": checks}


def audit_loaded_modules(bundle_root, manifest, designated, forbidden_roots):
    bundle_root = Path(bundle_root).resolve(strict=True)
    approved = approved_origin_roots(bundle_root)
    files = {record["bundle_path"]: record for record in manifest["files"]}
    origins, bundle_origins, generated = [], {}, []
    generated_modules = []
    for name, module in sorted(sys.modules.items()):
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
        if not origin:
            continue
        try:
            classification = _base_classify_origin(origin, approved, designated, forbidden_roots)
            generated_record = None
        except RuntimeError:
            if name != _generated_name:
                raise
            generated_record = validate_runtime_generated_thirdparty_module(name, module, forbidden_roots)
            classification = generated_record["classification"]
            generated.append(generated_record)
            generated_modules.append(module)
        reject_repository_local_zip_origin(name, classification, manifest, origin)
        record = {"module": name, "origin": origin, "classification": classification}
        if generated_record is not None:
            record["runtime_generated_manifest_index"] = len(generated) - 1
        if classification == "verified_bundle":
            path = Path(origin).resolve()
            relative = path.relative_to(bundle_root).as_posix()
            if relative not in files:
                raise RuntimeError(f"bundle module absent from manifest: {name} -> {relative}")
            actual_hash = sha256(path)
            if actual_hash != files[relative]["sha256"]:
                raise RuntimeError(f"bundle module hash mismatch: {name} -> {relative}")
            record.update({"bundle_path": relative, "sha256": actual_hash})
            bundle_origins[relative] = name
        origins.append(record)
    if len(generated) != 1:
        raise RuntimeError(f"expected exactly one approved runtime-generated module, found {len(generated)}")
    expected = set(manifest["repository_local_import_closure"])
    missing = expected - set(bundle_origins)
    if missing:
        raise RuntimeError(f"repository-local closure modules were not imported: {sorted(missing)}")
    post = [revalidate_runtime_generated_module(module, record) for module, record in zip(generated_modules, generated)]
    return {
        "approved_roots": approved,
        "modules": origins,
        "bundle_origins": bundle_origins,
        "runtime_generated_thirdparty_modules": generated,
        "runtime_generated_post_import_revalidation": post,
    }
