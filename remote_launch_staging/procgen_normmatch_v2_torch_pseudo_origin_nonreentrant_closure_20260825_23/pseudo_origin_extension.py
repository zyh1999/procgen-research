#!/usr/bin/env python3
"""Narrow Task23 extension for the synthetic ``torch.classes`` pseudo-origin."""
import ast
import hashlib
import importlib.metadata
import os
import stat
import sys
from pathlib import Path

CLASSIFICATION = "APPROVED_INSTALLED_DISTRIBUTION_PSEUDO_ORIGIN"
EXPECTED_VERSION = "2.5.1+cu121"
EXPECTED_SOURCE_SHA256 = "2a3dd93d72e9f19450670b89f3a57b5b5adf245709f6a49a551bbfad33c434bf"
EXPECTED_SOURCE_SIZE = 1721
EXPECTED_RECORD_HASH = "sha256=Kj3ZPXLp8ZRQZwuJ86V7W1rfJFcJ9qSaVRu_rTPENL8"
EXPECTED_RECORD_PATH = "torch/_classes.py"
EXPECTED_ASSIGNMENT_LINE = 20


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_assignment_proof(text):
    tree = ast.parse(text, filename=EXPECTED_RECORD_PATH)
    compile(tree, EXPECTED_RECORD_PATH, "exec")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_Classes"]
    if len(classes) != 1:
        raise RuntimeError("installed source must define exactly one _Classes")
    assignments = []
    for node in classes[0].body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "__file__" and isinstance(node.value, ast.Constant):
            assignments.append(node)
    if len(assignments) != 1 or assignments[0].value.value != "_classes.py":
        raise RuntimeError("installed _Classes pseudo-file assignment mismatch")
    if assignments[0].lineno != EXPECTED_ASSIGNMENT_LINE:
        raise RuntimeError("installed _Classes pseudo-file assignment position mismatch")
    return {
        "class_line": classes[0].lineno,
        "assignment_line": assignments[0].lineno,
        "ast_parse": "PASS",
        "compile": "PASS",
    }


def validate_torch_classes_pseudo_origin(name, module, approved, designated,
                                         expected_version=EXPECTED_VERSION,
                                         expected_sha256=EXPECTED_SOURCE_SHA256,
                                         expected_size=EXPECTED_SOURCE_SIZE,
                                         expected_record_hash=EXPECTED_RECORD_HASH):
    if name != "torch.classes" or sys.modules.get(name) is not module:
        raise RuntimeError("torch.classes sys.modules key/object mismatch")
    import torch
    if type(module) is not torch._classes._Classes:
        raise RuntimeError("torch.classes exact module type mismatch")
    values = module.__dict__
    if values.get("__name__") != "torch.classes" or values.get("__file__") != "_classes.py":
        raise RuntimeError("torch.classes exact name/pseudo-file mismatch")
    if any(values.get(key) is not None for key in ("__spec__", "__loader__", "__package__")):
        raise RuntimeError("torch.classes spec/loader/package must all be None")
    if values.get("__origin__") is not None:
        raise RuntimeError("torch.classes origin must be absent")
    roots = [Path(designated)]
    for key in ("bundle", "site_packages", "stdlib"):
        values = approved.get(key) or []
        if isinstance(values, (str, os.PathLike)):
            values = [values]
        roots.extend(Path(value) for value in values)
    physical_checks = []
    for root in roots:
        candidate = root / "_classes.py"
        exists = os.path.lexists(candidate)
        physical_checks.append({"root": str(root), "candidate": str(candidate), "exists": exists})
        if exists:
            raise RuntimeError("torch.classes pseudo-origin has a corresponding physical file")
    dist = importlib.metadata.distribution("torch")
    if dist.metadata["Name"] != "torch" or dist.version != expected_version:
        raise RuntimeError("installed Torch distribution/version mismatch")
    files = {str(item): item for item in (dist.files or [])}
    item = files.get(EXPECTED_RECORD_PATH)
    if item is None or str(item.hash) != expected_record_hash or item.size != expected_size:
        raise RuntimeError("installed torch/_classes.py RECORD mismatch")
    source = Path(dist.locate_file(item)).resolve(strict=True)
    source_stat = os.lstat(source)
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise RuntimeError("installed torch/_classes.py is not a regular non-symlink file")
    source_hash = _sha256(source)
    if source_hash != expected_sha256 or source_stat.st_size != expected_size:
        raise RuntimeError("installed torch/_classes.py SHA/size mismatch")
    static = _source_assignment_proof(source.read_text())
    snapshot = {
        "object_id": id(module),
        "name": values.get("__name__"),
        "file": values.get("__file__"),
        "spec": values.get("__spec__"),
        "loader": values.get("__loader__"),
        "package": values.get("__package__"),
        "origin": values.get("__origin__"),
    }
    return {
        "classification": CLASSIFICATION,
        "sys_modules_key": name,
        "module_type_module": type(module).__module__,
        "module_type_name": type(module).__name__,
        "module_snapshot": snapshot,
        "physical_absence": physical_checks,
        "distribution": "torch",
        "version": dist.version,
        "source": str(source),
        "source_sha256": source_hash,
        "source_size": source_stat.st_size,
        "distribution_path": str(item),
        "record_hash": str(item.hash),
        "record_size": item.size,
        "static_source_proof": static,
    }


def revalidate_torch_classes_pseudo_origin(module, first):
    values = module.__dict__
    snap = first["module_snapshot"]
    checks = {
        "sys_modules_object": sys.modules.get("torch.classes") is module,
        "object_identity": id(module) == snap["object_id"],
        "name": values.get("__name__") == snap["name"],
        "file": values.get("__file__") == snap["file"],
        "spec": values.get("__spec__") is snap["spec"],
        "loader": values.get("__loader__") is snap["loader"],
        "package": values.get("__package__") is snap["package"],
        "origin": values.get("__origin__") is snap["origin"],
        "source_sha256": _sha256(first["source"]) == first["source_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError("torch.classes object/attributes/source changed during audit: " + str(checks))
    return {"result": "PSEUDO_ORIGIN_POST_AUDIT_REVALIDATION_PASS", "checks": checks}


def install(namespace):
    base_classify = namespace["_base_classify_origin"]
    validate_generated = namespace["validate_runtime_generated_thirdparty_module"]
    revalidate_generated = namespace["revalidate_runtime_generated_module"]
    reject_zip = namespace["reject_repository_local_zip_origin"]
    approved_roots = namespace["approved_origin_roots"]

    def audit_loaded_modules(bundle_root, manifest, designated, forbidden_roots):
        bundle_root = Path(bundle_root).resolve(strict=True)
        approved = approved_roots(bundle_root)
        files = {record["bundle_path"]: record for record in manifest["files"]}
        origins, bundle_origins, generated, pseudo = [], {}, [], []
        generated_modules, pseudo_modules = [], []
        for name, module in sorted(sys.modules.items()):
            spec = module.__dict__.get("__spec__") if hasattr(module, "__dict__") else None
            origin = getattr(spec, "origin", None) if spec is not None else None
            origin = origin or (module.__dict__.get("__file__") if hasattr(module, "__dict__") else None)
            if not origin:
                continue
            generated_record = pseudo_record = None
            try:
                classification = base_classify(origin, approved, designated, forbidden_roots)
            except RuntimeError:
                if name == namespace["_generated_name"]:
                    generated_record = validate_generated(name, module, forbidden_roots)
                    classification = generated_record["classification"]
                    generated.append(generated_record)
                    generated_modules.append(module)
                elif name == "torch.classes":
                    pseudo_record = validate_torch_classes_pseudo_origin(name, module, approved, designated)
                    classification = pseudo_record["classification"]
                    pseudo.append(pseudo_record)
                    pseudo_modules.append(module)
                else:
                    raise
            reject_zip(name, classification, manifest, origin)
            record = {"module": name, "origin": origin, "classification": classification}
            if generated_record is not None:
                record["runtime_generated_manifest_index"] = len(generated) - 1
            if pseudo_record is not None:
                record["pseudo_origin_manifest_index"] = len(pseudo) - 1
            if classification == "verified_bundle":
                path = Path(origin).resolve()
                relative = path.relative_to(bundle_root).as_posix()
                if relative not in files or _sha256(path) != files[relative]["sha256"]:
                    raise RuntimeError("bundle module absent from manifest or hash mismatch: " + name)
                record.update({"bundle_path": relative, "sha256": _sha256(path)})
                bundle_origins[relative] = name
            origins.append(record)
        if len(generated) != 1 or len(pseudo) != 1:
            raise RuntimeError(f"expected one physical generated module and one pseudo-origin, got {len(generated)}/{len(pseudo)}")
        missing = set(manifest["repository_local_import_closure"]) - set(bundle_origins)
        if missing:
            raise RuntimeError("repository-local closure modules were not imported: " + str(sorted(missing)))
        generated_post = [revalidate_generated(module, record) for module, record in zip(generated_modules, generated)]
        pseudo_post = [revalidate_torch_classes_pseudo_origin(module, record) for module, record in zip(pseudo_modules, pseudo)]
        return {
            "approved_roots": approved,
            "modules": origins,
            "bundle_origins": bundle_origins,
            "runtime_generated_thirdparty_modules": generated,
            "runtime_generated_post_import_revalidation": generated_post,
            "installed_distribution_pseudo_origins": pseudo,
            "pseudo_origin_post_audit_revalidation": pseudo_post,
        }

    namespace["audit_loaded_modules"] = audit_loaded_modules
    namespace["validate_torch_classes_pseudo_origin"] = validate_torch_classes_pseudo_origin
    namespace["revalidate_torch_classes_pseudo_origin"] = revalidate_torch_classes_pseudo_origin
    return namespace
