#!/usr/bin/env python3
"""Task25 narrow class-attribute pseudo-origin classifier overlay."""
import ast
import hashlib
import inspect
import os
import stat
import sys
import types
from pathlib import Path

BASE_SHA256 = "8d6de5470d118e5b4a2c86f6e23f7bcd1d63e47ae360ed5804f989ca71b80d01"
_base_value = os.environ.get("TASK23_PSEUDO_ORIGIN_BASE")
if _base_value:
    BASE_PATH = Path(_base_value)
else:
    BASE_PATH = (Path(__file__).resolve().parents[1]
                 / "procgen_normmatch_v2_torch_pseudo_origin_nonreentrant_closure_20260825_23/pseudo_origin_extension.py")
if hashlib.sha256(BASE_PATH.read_bytes()).hexdigest() != BASE_SHA256:
    raise RuntimeError("Task23 pseudo-origin base hash mismatch")
exec(compile(BASE_PATH.read_bytes(), str(BASE_PATH), "exec"), globals())

CLASSIFICATION = "APPROVED_INSTALLED_DISTRIBUTION_CLASS_ATTRIBUTE_PSEUDO_ORIGIN"
EXPECTED_CLASS_KEYS = frozenset({
    "__module__", "__file__", "__init__", "__getattr__", "load_library",
    "loaded_libraries", "__doc__",
})
EXPECTED_GETATTR_CODE_SHA256 = "a34c3dda08c0b70465ab4671936e773f03cfbc7ed5d4c5be861bb9180fb843dc"
EXPECTED_GETATTR_LINE = 25


def _identity_map(mapping):
    return {key: id(value) for key, value in mapping.items()}


def _exact_static_source_proof(text):
    tree = ast.parse(text, filename=EXPECTED_RECORD_PATH)
    compile(tree, EXPECTED_RECORD_PATH, "exec")
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_Classes"]
    if len(classes) != 1 or classes[0].lineno != 19:
        raise RuntimeError("installed _Classes definition position mismatch")
    assignments = [
        node for node in classes[0].body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__file__"
    ]
    if (len(assignments) != 1 or assignments[0].lineno != 20
            or not isinstance(assignments[0].value, ast.Constant)
            or assignments[0].value.value != "_classes.py"):
        raise RuntimeError("installed class-level pseudo-file assignment mismatch")
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == "__getattr__"]
    if len(methods) != 1 or methods[0].lineno != 25 or methods[0].end_lineno != 28:
        raise RuntimeError("installed __getattr__ source position mismatch")
    method = methods[0]
    if len(method.body) != 3:
        raise RuntimeError("installed __getattr__ body shape mismatch")
    first, second, third = method.body
    exact_semantics = (
        isinstance(first, ast.Assign) and len(first.targets) == 1
        and isinstance(first.targets[0], ast.Name) and first.targets[0].id == "namespace"
        and isinstance(first.value, ast.Call) and isinstance(first.value.func, ast.Name)
        and first.value.func.id == "_ClassNamespace"
        and isinstance(second, ast.Expr) and isinstance(second.value, ast.Call)
        and isinstance(second.value.func, ast.Name) and second.value.func.id == "setattr"
        and isinstance(third, ast.Return) and isinstance(third.value, ast.Name)
        and third.value.id == "namespace"
    )
    if not exact_semantics:
        raise RuntimeError("installed __getattr__ semantics mismatch")
    return {
        "class_line": 19,
        "class_file_assignment_line": 20,
        "getattr_start_line": 25,
        "getattr_end_line": 28,
        "getattr_only_handles_missing_names": True,
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
    cls = type(module)
    if cls is not torch._classes._Classes:
        raise RuntimeError("torch.classes exact type mismatch")
    if cls.__mro__ != (torch._classes._Classes, types.ModuleType, object):
        raise RuntimeError("torch.classes frozen MRO mismatch")
    if not issubclass(cls, types.ModuleType):
        raise RuntimeError("torch.classes is not a ModuleType subclass")
    class_before = dict(vars(cls))
    if set(class_before) != EXPECTED_CLASS_KEYS:
        raise RuntimeError("torch.classes exact class dictionary keys mismatch")
    instance_before = dict(vars(module))
    if instance_before.get("__name__") != "torch.classes" or "__file__" in instance_before:
        raise RuntimeError("torch.classes instance dictionary identity mismatch")
    if any(instance_before.get(key) is not None for key in ("__spec__", "__loader__", "__package__")):
        raise RuntimeError("torch.classes spec/loader/package must all be None")
    if instance_before.get("__origin__") is not None:
        raise RuntimeError("torch.classes origin must be absent")
    class_file = class_before.get("__file__")
    static_file = inspect.getattr_static(module, "__file__")
    if class_file != "_classes.py" or static_file != class_file:
        raise RuntimeError("torch.classes class/static pseudo-file mismatch")
    provider = class_before.get("__getattr__")
    installed_provider = torch._classes._Classes.__dict__.get("__getattr__")
    if provider is not installed_provider or provider.__module__ != "torch._classes" or provider.__qualname__ != "_Classes.__getattr__":
        raise RuntimeError("torch.classes frozen __getattr__ identity mismatch")
    if provider.__code__.co_firstlineno != EXPECTED_GETATTR_LINE:
        raise RuntimeError("torch.classes __getattr__ line mismatch")
    provider_code_sha = hashlib.sha256(provider.__code__.co_code).hexdigest()
    if provider_code_sha != EXPECTED_GETATTR_CODE_SHA256:
        raise RuntimeError("torch.classes __getattr__ bytecode mismatch")

    dist = importlib.metadata.distribution("torch")
    if dist.metadata["Name"] != "torch" or dist.version != expected_version:
        raise RuntimeError("installed Torch distribution/version mismatch")
    files = {str(item): item for item in (dist.files or [])}
    item = files.get(EXPECTED_RECORD_PATH)
    record_hash = None if item is None or item.hash is None else f"{item.hash.mode}={item.hash.value}"
    if item is None or record_hash != expected_record_hash or item.size != expected_size:
        raise RuntimeError("installed torch/_classes.py RECORD mismatch")
    source = Path(dist.locate_file(item)).resolve(strict=True)
    source_stat = os.lstat(source)
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISREG(source_stat.st_mode):
        raise RuntimeError("installed torch/_classes.py is not a regular non-symlink")
    source_hash = _sha256(source)
    if source_hash != expected_sha256 or source_stat.st_size != expected_size:
        raise RuntimeError("installed torch/_classes.py SHA/size mismatch")
    if Path(provider.__code__.co_filename).resolve(strict=True) != source:
        raise RuntimeError("torch.classes __getattr__ source path mismatch")
    static_proof = _exact_static_source_proof(source.read_text())

    roots = [Path(designated)]
    for key in ("bundle", "site_packages", "stdlib"):
        values = approved.get(key) or []
        if isinstance(values, (str, os.PathLike)):
            values = [values]
        roots.extend(Path(value) for value in values)
    physical_before = []
    for root in roots:
        candidate = root / "_classes.py"
        exists = os.path.lexists(candidate)
        physical_before.append({"root": str(root), "candidate": str(candidate), "exists": exists})
        if exists:
            raise RuntimeError("class-attribute pseudo-origin has a physical root file")

    provider_calls = []
    previous_profile = sys.getprofile()
    if previous_profile is not None:
        raise RuntimeError("unexpected preexisting profile callback")
    provider_code = provider.__code__
    def profile(frame, event, arg):
        if event == "call" and frame.f_code is provider_code:
            provider_calls.append(frame.f_lineno)
    sys.setprofile(profile)
    try:
        public_file = getattr(module, "__file__")
    finally:
        sys.setprofile(previous_profile)
    if public_file != "_classes.py" or provider_calls:
        raise RuntimeError("public pseudo-file lookup mismatch or invoked __getattr__")

    instance_after = dict(vars(module))
    class_after = dict(vars(cls))
    physical_after = [{**record, "exists_after": os.path.lexists(record["candidate"])} for record in physical_before]
    if instance_after != instance_before or _identity_map(class_after) != _identity_map(class_before):
        raise RuntimeError("public lookup changed module/class dictionaries")
    if any(record["exists_after"] for record in physical_after):
        raise RuntimeError("public lookup created a physical pseudo-origin file")
    if sys.modules.get(name) is not module or type(module) is not cls:
        raise RuntimeError("public lookup replaced module/type")

    return {
        "classification": CLASSIFICATION,
        "sys_modules_key": name,
        "module_type_module": cls.__module__,
        "module_type_name": cls.__name__,
        "mro": [item.__module__ + "." + item.__name__ for item in cls.__mro__],
        "module_snapshot": {
            "object_id": id(module), "type_id": id(cls),
            "instance_identity": _identity_map(instance_before),
            "class_identity": _identity_map(class_before),
        },
        "lookup_ledger": {
            "dict_file_present": False,
            "dict_file": None,
            "static_file": static_file,
            "class_level_file": class_file,
            "public_file": public_file,
            "provider": "class attribute",
            "getattr_provider_call_count": len(provider_calls),
            "instance_dictionary_unchanged": True,
            "class_dictionary_unchanged": True,
            "lookup_created_physical_file": False,
        },
        "dynamic_provider_identity": {
            "type": cls.__module__ + "." + cls.__name__,
            "method_module": provider.__module__,
            "method_qualname": provider.__qualname__,
            "method_firstlineno": provider.__code__.co_firstlineno,
            "method_code_sha256": provider_code_sha,
            "method_source": str(source),
            "method_source_sha256": source_hash,
            "not_invoked_for_file": True,
        },
        "physical_absence": physical_after,
        "distribution": "torch",
        "version": dist.version,
        "source": str(source),
        "source_sha256": source_hash,
        "source_size": source_stat.st_size,
        "distribution_path": str(item),
        "record_hash": record_hash,
        "record_size": item.size,
        "static_source_proof": static_proof,
    }


def revalidate_torch_classes_pseudo_origin(module, first):
    cls = type(module)
    snapshot = first["module_snapshot"]
    checks = {
        "sys_modules_object": sys.modules.get("torch.classes") is module,
        "module_identity": id(module) == snapshot["object_id"],
        "type_identity": id(cls) == snapshot["type_id"],
        "instance_dictionary": _identity_map(vars(module)) == snapshot["instance_identity"],
        "class_dictionary": _identity_map(vars(cls)) == snapshot["class_identity"],
        "static_file": inspect.getattr_static(module, "__file__") == first["lookup_ledger"]["static_file"],
        "public_file": getattr(module, "__file__") == first["lookup_ledger"]["public_file"],
        "source_sha256": _sha256(first["source"]) == first["source_sha256"],
    }
    if not all(checks.values()):
        raise RuntimeError("class-attribute pseudo-origin changed during audit: " + str(checks))
    return {"result": "CLASS_ATTRIBUTE_PSEUDO_ORIGIN_POST_AUDIT_REVALIDATION_PASS", "checks": checks}
