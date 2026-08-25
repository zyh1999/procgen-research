#!/usr/bin/env python3
"""Task31R in-path capture wrapper; no acceptance, import, or hook changes."""
import hashlib
import json
import os
import runpy
import stat
import sys
from pathlib import Path

FROZEN_PROBE_SHA256 = "c3529cb171306d7b3b0517974a682ddffb91d65dc015c102b1e84658e9eeb1f5"
FROZEN_PROBE = Path(sys.argv[1])
FROZEN_PROBE_RESOLVED = FROZEN_PROBE.resolve(strict=True)
CAPTURE_MODE = os.environ["TASK31R_CAPTURE_MODE"]
CAPTURE_OUTPUT = Path(os.environ["TASK31R_CAPTURE_OUTPUT"])
if CAPTURE_MODE not in {"on", "off"}:
    raise RuntimeError("Task31R capture mode must be on or off")
if hashlib.sha256(FROZEN_PROBE_RESOLVED.read_bytes()).hexdigest() != FROZEN_PROBE_SHA256:
    raise RuntimeError("Task31R frozen Task23 probe identity mismatch")

_MODULE_TYPE = type(sys)
_FUNCTION_TYPE = type(lambda: None)
_ENTRY_MAIN = sys.modules.get("__main__")
_ENTRY_MP_MAIN = sys.modules.get("__mp_main__")
_ENTRY_KEYS = tuple(sys.modules)


def _stat_record(value):
    return {
        "device": value.st_dev, "inode": value.st_ino,
        "uid": value.st_uid, "gid": value.st_gid,
        "mode": oct(stat.S_IMODE(value.st_mode)), "size": value.st_size,
        "regular_file": stat.S_ISREG(value.st_mode),
        "symlink": stat.S_ISLNK(value.st_mode),
    }


def _fd_backing(raw):
    if not raw:
        return None
    resolved = os.path.realpath(raw)
    raw_lstat, raw_stat = os.lstat(raw), os.stat(raw)
    resolved_lstat, resolved_stat = os.lstat(resolved), os.stat(resolved)
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise RuntimeError("Task31R O_NOFOLLOW unavailable")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow)
    try:
        opened = os.fstat(descriptor)
        chunks, remaining = [], opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(descriptor)
    data = b"".join(chunks)
    if len(data) != opened.st_size:
        raise RuntimeError("Task31R backing fd short read")
    return {
        "raw_path": str(raw), "resolved_path": resolved,
        "samefile": os.path.samefile(raw, resolved),
        "raw_lstat": _stat_record(raw_lstat), "raw_stat": _stat_record(raw_stat),
        "resolved_lstat": _stat_record(resolved_lstat),
        "resolved_stat": _stat_record(resolved_stat),
        "opened_fd": _stat_record(opened),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _code_record(code):
    return {
        "filename": code.co_filename, "name": code.co_name,
        "firstlineno": code.co_firstlineno,
        "bytecode_sha256": hashlib.sha256(code.co_code).hexdigest(),
        "names": list(code.co_names), "varnames": list(code.co_varnames),
    }


def _module_record(module, top_code=None):
    if module is None:
        return {"present": False}
    values = vars(module)
    spec = values.get("__spec__")
    loader = values.get("__loader__")
    spec_values = {} if spec is None else vars(spec)
    loader_values = {} if loader is None or not hasattr(loader, "__dict__") else vars(loader)
    function_codes = []
    for key in sorted(values):
        value = values[key]
        if type(value) is _FUNCTION_TYPE:
            function_codes.append({"key": key, **_code_record(value.__code__)})
    return {
        "present": True, "object_id": id(module), "dictionary_id": id(values),
        "type_module": type(module).__module__, "type_name": type(module).__qualname__,
        "mro": [item.__module__ + "." + item.__qualname__ for item in type(module).__mro__],
        "name": values.get("__name__"), "file": values.get("__file__"),
        "package": values.get("__package__"), "origin": values.get("__origin__"),
        "spec": None if spec is None else {
            "name": spec_values.get("name"), "origin": spec_values.get("origin"),
            "loader_type_module": type(spec_values.get("loader")).__module__,
            "loader_type_name": type(spec_values.get("loader")).__qualname__,
        },
        "loader": None if loader is None else {
            "type_module": type(loader).__module__, "type_name": type(loader).__qualname__,
            "name": loader_values.get("name"), "path": loader_values.get("path"),
        },
        "dictionary_keys": sorted(values),
        "dictionary_key_sha256": hashlib.sha256(
            json.dumps(sorted(values), separators=(",", ":")).encode()
        ).hexdigest(),
        "function_codes": function_codes,
        "top_level_code": None if top_code is None else _code_record(top_code),
        "backing": _fd_backing(values.get("__file__")),
    }


def _find_frames(error):
    frames = []
    traceback = error.__traceback__
    while traceback is not None:
        frames.append(traceback.tb_frame)
        traceback = traceback.tb_next
    policy = None
    natural_main = None
    natural_main_code = None
    for frame in frames:
        locals_ = frame.f_locals
        if (frame.f_code.co_name == "audit_loaded_modules"
                and isinstance(locals_.get("origins"), list)
                and locals_.get("name") == "__mp_main__"):
            policy = frame
        for value in locals_.values():
            if type(value).__name__ != "_TempModule" or not hasattr(value, "__dict__"):
                continue
            candidate = vars(value).get("module")
            if type(candidate) is _MODULE_TYPE:
                candidate_values = vars(candidate)
                if candidate_values.get("__file__") == str(FROZEN_PROBE):
                    natural_main = candidate
    for frame in frames:
        if frame.f_code.co_filename == str(FROZEN_PROBE) and frame.f_code.co_name == "<module>":
            natural_main_code = frame.f_code
    if policy is None or natural_main is None or natural_main_code is None:
        raise RuntimeError("Task31R required existing origin/runpy frames unavailable")
    return policy, natural_main, natural_main_code


def _normalized_module(record, role):
    backing = record.get("backing") or {}
    functions = [
        {
            "key": item["key"], "name": item["name"],
            "firstlineno": item["firstlineno"],
            "bytecode_sha256": item["bytecode_sha256"],
        }
        for item in record.get("function_codes", [])
    ]
    top = record.get("top_level_code")
    return {
        "role": role, "present": record.get("present"),
        "type_module": record.get("type_module"), "type_name": record.get("type_name"),
        "mro": record.get("mro"), "name": record.get("name"),
        "file_role": role, "package": record.get("package"),
        "spec": None if record.get("spec") is None else {
            "name": record["spec"].get("name"),
            "origin_role": role if record["spec"].get("origin") else None,
            "loader_type_module": record["spec"].get("loader_type_module"),
            "loader_type_name": record["spec"].get("loader_type_name"),
        },
        "loader": None if record.get("loader") is None else {
            "type_module": record["loader"].get("type_module"),
            "type_name": record["loader"].get("type_name"),
        },
        "dictionary_keys": record.get("dictionary_keys"),
        "dictionary_key_sha256": record.get("dictionary_key_sha256"),
        "backing_sha256": backing.get("sha256"),
        "backing_size": (backing.get("opened_fd") or {}).get("size"),
        "function_codes": functions,
        "top_level_code": None if top is None else {
            "name": top["name"], "firstlineno": top["firstlineno"],
            "bytecode_sha256": top["bytecode_sha256"],
        },
    }


def _rng_summary():
    torch = sys.modules.get("torch")
    if torch is None:
        return {"torch_loaded": False}
    cpu = bytes(torch.get_rng_state().tolist())
    result = {
        "torch_loaded": True,
        "cpu_state_sha256": hashlib.sha256(cpu).hexdigest(),
        "cuda_initialized": torch.cuda.is_available() and torch.cuda.is_initialized(),
    }
    if result["cuda_initialized"]:
        cuda = bytes(torch.cuda.get_rng_state().tolist())
        result["cuda_state_sha256"] = hashlib.sha256(cuda).hexdigest()
    else:
        result["cuda_state_sha256"] = None
    return result


def _capture(error):
    policy_frame, natural_main, natural_main_code = _find_frames(error)
    locals_ = policy_frame.f_locals
    mp_main = locals_["module"]
    if type(mp_main) is not _MODULE_TYPE:
        raise RuntimeError("Task31R existing __mp_main__ record is not a module")
    main = _module_record(natural_main, natural_main_code)
    alias = _module_record(mp_main)
    prior = [item for item in locals_["origins"] if item.get("module") == "__main__"]
    if len(prior) != 1 or prior[0].get("origin") != str(FROZEN_PROBE):
        raise RuntimeError("Task31R existing __main__ origin record mismatch")
    current = locals_.get("record")
    if (not isinstance(current, dict) or current.get("module") != "__mp_main__"
            or current.get("origin") != alias["file"]):
        raise RuntimeError("Task31R existing __mp_main__ origin record mismatch")
    if natural_main is mp_main:
        raise RuntimeError("Task31R natural terminal modules unexpectedly identical")
    main_norm = _normalized_module(main, "FROZEN_TASK23_PROBE")
    alias_norm = _normalized_module(alias, "DEPLOYED_TASK27_PREFLIGHT")
    relation = {
        "object_identity": False,
        "main": main_norm, "mp_main": alias_norm,
        "dictionary_difference": {
            "only_main": sorted(set(main["dictionary_keys"]) - set(alias["dictionary_keys"])),
            "only_mp_main": sorted(set(alias["dictionary_keys"]) - set(main["dictionary_keys"])),
            "shared_key_count": len(set(main["dictionary_keys"]) & set(alias["dictionary_keys"])),
        },
        "existing_origin_records": {"main": prior[0], "mp_main": current},
    }
    stable_relation = {
        "object_identity": False,
        "main": main_norm, "mp_main": alias_norm,
        "dictionary_difference": relation["dictionary_difference"],
        "existing_origin_records": {
            "main": {"module": prior[0].get("module"),
                     "classification": prior[0].get("classification"),
                     "origin_role": "FROZEN_TASK23_PROBE"},
            "mp_main": {"module": current.get("module"),
                        "classification": current.get("classification"),
                        "origin_role": "DEPLOYED_TASK27_PREFLIGHT"},
        },
    }
    stable = json.dumps(stable_relation, sort_keys=True, separators=(",", ":")).encode()
    relation_sha = hashlib.sha256(stable).hexdigest()
    entry = {
        "main_present": _ENTRY_MAIN is not None,
        "mp_main_present": _ENTRY_MP_MAIN is not None,
        "object_identity": _ENTRY_MAIN is not None and _ENTRY_MAIN is _ENTRY_MP_MAIN,
        "module_cardinality": len(_ENTRY_KEYS),
        "normalized_module_set_sha256": hashlib.sha256(
            json.dumps(sorted(_ENTRY_KEYS), separators=(",", ":")).encode()
        ).hexdigest(),
    }
    module_keys = tuple(policy_frame.f_globals["sys"].modules)
    milestones = [
        {"label": "child_entry", **entry},
        {"label": "closure_probe_start", "main_present": True,
         "mp_main_present": False, "object_identity": False,
         "evidence": "frozen probe traceback module plus entry absence"},
        {"label": "trainer_import_before", "main_present": True,
         "mp_main_present": True, "object_identity": True,
         "evidence": "deployed preflight runpy module retained as __mp_main__"},
        {"label": "trainer_import_after", "main_present": True,
         "mp_main_present": True, "object_identity": True,
         "evidence": "deployed preflight runpy module retained as __mp_main__"},
        {"label": "production_model_construction_after", "main_present": True,
         "mp_main_present": True, "object_identity": True,
         "evidence": "Task27 artifacts emitted before origin scan"},
        {"label": "origin_scan_before", "main_present": True,
         "mp_main_present": True, "object_identity": False,
         "evidence": "existing origin policy frame records"},
    ]
    return {
        "result": "TASK31R_INPATH_CAPTURE_COMPLETE",
        "capture_mode": CAPTURE_MODE,
        "frozen_probe_sha256": FROZEN_PROBE_SHA256,
        "milestones": milestones,
        "terminal_relation": relation,
        "terminal_relation_normalized": stable_relation,
        "terminal_relation_sha256": relation_sha,
        "origin_scan_module_cardinality": len(module_keys),
        "origin_scan_normalized_module_set": sorted(module_keys),
        "origin_scan_normalized_module_set_sha256": hashlib.sha256(
            json.dumps(sorted(module_keys), separators=(",", ":")).encode()
        ).hexdigest(),
        "rng_summary": _rng_summary(),
        "original_exception": type(error).__name__ + ": " + str(error),
    }


forwarded = [str(FROZEN_PROBE), *sys.argv[2:]]
old_argv = sys.argv[:]
try:
    sys.argv = forwarded
    runpy.run_path(str(FROZEN_PROBE), run_name="__main__")
except Exception as error:
    payload = _capture(error)
    print("TASK31R_INPATH_RELATION_SHA256=" + payload["terminal_relation_sha256"])
    print("TASK31R_INPATH_MODULE_SET_SHA256=" + payload["origin_scan_normalized_module_set_sha256"])
    if CAPTURE_MODE == "on":
        data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        temporary = str(CAPTURE_OUTPUT) + ".tmp." + str(os.getpid())
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, CAPTURE_OUTPUT)
    raise
finally:
    sys.argv = old_argv
