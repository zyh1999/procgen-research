"""Task30 process-start observer; no acceptance or module rebinding."""
import sys as _task30_sys

_task30_os = _task30_sys.modules.get("os")
_task30_probe = _task30_os.environ["TASK30_PROBE_PATH"]
_task30_preflight = _task30_os.environ["TASK30_PREFLIGHT_BASE_PATH"]
_task30_trainer = _task30_os.environ["TASK30_TRAINER_PATH"]
_task30_output = _task30_os.environ["TASK30_OBSERVER_LEDGER"]
_task30_snapshots = []
_task30_removed_self = False


def _task30_scalar(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    return None


def _task30_code(code):
    if code is None:
        return None
    constants = []
    for value in code.co_consts:
        scalar = _task30_scalar(value)
        if scalar is not None or value is None:
            constants.append({"kind": "scalar", "value": scalar})
        elif isinstance(value, type(code)):
            constants.append({
                "kind": "code", "filename": value.co_filename,
                "name": value.co_name, "firstlineno": value.co_firstlineno,
                "bytecode_hex": value.co_code.hex(),
            })
        else:
            constants.append({
                "kind": "typed", "type_module": type(value).__module__,
                "type_name": type(value).__qualname__,
            })
    return {
        "filename": code.co_filename, "name": code.co_name,
        "firstlineno": code.co_firstlineno, "flags": code.co_flags,
        "argcount": code.co_argcount, "kwonlyargcount": code.co_kwonlyargcount,
        "posonlyargcount": getattr(code, "co_posonlyargcount", 0),
        "names": list(code.co_names), "varnames": list(code.co_varnames),
        "bytecode_hex": code.co_code.hex(), "constants": constants,
    }


def _task30_value(value):
    scalar = _task30_scalar(value)
    if scalar is not None or value is None:
        return {"kind": "scalar", "value": scalar}
    if isinstance(value, type(_task30_value.__code__)):
        return {"kind": "code", "code": _task30_code(value)}
    if hasattr(value, "__code__"):
        return {
            "kind": "callable", "type_module": type(value).__module__,
            "type_name": type(value).__qualname__, "object_id": id(value),
            "code": _task30_code(value.__code__),
        }
    if isinstance(value, type(_task30_sys)):
        values = vars(value)
        return {
            "kind": "module", "name": values.get("__name__"),
            "file": values.get("__file__"), "object_id": id(value),
        }
    return {
        "kind": "typed", "type_module": type(value).__module__,
        "type_name": type(value).__qualname__, "object_id": id(value),
    }


def _task30_backing(raw):
    os_module = _task30_sys.modules.get("os")
    if not raw or os_module is None:
        return None
    try:
        resolved = os_module.path.realpath(raw)
        raw_lstat = os_module.lstat(raw)
        raw_stat = os_module.stat(raw)
        resolved_lstat = os_module.lstat(resolved)
        resolved_stat = os_module.stat(resolved)
    except (OSError, TypeError, ValueError) as error:
        return {"raw_path": raw, "error": type(error).__name__ + ": " + str(error)}
    def record(value):
        return {
            "device": value.st_dev, "inode": value.st_ino,
            "uid": value.st_uid, "gid": value.st_gid,
            "mode": oct(value.st_mode & 0o7777), "size": value.st_size,
        }
    return {
        "raw_path": raw, "resolved_path": resolved,
        "samefile": os_module.path.samefile(raw, resolved),
        "raw_lstat": record(raw_lstat), "raw_stat": record(raw_stat),
        "resolved_lstat": record(resolved_lstat), "resolved_stat": record(resolved_stat),
    }


def _task30_module(module):
    if module is None:
        return {"present": False}
    values = vars(module)
    spec = values.get("__spec__")
    loader = values.get("__loader__")
    module_type = type(module)
    keys = sorted(values)
    content = {key: _task30_value(values[key]) for key in keys}
    return {
        "present": True, "object_id": id(module), "dictionary_id": id(values),
        "type_module": module_type.__module__, "type_name": module_type.__qualname__,
        "mro": [item.__module__ + "." + item.__qualname__ for item in module_type.__mro__],
        "name": values.get("__name__"), "file": values.get("__file__"),
        "package": values.get("__package__"), "origin": values.get("__origin__"),
        "spec": None if spec is None else {
            "name": getattr(spec, "name", None), "origin": getattr(spec, "origin", None),
            "loader_module": type(getattr(spec, "loader", None)).__module__,
            "loader_name": type(getattr(spec, "loader", None)).__qualname__,
        },
        "loader": None if loader is None else {
            "type_module": type(loader).__module__, "type_name": type(loader).__qualname__,
            "name": getattr(loader, "name", None), "path": getattr(loader, "path", None),
        },
        "keys": keys, "normalized_content": content,
        "backing": _task30_backing(values.get("__file__")),
    }


def _task30_rng():
    torch = _task30_sys.modules.get("torch")
    if torch is None:
        return {"torch_loaded": False}
    result = {"torch_loaded": True, "cpu_state_hex": bytes(torch.get_rng_state().tolist()).hex()}
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        result["cuda_state_hex"] = bytes(torch.cuda.get_rng_state().tolist()).hex()
    else:
        result["cuda_state_hex"] = None
    return result


def _task30_snapshot(label, frame):
    main = _task30_sys.modules.get("__main__")
    alias = _task30_sys.modules.get("__mp_main__")
    main_record, alias_record = _task30_module(main), _task30_module(alias)
    main_content = main_record.get("normalized_content", {})
    alias_content = alias_record.get("normalized_content", {})
    main_keys, alias_keys = set(main_content), set(alias_content)
    differences = {
        "only_main": sorted(main_keys - alias_keys),
        "only_mp_main": sorted(alias_keys - main_keys),
        "different_fields": sorted(
            key for key in main_keys & alias_keys if main_content[key] != alias_content[key]
        ),
    }
    _task30_snapshots.append({
        "label": label, "module_count": len(_task30_sys.modules),
        "module_keys_in_order": list(_task30_sys.modules),
        "main_present": main is not None, "mp_main_present": alias is not None,
        "object_identity": main is not None and main is alias,
        "main": main_record, "mp_main": alias_record,
        "dictionary_difference": differences,
        "current_top_level_code": _task30_code(frame.f_code),
        "rng": _task30_rng(),
    })


def _task30_write():
    json_module = _task30_sys.modules.get("json")
    hashlib_module = _task30_sys.modules.get("hashlib")
    os_module = _task30_sys.modules.get("os")
    if json_module is None or hashlib_module is None or os_module is None:
        raise RuntimeError("Task30 natural imports missing before atomic ledger write")
    for item in _task30_snapshots:
        normalized = json_module.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        item["snapshot_sha256"] = hashlib_module.sha256(normalized).hexdigest()
    payload = {
        "result": "TASK30_NATURAL_STATE_OBSERVATION_COMPLETE",
        "observer": {
            "premature_multiprocessing_import": False,
            "sys_modules_assignment_or_rebinding": False,
            "observer_module_removed_before_probe_body": _task30_removed_self,
            "trace_disabled_before_origin_scan": True,
        },
        "snapshots": _task30_snapshots,
    }
    data = (json_module.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = _task30_output + ".tmp." + str(os_module.getpid())
    descriptor = os_module.open(temporary, os_module.O_WRONLY | os_module.O_CREAT | os_module.O_EXCL, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os_module.write(descriptor, data[offset:])
        os_module.fsync(descriptor)
    finally:
        os_module.close(descriptor)
    os_module.replace(temporary, _task30_output)


def _task30_local(frame, event, arg):
    global _task30_removed_self
    if event != "line":
        return _task30_local
    filename, line = frame.f_code.co_filename, frame.f_lineno
    if filename == _task30_probe:
        if line == 2 and not any(item["label"] == "closure_probe_start" for item in _task30_snapshots):
            _task30_sys.modules.pop("sitecustomize", None)
            _task30_removed_self = "sitecustomize" not in _task30_sys.modules
            _task30_snapshot("closure_probe_start", frame)
        elif line == 90 and not any(item["label"] == "origin_scan_before" for item in _task30_snapshots):
            _task30_snapshot("origin_scan_before", frame)
            _task30_sys.settrace(None)
            _task30_write()
            return None
    elif filename == _task30_preflight:
        mapping = {
            46: "trainer_import_before", 50: "trainer_import_after",
            145: "production_model_construction_after",
        }
        label = mapping.get(line)
        if label is not None and not any(item["label"] == label for item in _task30_snapshots):
            _task30_snapshot(label, frame)
    elif filename == _task30_trainer and frame.f_code.co_name == "<module>":
        if not any(item["label"] == "trainer_module_entry" for item in _task30_snapshots):
            _task30_snapshot("trainer_module_entry", frame)
    return _task30_local


def _task30_global(frame, event, arg):
    if event != "call":
        return None
    if frame.f_code.co_filename in (_task30_probe, _task30_preflight, _task30_trainer):
        return _task30_local
    return None


_task30_snapshot("child_process_entry", _task30_sys._getframe())
_task30_sys.settrace(_task30_global)
