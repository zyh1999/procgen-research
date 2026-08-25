#!/usr/bin/env python3
"""Task23 full production construction plus approved physical/synthetic closure."""
import hashlib
import json
import os
import runpy
import stat
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nonreentrant_audit_hook import NonReentrantAuditRecorder

DEPLOY = Path(sys.argv[1]).resolve(strict=True)
OUTPUT = Path(sys.argv[2])
DESIGNATED = Path(sys.argv[3]).resolve(strict=True)
CODE = DEPLOY / "code"
FROZEN = DEPLOY / "frozen"
TRAINER = CODE / "train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
CONFIG = CODE / "configs/adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml"
PREFLIGHT = FROZEN / "gpuh_preflight_normmatch_v2.py"
MANIFEST = json.loads((DEPLOY / "BUNDLE_MANIFEST.json").read_text())
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


prestart = snapshot()
if prestart:
    raise RuntimeError("designated directory was not empty before process start")

recorder = NonReentrantAuditRecorder()
sys.addaudithook(recorder)

support_ns = {}
support_path = os.environ["NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT"]
exec(compile(open(support_path, "rb").read(), support_path, "exec"), support_ns)
policy_ns, policy_ledger = support_ns["load_explicit_policy"](str(DESIGNATED))
extension_path = os.environ["TASK23_PSEUDO_ORIGIN_EXTENSION"]
extension_ns = {}
exec(compile(open(extension_path, "rb").read(), extension_path, "exec"), extension_ns)
extension_ns["install"](policy_ns)

old_argv = sys.argv[:]
old_cwd = Path.cwd()
old_env = dict(os.environ)
try:
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONPATH"] = str(CODE)
    os.environ["PROCGEN_ENV"] = "bigfish-easy-0-10"
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
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

forbidden = [Path(item) for item in os.environ.get("FORBIDDEN_SOURCE_ROOTS", "").split(os.pathsep) if item]
origins = policy_ns["audit_loaded_modules"](DEPLOY, MANIFEST, DESIGNATED, forbidden)
post = snapshot()
ledger = recorder.ledger()
if ledger["reentrant_total"] != 0:
    raise RuntimeError("normal production reproduction emitted reentrant audit events: " + str(ledger))
event_counts = {}
for item in recorder.events:
    event_counts[item["event"]] = event_counts.get(item["event"], 0) + 1
for required in ("import", "open"):
    if event_counts.get(required, 0) == 0:
        raise RuntimeError("missing required first-level audit event: " + required)

payload = {
    "result": "TASK23_RUNTIME_CLOSURE_PROBE_PASS",
    "process": {"pid": os.getpid(), "python": sys.version},
    "production_construction": {
        "environment": "bigfish-easy-0-10",
        "trainer_sha256": sha256(TRAINER),
        "config_sha256": sha256(CONFIG),
        "preflight_sha256": sha256(PREFLIGHT),
        "result": "GPUH_HYBRID_HEAD_COMPATIBILITY_PASS",
    },
    "origin_policy": policy_ledger,
    "designated": {"path": str(DESIGNATED), "prestart": prestart, "post": post},
    "hook": {"ledger": ledger, "event_counts": event_counts, "first_level_events": recorder.events},
    "origin_closure": origins,
}
OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("TASK23_PRODUCTION_MODEL_CONSTRUCTION_PASS")
print("TASK23_NONREENTRANT_HOOK_PASS")
print("TASK23_PSEUDO_ORIGIN_PASS")
print("TASK23_RUNTIME_CLOSURE_PROBE_PASS")
