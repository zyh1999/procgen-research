#!/usr/bin/env python3
"""Formal Task23 clean-room audit with the frozen policy plus narrow extension."""
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

support_namespace = {}
support_path = os.environ["NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT"]
exec(compile(open(support_path, "rb").read(), support_path, "exec"), support_namespace)
namespace, policy_ledger = support_namespace["load_explicit_policy"](
    os.environ["DESIGNATED_EMPTY"],
    expected_ledger_path=os.environ["ORIGIN_POLICY_PRESTART_LEDGER"],
)
extension_namespace = {}
extension_path = os.environ["TASK23_PSEUDO_ORIGIN_EXTENSION"]
exec(compile(open(extension_path, "rb").read(), extension_path, "exec"), extension_namespace)
extension_namespace["install"](namespace)
snapshot_empty_directory = namespace["snapshot_empty_directory"]
audit_sys_path = namespace["audit_sys_path"]
audit_loaded_modules = namespace["audit_loaded_modules"]
approved_origin_roots = namespace["approved_origin_roots"]
write_json = namespace["write_json"]

BUNDLE = Path(os.environ["BUNDLE_ROOT"]).resolve(strict=True)
CODE = BUNDLE / "code"
OUT = Path(os.environ["AUDIT_OUTPUT"])
ORIGIN_OUT = Path(os.environ["IMPORT_ORIGIN_OUTPUT"])
DESIGNATED = Path(os.environ["DESIGNATED_EMPTY"]).resolve(strict=True)
PRESTART = json.loads(Path(os.environ["DESIGNATED_PRESTART_JSON"]).read_text())
FORBIDDEN = [Path(item) for item in os.environ.get("FORBIDDEN_SOURCE_ROOTS", "").split(os.pathsep) if item]
TRAINER = CODE / "train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
CONFIG = CODE / "configs/adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml"
MANIFEST = json.loads((BUNDLE / "BUNDLE_MANIFEST.json").read_text())
EXPECTED_TRAINER = "0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b"
EXPECTED_CONFIG = "9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda"
ENVS = ["bigfish-easy-0-10", "bossfight-easy-0-10", "caveflyer-easy-0-10", "coinrun-easy-0-10"]

if os.environ.get("PYTHONNOUSERSITE") != "1":
    raise RuntimeError("PYTHONNOUSERSITE must be 1")
if hashlib.sha256(TRAINER.read_bytes()).hexdigest() != EXPECTED_TRAINER:
    raise RuntimeError("frozen trainer hash mismatch")
if hashlib.sha256(CONFIG.read_bytes()).hexdigest() != EXPECTED_CONFIG:
    raise RuntimeError("frozen config hash mismatch")
before = snapshot_empty_directory(DESIGNATED, "interpreter_start", PRESTART)
approved = approved_origin_roots(BUNDLE)
path_manifest = audit_sys_path(BUNDLE, DESIGNATED, approved, FORBIDDEN)

spec = importlib.util.spec_from_file_location("normmatch_v2_trainer", TRAINER)
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)
for name in [
    "utils.logger", "utils.runners", "utils.utils", "utils.vision_transformers",
    "utils.vit", "utils.resnet", "utils.convnet", "utils.popart",
    "utils.running_mean_std", "utils.transformer", "utils.rope",
    "utils.seq_running_mean_std", "utils.monitor", "vec_env",
    "vec_env.vec_env", "vec_env.shmem_vec_env", "vec_env.subproc_vec_env",
    "vec_env.vec_monitor", "vec_env.vec_normalize", "vec_env.vec_remove_dict_obs",
    "vec_env.dummy_vec_env", "vec_env.util",
]:
    importlib.import_module(name)

resolved = {}
original_argv, original_cwd, original_train = sys.argv[:], Path.cwd(), trainer.train_fn
try:
    os.chdir(CODE)
    for env_name in ENVS:
        captured = {}
        def capture(rank, world_size, algo, seed, algo_cfg, env_cfg, nets_cfg, log_cfg, device=-1):
            captured.update({
                "rank": rank, "world_size": world_size, "algo": algo, "seed": seed,
                "device": device, "algo_config": vars(algo_cfg), "env_config": vars(env_cfg),
                "nets_config": vars(nets_cfg), "log_config": vars(log_cfg),
            })
        trainer.train_fn = capture
        sys.argv = [str(TRAINER), "--config", CONFIG.name, "--env_name", env_name, "--seed", "0", "--device", "0"]
        trainer.main()
        if captured["env_config"]["env_name"] != env_name:
            raise RuntimeError("resolved environment mismatch: " + env_name)
        resolved[env_name] = captured
finally:
    trainer.train_fn = original_train
    sys.argv = original_argv
    os.chdir(original_cwd)

origins = audit_loaded_modules(BUNDLE, MANIFEST, DESIGNATED, FORBIDDEN)
after = snapshot_empty_directory(DESIGNATED, "after_import", PRESTART)
canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
write_json(ORIGIN_OUT, {
    "result": "IMPORT_ORIGIN_AUDIT_PASS", "origin_policy": policy_ledger,
    "designated_prestart": PRESTART, "designated_interpreter_start": before,
    "designated_after_import": after, "sys_path": path_manifest, **origins,
})
write_json(OUT, {
    "result": "CLEAN_ROOM_BUNDLE_AUDIT_PASS", "origin_policy": policy_ledger,
    "bundle_root": str(BUNDLE), "trainer_sha256": EXPECTED_TRAINER,
    "config_sha256": EXPECTED_CONFIG,
    "bundle_archive_sha256": "3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f",
    "bundle_manifest_sha256": "99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa",
    "four_environment_resolved_config_sha256": hashlib.sha256(canonical).hexdigest(),
    "resolved_environments": ENVS, "import_origin_manifest": str(ORIGIN_OUT),
})
print("TASK23_PSEUDO_ORIGIN_FORMAL_AUDIT_PASS")
print("CLEAN_ROOM_BUNDLE_AUDIT_PASS")
print("IMPORT_ORIGIN_AUDIT_PASS")
