#!/usr/bin/env python3
"""Run inside a clean process after the bundle path is explicitly installed."""
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import sysconfig
from pathlib import Path

BUNDLE = Path(os.environ["BUNDLE_ROOT"]).resolve()
CODE = BUNDLE / "code"
OUT = Path(os.environ["AUDIT_OUTPUT"])
TRAINER = CODE / "train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
CONFIG = CODE / "configs/adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml"
EXPECTED_TRAINER = "0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b"
EXPECTED_CONFIG = "9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda"
ENVS = ["bigfish-easy-0-10", "bossfight-easy-0-10", "caveflyer-easy-0-10", "coinrun-easy-0-10"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha(TRAINER) != EXPECTED_TRAINER or sha(CONFIG) != EXPECTED_CONFIG:
    raise RuntimeError("frozen trainer/config hash mismatch")
if os.environ.get("PYTHONNOUSERSITE") != "1":
    raise RuntimeError("PYTHONNOUSERSITE must be 1")

resolved_sys_path = [str(Path(item or os.getcwd()).resolve()) for item in sys.path]
allowed_prefixes = {
    str(BUNDLE), str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve()),
    *(str(Path(item).resolve()) for item in sysconfig.get_paths().values() if item),
}
for item in resolved_sys_path:
    if not any(item == prefix or item.startswith(prefix + os.sep) for prefix in allowed_prefixes):
        raise RuntimeError(f"non-hermetic sys.path entry: {item}")

spec = importlib.util.spec_from_file_location("normmatch_v2_trainer", TRAINER)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)

local_modules = {}
for name in [
    "utils.logger", "utils.runners", "utils.utils", "utils.vision_transformers",
    "utils.vit", "utils.resnet", "utils.convnet", "utils.popart",
    "utils.running_mean_std", "utils.transformer", "utils.rope",
    "utils.seq_running_mean_std", "utils.monitor", "vec_env",
    "vec_env.vec_env", "vec_env.shmem_vec_env", "vec_env.subproc_vec_env",
    "vec_env.vec_monitor", "vec_env.vec_normalize", "vec_env.vec_remove_dict_obs",
    "vec_env.dummy_vec_env", "vec_env.util",
]:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if CODE not in path.parents:
        raise RuntimeError(f"local module escaped bundle: {name} -> {path}")
    local_modules[name] = str(path)

resolved = {}
original_argv, original_cwd, original_train = sys.argv[:], Path.cwd(), trainer.train_fn
try:
    os.chdir(CODE)
    for env_name in ENVS:
        captured = {}

        def capture(rank, world_size, algo, seed, algo_cfg, env_cfg, nets_cfg, log_cfg, device=-1):
            captured.update({
                "rank": rank, "world_size": world_size, "algo": algo, "seed": seed,
                "device": device, "algo_config": vars(algo_cfg),
                "env_config": vars(env_cfg), "nets_config": vars(nets_cfg),
                "log_config": vars(log_cfg),
            })

        trainer.train_fn = capture
        sys.argv = [str(TRAINER), "--config", CONFIG.name, "--env_name", env_name, "--seed", "0", "--device", "0"]
        trainer.main()
        if captured["env_config"]["env_name"] != env_name:
            raise RuntimeError(f"resolved environment mismatch: {env_name}")
        resolved[env_name] = captured
finally:
    trainer.train_fn = original_train
    sys.argv = original_argv
    os.chdir(original_cwd)

canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
payload = {
    "result": "CLEAN_ROOM_BUNDLE_AUDIT_PASS",
    "bundle_root": str(BUNDLE),
    "sys_path": resolved_sys_path,
    "utils_logger_path": local_modules["utils.logger"],
    "utils_utils_path": local_modules["utils.utils"],
    "vec_env_path": local_modules["vec_env"],
    "repository_local_modules": local_modules,
    "trainer_sha256": sha(TRAINER),
    "config_sha256": sha(CONFIG),
    "four_environment_resolved_config_sha256": hashlib.sha256(canonical).hexdigest(),
    "resolved_environments": ENVS,
}
OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("CLEAN_ROOM_BUNDLE_AUDIT_PASS")
print(f"utils_logger_path={local_modules['utils.logger']}")
print(f"utils_utils_path={local_modules['utils.utils']}")
print(f"vec_env_path={local_modules['vec_env']}")
print(f"resolved_config_sha256={payload['four_environment_resolved_config_sha256']}")
