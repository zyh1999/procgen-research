#!/usr/bin/env python3
"""Static guard against regressing to a hand-built preflight model mock."""
import ast
import hashlib
from pathlib import Path

here = Path(__file__).resolve().parent
harness = here / "gpuh_preflight.py"
launcher = here / "hybrid_head_preflight_gpuh.sbatch"
scientific = here / "hybrid_head_detggn_6m_gpuh.sbatch"
trainer = here / "train_shared_paper_hybrid_head_detggn_v1.py"
config = here / "adv_resnet_shared_paper_hybrid_head_detggn_v1_6m.yaml"
if not config.exists():
    trainer = here.parent / "code/train_shared_paper_hybrid_head_detggn_v1.py"
    config = here.parent / "code/configs/adv_resnet_shared_paper_hybrid_head_detggn_v1_6m.yaml"
monitor = here / "stage_monitor.py"

assert hashlib.sha256(trainer.read_bytes()).hexdigest() == "7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54"
assert hashlib.sha256(config.read_bytes()).hexdigest() == "9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda"
assert hashlib.sha256(scientific.read_bytes()).hexdigest() == "ae7104e7b0118cab902d173cd6bb1ea634dc3eb9586fbb5e055c7b8477cceb8e"
assert hashlib.sha256(monitor.read_bytes()).hexdigest() == "536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e"

text = harness.read_text()
tree = ast.parse(text)
assert "SimpleNamespace" not in text
assert "module.main()" in text
assert "original_train_fn" in text
assert "resolved_configuration_three_way=BIT_IDENTICAL" in text
assert "scientific_launcher_dry_run" in text
assert "trainer_entry" in text
assert "capture_model=True" in text
assert "norm_obs" not in text
assert sum(
    isinstance(node, ast.FunctionDef) and node.name == "invoke_main"
    for node in ast.walk(tree)
) == 1
assert "test_canonical_preflight.py" in launcher.read_text()

print("CANONICAL_PREFLIGHT_STATIC_TEST_PASS")
print("scientific_identity_hashes=UNCHANGED")
print("hand_built_namespace=ABSENT")
print("trainer_main_and_original_train_fn=CANONICAL_PATH")
