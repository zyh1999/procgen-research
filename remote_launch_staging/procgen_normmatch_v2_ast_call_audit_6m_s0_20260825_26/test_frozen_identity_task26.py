#!/usr/bin/env python3
"""Task26 immutable scientific/audit identities and authorized delta."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T14 = ROOT / "procgen_paper_hybrid_head_normmatch_detggn_6m_s0_20260825_14"
T15 = ROOT / "procgen_normmatch_v2_hermetic_bundle_6m_s0_20260825_15"
T23 = ROOT / "procgen_normmatch_v2_torch_pseudo_origin_nonreentrant_closure_20260825_23"
T25 = ROOT / "procgen_normmatch_v2_torch_class_attribute_pseudo_origin_20260825_25"
EXPECTED = {
    T14 / "train_shared_paper_hybrid_head_detggn_papernorm_v2.py": "0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b",
    T14 / "adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml": "9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda",
    T14 / "gpuh_preflight_normmatch_v2.py": "b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc",
    T14 / "test_hybrid_head_normmatch_v2.py": "f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c",
    T14 / "stage_monitor.py": "536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e",
    T15 / "bundle/normmatch_v2_source_3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f.tar": "3da17520965bc16feccccad0fd334161b60471e5744327aaed93701710f73f6f",
    T15 / "bundle/BUNDLE_MANIFEST.json": "99191542a38f77006b3a7f52aaa8223e7f957a7894fc89cc489a9dae112d46aa",
    T15 / "normmatch_v2_6m_hermetic_gpuh.sbatch": "ec60864aaa9940fd61eb1391008b50f2f48402eeae8f78c91c0b0c1fc313a398",
    T15 / "normmatch_v2_preflight_hermetic_gpuh.sbatch": "374d24881d108bbdd08dee0880b7392a7cc3adf6af177f6eb4deaac15535b228",
    T23 / "nonreentrant_audit_hook.py": "8d9206a6defc4525114398a952d29ffdd4872cd933dc5c9b96fc838bd1273dbe",
    T25 / "class_attribute_classifier.py": "f80de2abbcbce29e7a57ef456156c86636798c4e1ea37171922b3b466b6790fc",
}
for path, expected in EXPECTED.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError("frozen identity mismatch: " + str((path, actual, expected)))
print("TASK26_FROZEN_IDENTITIES_PASS")

