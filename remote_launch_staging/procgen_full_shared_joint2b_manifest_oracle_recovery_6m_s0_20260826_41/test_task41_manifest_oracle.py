#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def load(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = argparse.ArgumentParser()
parser.add_argument("--oracle", required=True)
parser.add_argument("--oracle-sha256", required=True)
args = parser.parse_args()
oracle_module = load("manifest_oracle.py")
preflight_module = load("preflight_full_shared_joint2b_manifest_oracle_recovery.py")
oracle_bytes = Path(args.oracle).read_bytes()
assert hashlib.sha256(oracle_bytes).hexdigest() == args.oracle_sha256
oracle = json.loads(oracle_bytes)
assert oracle_module.canonical_bytes(oracle) == oracle_bytes
assert oracle_module.validate_manifest(oracle)
rejected = oracle_module.run_negative_tests(oracle)
assert set(rejected) == {
    "missing_parameter", "extra_parameter", "duplicate_parameter", "reordered_parameters",
    "shape_drift", "numel_drift", "dtype_drift", "requires_grad_drift",
    "role_drift", "optimizer_joint_mismatch", "nontraining_state_in_solver",
    "same_total_different_member", "trainer_binding_drift", "config_binding_drift",
    "construction_binding_drift",
}
tampered = oracle_bytes.replace(b'"schema"', b'"schemaX"', 1)
assert hashlib.sha256(tampered).hexdigest() != args.oracle_sha256
image_size, model_shape, layout = preflight_module.resolve_production_observation_semantics((64, 64, 3))
assert image_size == 64 and model_shape == (3, 64, 64) and layout == "HWC_to_CHW"
assert len(preflight_module.run_shape_negative_tests((64, 64, 3))) == 5
assert oracle["counts"]["all_parameter_numel"] == 938979
assert oracle["counts"]["trainable_parameter_numel"] == 938976
explanation = oracle["three_parameter_difference_explanation"]
assert explanation["difference_numel"] == 3
assert [entry["name"] for entry in explanation["entries"]] == [
    "last_v_layer.mean", "last_v_layer.mean_sq", "last_v_layer.debiasing_term"]
print("TASK41_MANIFEST_ORACLE_NEGATIVE_GATES_PASS")
