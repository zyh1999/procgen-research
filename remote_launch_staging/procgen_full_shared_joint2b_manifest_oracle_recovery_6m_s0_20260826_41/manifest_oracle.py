#!/usr/bin/env python3
"""Ordered production parameter-manifest oracle for Task41."""
import copy
import hashlib
import json
import math
from pathlib import Path


TRAINER_SHA256 = "b13b3eae445f83f0e1d6565cb942c8390b6cd143fcc7f366410406c370aa0815"
CONFIG_SHA256 = "1b0cf73885dcb7c61078e463f826d6b296abfedb5a6a019f6782c6b899896b52"
SCIENCE_LAUNCHER_SHA256 = "59dea11cb0fed21974f4985d2eb2016703dd33f06d2e11daa7ca419d5a888fb4"
METHOD = "FULL_SHARED_JOINT2B_SCALE_RECOVERY_V1"


def canonical_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def parameter_role(name, requires_grad):
    if name.startswith("pi_head."):
        return "POLICY_EXCLUSIVE"
    if name in ("last_v_layer.weight", "last_v_layer.bias"):
        return "CRITIC_EXCLUSIVE"
    if not requires_grad:
        return "NONTRAINING_STATE"
    return "SHARED"


def build_manifest(model, optimizer, trainer_path, config_path, observation_shape,
                   model_input_shape, image_size):
    named = list(model.named_parameters())
    model_params = [p for _, p in named]
    optimizer_params = [p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in model_params}) != len(model_params):
        raise AssertionError("duplicate model parameter object")
    if len(optimizer_params) != len(model_params):
        raise AssertionError("optimizer/model parameter length mismatch")
    for index, (model_param, optimizer_param) in enumerate(zip(model_params, optimizer_params)):
        if optimizer_param is not model_param:
            raise AssertionError(f"optimizer object/order mismatch at {index}")

    trainable = [(n, p) for n, p in named if p.requires_grad]
    optimizer_trainable = [(n, p) for n, p in named if p.requires_grad and
                           any(q is p for q in optimizer_params)]
    joint_columns = list(trainable)
    if len(trainable) != len(optimizer_trainable):
        raise AssertionError("trainable optimizer set differs from autograd set")
    for index, ((n1, p1), (n2, p2), (n3, p3)) in enumerate(
            zip(trainable, optimizer_trainable, joint_columns)):
        if n1 != n2 or n1 != n3 or p1 is not p2 or p1 is not p3:
            raise AssertionError(f"optimizer/autograd/Joint-2B identity mismatch at {index}")

    parameters = []
    trainable_position = 0
    for position, (name, parameter) in enumerate(named):
        role = parameter_role(name, bool(parameter.requires_grad))
        entry = {
            "name": name,
            "position": position,
            "shape": list(parameter.shape),
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
            "requires_grad": bool(parameter.requires_grad),
            "role": role,
            "production_optimizer_member": True,
            "production_optimizer_position": position,
            "production_optimizer_trainable_member": bool(parameter.requires_grad),
            "autograd_member": bool(parameter.requires_grad),
            "joint2b_column_member": bool(parameter.requires_grad),
            "joint2b_column_position": trainable_position if parameter.requires_grad else None,
        }
        parameters.append(entry)
        if parameter.requires_grad:
            trainable_position += 1

    buffers = [{
        "name": name,
        "position": position,
        "shape": list(buffer.shape),
        "numel": int(buffer.numel()),
        "dtype": str(buffer.dtype),
        "requires_grad": bool(buffer.requires_grad),
        "role": "NONTRAINING_BUFFER_STATE",
        "production_optimizer_member": False,
        "autograd_member": False,
        "joint2b_column_member": False,
    } for position, (name, buffer) in enumerate(model.named_buffers())]

    nontraining_parameters = [entry for entry in parameters if not entry["requires_grad"]]
    total_numel = sum(entry["numel"] for entry in parameters)
    trainable_numel = sum(entry["numel"] for entry in parameters if entry["requires_grad"])
    nontraining_numel = sum(entry["numel"] for entry in nontraining_parameters)
    manifest = {
        "schema": "task41_production_parameter_manifest_oracle_v1",
        "binding": {
            "method": METHOD,
            "trainer_sha256": sha256_file(trainer_path),
            "config_sha256": sha256_file(config_path),
            "science_launcher_sha256": SCIENCE_LAUNCHER_SHA256,
            "construction_entry": "ProcgenEnv->VecExtractDictObs(rgb)->VecMonitor->build_resnet->SharedActorCritic",
            "observation_shape_hwc": list(observation_shape),
            "model_input_shape_chw": list(model_input_shape),
            "resnet_image_size": int(image_size),
        },
        "counts": {
            "all_parameter_tensors": len(parameters),
            "all_parameter_numel": total_numel,
            "trainable_parameter_tensors": len(trainable),
            "trainable_parameter_numel": trainable_numel,
            "nontraining_parameter_tensors": len(nontraining_parameters),
            "nontraining_parameter_numel": nontraining_numel,
            "buffer_tensors": len(buffers),
            "buffer_numel": sum(entry["numel"] for entry in buffers),
        },
        "three_parameter_difference_explanation": {
            "legacy_all_parameter_numel": total_numel,
            "ordered_trainable_joint2b_numel": trainable_numel,
            "difference_numel": total_numel - trainable_numel,
            "source": "production model named_parameters entries with requires_grad=False",
            "entries": [{"name": e["name"], "numel": e["numel"], "role": e["role"]}
                        for e in nontraining_parameters],
            "semantics": "PopArt running normalization state; retained in model/optimizer container but excluded from autograd, Jacobians, solver columns and delta",
        },
        "parameters": parameters,
        "buffers": buffers,
        "ordered_trainable_names": [name for name, _ in trainable],
        "ordered_optimizer_trainable_names": [name for name, _ in optimizer_trainable],
        "ordered_joint2b_column_names": [name for name, _ in joint_columns],
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if manifest.get("schema") != "task41_production_parameter_manifest_oracle_v1":
        raise AssertionError("oracle schema mismatch")
    binding = manifest["binding"]
    if binding["trainer_sha256"] != TRAINER_SHA256 or binding["config_sha256"] != CONFIG_SHA256:
        raise AssertionError("oracle frozen source binding mismatch")
    if binding["method"] != METHOD or binding["science_launcher_sha256"] != SCIENCE_LAUNCHER_SHA256:
        raise AssertionError("oracle scientific binding mismatch")
    parameters = manifest["parameters"]
    names = [entry["name"] for entry in parameters]
    if len(names) != len(set(names)):
        raise AssertionError("missing/extra/duplicate parameter identity")
    if [entry["position"] for entry in parameters] != list(range(len(parameters))):
        raise AssertionError("parameter ordering drift")
    for entry in parameters:
        if math.prod(entry["shape"]) != entry["numel"]:
            raise AssertionError(f"shape/numel mismatch: {entry['name']}")
        expected_role = parameter_role(entry["name"], entry["requires_grad"])
        if entry["role"] != expected_role:
            raise AssertionError(f"parameter role drift: {entry['name']}")
        if not entry["production_optimizer_member"]:
            raise AssertionError(f"production optimizer membership drift: {entry['name']}")
        expected_trainable = bool(entry["requires_grad"])
        for field in ("production_optimizer_trainable_member", "autograd_member",
                      "joint2b_column_member"):
            if bool(entry[field]) != expected_trainable:
                raise AssertionError(f"trainable/Joint-2B membership drift: {entry['name']}:{field}")
    trainable_names = [e["name"] for e in parameters if e["requires_grad"]]
    if not (manifest["ordered_trainable_names"] ==
            manifest["ordered_optimizer_trainable_names"] ==
            manifest["ordered_joint2b_column_names"] == trainable_names):
        raise AssertionError("optimizer/autograd/Joint-2B ordered collection mismatch")
    counts = manifest["counts"]
    expected_counts = {
        "all_parameter_tensors": len(parameters),
        "all_parameter_numel": sum(e["numel"] for e in parameters),
        "trainable_parameter_tensors": sum(e["requires_grad"] for e in parameters),
        "trainable_parameter_numel": sum(e["numel"] for e in parameters if e["requires_grad"]),
        "nontraining_parameter_tensors": sum(not e["requires_grad"] for e in parameters),
        "nontraining_parameter_numel": sum(e["numel"] for e in parameters if not e["requires_grad"]),
        "buffer_tensors": len(manifest["buffers"]),
        "buffer_numel": sum(e["numel"] for e in manifest["buffers"]),
    }
    if counts != expected_counts:
        raise AssertionError("oracle count ledger mismatch")
    explanation = manifest["three_parameter_difference_explanation"]
    nontraining = [e for e in parameters if not e["requires_grad"]]
    if explanation["difference_numel"] != sum(e["numel"] for e in nontraining):
        raise AssertionError("all/trainable difference explanation mismatch")
    if explanation["entries"] != [{"name": e["name"], "numel": e["numel"], "role": e["role"]}
                                      for e in nontraining]:
        raise AssertionError("nontraining state name-level explanation mismatch")
    return True


def compare_manifest(actual, oracle):
    validate_manifest(actual)
    validate_manifest(oracle)
    if canonical_bytes(actual) != canonical_bytes(oracle):
        raise AssertionError("production manifest differs from frozen ordered oracle")
    return True


def run_negative_tests(oracle):
    mutations = {}
    def reject(name, mutate):
        candidate = copy.deepcopy(oracle)
        mutate(candidate)
        try:
            compare_manifest(candidate, oracle)
        except (AssertionError, KeyError, TypeError):
            mutations[name] = True
        else:
            raise AssertionError(f"negative oracle case accepted: {name}")
    reject("missing_parameter", lambda x: x["parameters"].pop(0))
    reject("extra_parameter", lambda x: x["parameters"].append(dict(
        copy.deepcopy(x["parameters"][0]), name="extra.unique.parameter")))
    reject("duplicate_parameter", lambda x: x["parameters"].append(copy.deepcopy(x["parameters"][0])))
    reject("reordered_parameters", lambda x: x["parameters"].__setitem__(slice(0, 2), list(reversed(x["parameters"][:2]))))
    reject("shape_drift", lambda x: x["parameters"][0]["shape"].__setitem__(0, x["parameters"][0]["shape"][0] + 1))
    reject("numel_drift", lambda x: x["parameters"][0].__setitem__("numel", x["parameters"][0]["numel"] + 1))
    reject("dtype_drift", lambda x: x["parameters"][0].__setitem__("dtype", "torch.float64"))
    reject("requires_grad_drift", lambda x: x["parameters"][0].__setitem__("requires_grad", False))
    reject("role_drift", lambda x: x["parameters"][0].__setitem__("role", "CRITIC_EXCLUSIVE"))
    reject("optimizer_joint_mismatch", lambda x: x["parameters"][0].__setitem__("joint2b_column_member", False))
    reject("nontraining_state_in_solver", lambda x: x["parameters"][-1].__setitem__("joint2b_column_member", True))
    reject("same_total_different_member", lambda x: x["parameters"][0].__setitem__("name", "wrong.same_numel"))
    reject("trainer_binding_drift", lambda x: x["binding"].__setitem__("trainer_sha256", "0" * 64))
    reject("config_binding_drift", lambda x: x["binding"].__setitem__("config_sha256", "0" * 64))
    reject("construction_binding_drift", lambda x: x["binding"].__setitem__("construction_entry", "mock_model"))
    return mutations
