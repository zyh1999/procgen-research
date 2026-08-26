#!/usr/bin/env python3
"""Single production-network preflight for Task41."""
import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from manifest_oracle import (CONFIG_SHA256, TRAINER_SHA256, build_manifest,
                             canonical_bytes, compare_manifest,
                             run_negative_tests, sha256_file)


def spectrum(matrix):
    values = torch.linalg.eigvalsh(matrix).clamp_min(0)
    positive = values[values > values.max() * 1e-12]
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "condition": float(values.max() / positive.min()) if positive.numel() else float("inf"),
        "effective_rank": int(positive.numel()),
    }


def resolve_production_observation_semantics(shape, expected_shape=None):
    shape = tuple(int(value) for value in shape)
    if len(shape) != 3:
        raise ValueError(f"Procgen RGB observation must have H,W,C, got {shape}")
    height, width, channels = shape
    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError(f"observation shape drift: {shape} != {tuple(expected_shape)}")
    if height <= 0 or width <= 0 or channels <= 0 or height != width:
        raise ValueError(f"invalid production Procgen spatial identity: {shape}")
    if channels != 3 or height == channels:
        raise ValueError(f"invalid Procgen RGB/channel-as-image-size identity: {shape}")
    return height, (channels, height, width), "HWC_to_CHW"


def run_shape_negative_tests(observed_shape):
    cases = {
        "missing_height_width": (3,),
        "channels_first_swapped": (3, observed_shape[0], observed_shape[1]),
        "nonproduction_spatial": (observed_shape[0] // 2, observed_shape[1] // 2, 3),
        "wrong_channel_count": (observed_shape[0], observed_shape[1], 4),
    }
    rejected = {}
    for name, candidate in cases.items():
        try:
            resolve_production_observation_semantics(candidate, observed_shape)
        except ValueError:
            rejected[name] = True
        else:
            raise AssertionError(f"negative shape case accepted: {name}")
    image_size, _, _ = resolve_production_observation_semantics(observed_shape, observed_shape)
    if image_size == observed_shape[2]:
        raise AssertionError("channel count accepted as image size")
    rejected["channel_as_image_size"] = True
    return rejected


def load_trainer(path):
    root = Path(path).resolve().parent
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("task41_frozen_trainer", path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    if Path(trainer.__file__).resolve() != Path(path).resolve():
        raise AssertionError("ambient trainer fallback")
    return trainer, root


def construct_production(args):
    if sha256_file(args.trainer) != TRAINER_SHA256:
        raise AssertionError("frozen trainer hash drift")
    if sha256_file(args.config) != CONFIG_SHA256:
        raise AssertionError("frozen config hash drift")
    trainer, root = load_trainer(args.trainer)
    import yaml
    from utils.utils import SharedActorCritic, build_resnet
    for obj in (SharedActorCritic, build_resnet):
        if root not in Path(importlib.util.find_spec(obj.__module__).origin).resolve().parents:
            raise AssertionError(f"ambient local-module fallback: {obj.__module__}")
    cfg = yaml.safe_load(Path(args.config).read_text())
    acfg = types.SimpleNamespace(**cfg["algo_config"])
    ncfg = types.SimpleNamespace(**cfg["nets_config"])
    trainer.validate_paper_matched_config(acfg)
    env_cfg = types.SimpleNamespace(**cfg["env_config"])
    base_env, distribution_mode, start_level, num_levels = args.env_name.split("-")
    venv = trainer.ProcgenEnv(
        num_envs=env_cfg.num_envs, env_name=base_env, num_levels=int(num_levels),
        start_level=int(start_level), distribution_mode=distribution_mode, rand_seed=0)
    venv = trainer.VecExtractDictObs(venv, "rgb")
    venv = trainer.VecMonitor(venv=venv, filename=None)
    observed_shape = tuple(int(value) for value in venv.observation_space.shape)
    image_size, model_input_shape, layout = resolve_production_observation_semantics(observed_shape)
    negative_shapes = run_shape_negative_tests(observed_shape)
    action_count = int(venv.action_space.n)
    torch.manual_seed(390026)
    device = torch.device("cuda:0")
    fn, _ = build_resnet(image_size, ncfg.hidden_size, with_bn=ncfg.with_bn,
                         depths=[8, 16], device=device)
    model = SharedActorCritic(
        fn, model_input_shape, nets_config=ncfg, n_actions=action_count,
        dim_actions=None, with_popart=acfg.with_popart,
        sigma_type=acfg.sigma_type, device=device).to(device).eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=acfg.lr, momentum=1e-6)
    venv.close()
    manifest = build_manifest(model, optimizer, args.trainer, args.config,
                              observed_shape, model_input_shape, image_size)
    return trainer, model, optimizer, manifest, {
        "observation_space_shape": observed_shape,
        "model_input_shape": model_input_shape,
        "observation_layout": layout,
        "resnet_image_size_argument": image_size,
        "action_count": action_count,
        "shape_negative_tests": negative_shapes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--emit-oracle", action="store_true")
    parser.add_argument("--oracle")
    parser.add_argument("--oracle-sha256")
    args = parser.parse_args()
    trainer, model, optimizer, actual_manifest, construction = construct_production(args)
    if args.emit_oracle:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(canonical_bytes(actual_manifest))
        print("TASK41_PRODUCTION_MANIFEST_ORACLE_EMITTED", hashlib.sha256(output.read_bytes()).hexdigest())
        return

    if not args.oracle or not args.oracle_sha256:
        raise ValueError("formal preflight requires --oracle and --oracle-sha256")
    oracle_bytes = Path(args.oracle).read_bytes()
    if hashlib.sha256(oracle_bytes).hexdigest() != args.oracle_sha256:
        raise AssertionError("oracle file hash mismatch")
    oracle = json.loads(oracle_bytes)
    if canonical_bytes(oracle) != oracle_bytes:
        raise AssertionError("oracle is not canonical or was hand-edited")
    compare_manifest(actual_manifest, oracle)
    negative_oracle_tests = run_negative_tests(oracle)

    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    optimizer_trainable = [parameter for group in optimizer.param_groups
                           for parameter in group["params"] if parameter.requires_grad]
    if len(named) != len(optimizer_trainable):
        raise AssertionError("optimizer/autograd trainable length mismatch")
    for index, ((name, parameter), optimizer_parameter) in enumerate(zip(named, optimizer_trainable)):
        if parameter is not optimizer_parameter or name != oracle["ordered_joint2b_column_names"][index]:
            raise AssertionError(f"optimizer/autograd/oracle object identity mismatch: {index}")
    params = {name: parameter.detach() for name, parameter in named}
    buffers = {name: buffer.detach() for name, buffer in model.named_buffers()}
    batch = 512
    model_input_shape = tuple(construction["model_input_shape"])
    observations = torch.randint(0, 256, (batch,) + model_input_shape,
                                 device="cuda:0", dtype=torch.uint8).float()
    actions = torch.arange(batch, device="cuda:0") % construction["action_count"]

    def logp_one(ps, bs, observation, action):
        _, logits = functional_call(model, (ps, bs), (observation[None],))
        return torch.log_softmax(logits, -1)[0, action]

    def value_one(ps, bs, observation):
        value, _ = functional_call(model, (ps, bs), (observation[None],))
        return value.reshape(())

    actor_tree = vmap(grad(logp_one), in_dims=(None, None, 0, 0), randomness="different")(
        params, buffers, observations, actions)
    critic_tree = vmap(grad(value_one), in_dims=(None, None, 0), randomness="different")(
        params, buffers, observations)
    actor_rows = torch.cat([actor_tree[name].reshape(batch, -1) for name, _ in named], 1)
    critic_rows = torch.cat([critic_tree[name].reshape(batch, -1) for name, _ in named], 1)
    if actor_rows.shape != critic_rows.shape or actor_rows.shape[0] + critic_rows.shape[0] != 1024:
        raise AssertionError("strict Joint-2B row identity failure")
    names = [name for name, _ in named]
    policy_indices = [index for index, name in enumerate(names) if name.startswith("pi_head.")]
    value_indices = [index for index, name in enumerate(names) if name.startswith("last_v_layer.")]
    offsets = np.cumsum([0] + [parameter.numel() for _, parameter in named])
    def columns(indices):
        return torch.cat([torch.arange(offsets[i], offsets[i + 1], device="cuda:0") for i in indices])
    policy_columns, value_columns = columns(policy_indices), columns(value_indices)
    if torch.count_nonzero(actor_rows[:, value_columns]) or torch.count_nonzero(critic_rows[:, policy_columns]):
        raise AssertionError("head role/Jacobian support drift")
    shared_mask = torch.ones(actor_rows.shape[1], dtype=torch.bool, device="cuda:0")
    shared_mask[policy_columns] = False
    shared_mask[value_columns] = False
    if not (torch.linalg.vector_norm(actor_rows[:, shared_mask]) > 0 and
            torch.linalg.vector_norm(critic_rows[:, shared_mask]) > 0):
        raise AssertionError("full shared Jacobian coverage missing")

    torch.manual_seed(390027)
    actor_rhs = torch.randn(batch, device="cuda:0")
    critic_rhs = torch.randn(batch, device="cuda:0")
    direction, info = trainer.solve_full_shared_joint2b_scale_recovery(
        actor_rows, critic_rows, actor_rhs, critic_rhs, 0.5)
    gram = info["gram"]
    direct_blocks = [[info["a_bar"] @ info["a_bar"].t(), info["a_bar"] @ info["c_bar"].t()],
                     [info["c_bar"] @ info["a_bar"].t(), info["c_bar"] @ info["c_bar"].t()]]
    torch.testing.assert_close(gram, torch.cat([torch.cat(row, 1) for row in direct_blocks], 0),
                               rtol=0, atol=0)
    if torch.linalg.matrix_norm(direct_blocks[0][1], ord="fro") <= 0:
        raise AssertionError("natural cross block is zero")
    direct = info["h_bar"].t() @ torch.linalg.solve(info["system"], info["b_bar"])
    torch.testing.assert_close(direction.double(), direct, rtol=2e-5, atol=2e-6)
    if abs(float(info["actor_normalized_mean_gram_diag"]) - 1) >= 1e-12:
        raise AssertionError("actor normalized scale drift")
    if abs(float(info["critic_normalized_mean_gram_diag"]) - 1) >= 1e-12:
        raise AssertionError("critic normalized scale drift")
    if int(info["cholesky_info"]) != 0 or float(info["relative_residual"]) >= 1e-10:
        raise AssertionError("Joint-2B solver failure")

    rescaling_errors = {}
    for name, (actor_scale, critic_scale) in {
            "overall": (7.0, 7.0), "actor_only": (11.0, 1.0),
            "critic_only": (1.0, 13.0), "opposite": (0.125, 16.0)}.items():
        scaled, scaled_info = trainer.solve_full_shared_joint2b_scale_recovery(
            actor_scale * actor_rows, critic_scale * critic_rows,
            actor_scale * actor_rhs, critic_scale * critic_rhs, 0.5)
        error = torch.linalg.vector_norm(scaled.double() - direction.double()) / (
            torch.linalg.vector_norm(direction.double()) + 1e-30)
        rescaling_errors[name] = float(error)
        if float(error) >= 3e-6:
            raise AssertionError(f"positive block-rescaling invariance failure: {name}")
        torch.testing.assert_close(scaled_info["gram"], gram, rtol=1e-12, atol=1e-12)
    popart_direction, _ = trainer.solve_full_shared_joint2b_scale_recovery(
        actor_rows, 3.75 * critic_rows, actor_rhs, 3.75 * critic_rhs, 0.5)
    torch.testing.assert_close(popart_direction, direction, rtol=3e-6, atol=2e-6)

    actor_component = info["a_bar"].t() @ info["z"][:batch]
    critic_component = info["c_bar"].t() @ info["z"][batch:]
    block_norms = {
        "shared_actor": float(torch.linalg.vector_norm(actor_component[shared_mask])),
        "shared_critic": float(torch.linalg.vector_norm(critic_component[shared_mask])),
        "policy_delta": float(torch.linalg.vector_norm(direction[policy_columns])),
        "value_delta": float(torch.linalg.vector_norm(direction[value_columns])),
    }
    if not all(value > 0 for value in block_norms.values()):
        raise AssertionError("production reconstructed delta lacks required block contribution")
    if not torch.isfinite(direction).all():
        raise AssertionError("nonfinite production direction")

    result = {
        "status": "PRECHECK_PASS",
        "marker": "GPUH_FULL_SHARED_JOINT2B_MANIFEST_ORACLE_RECOVERY_PASS",
        "env": args.env_name,
        "oracle_sha256": args.oracle_sha256,
        "manifest_counts": actual_manifest["counts"],
        "three_parameter_difference_explanation": actual_manifest["three_parameter_difference_explanation"],
        "optimizer_autograd_joint2b_object_identity": "PASS",
        "construction": construction,
        "oracle_negative_tests": negative_oracle_tests,
        "actor_rows": batch,
        "critic_rows": batch,
        "joint_rows": 2 * batch,
        "joint_columns": actor_rows.shape[1],
        "s_pi": float(info["s_pi"]),
        "s_v": float(info["s_v"]),
        "normalized_actor_mean_diag": float(info["actor_normalized_mean_gram_diag"]),
        "normalized_critic_mean_diag": float(info["critic_normalized_mean_gram_diag"]),
        "relative_damping": 0.5,
        "relative_residual": float(info["relative_residual"]),
        "cholesky_info": int(info["cholesky_info"]),
        "cross_frobenius": float(torch.linalg.matrix_norm(direct_blocks[0][1], ord="fro")),
        "rescaling_relative_errors": rescaling_errors,
        "popart_affine_invariance": "PASS",
        "block_delta_norms": block_norms,
        "raw_actor_spectrum": spectrum(actor_rows.double() @ actor_rows.double().t()),
        "raw_critic_spectrum": spectrum(critic_rows.double() @ critic_rows.double().t()),
        "normalized_joint_spectrum": spectrum(gram),
        "hard_nonfinite": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("GPUH_FULL_SHARED_JOINT2B_MANIFEST_ORACLE_RECOVERY_PASS")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
