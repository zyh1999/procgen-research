#!/usr/bin/env python3
"""One-shot functional preflight for Task39 on the production network."""
import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call, grad, vmap


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spectrum(x):
    values = torch.linalg.eigvalsh(x).clamp_min(0)
    positive = values[values > values.max() * 1e-12]
    return {
        "min": float(values.min()), "max": float(values.max()),
        "condition": float(values.max() / positive.min()) if positive.numel() else float("inf"),
        "effective_rank": int(positive.numel()),
    }


def resolve_production_observation_semantics(shape, expected_shape=None):
    """Validate the real Procgen HWC space and return trainer constructor args."""
    shape = tuple(int(v) for v in shape)
    if len(shape) != 3:
        raise ValueError(f'Procgen RGB observation must have H,W,C, got {shape}')
    height, width, channels = shape
    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError(f'observation shape drift: {shape} != {tuple(expected_shape)}')
    if height <= 0 or width <= 0 or channels <= 0:
        raise ValueError(f'nonpositive observation dimension: {shape}')
    if height != width:
        raise ValueError(f'production Procgen spatial dimensions differ: {shape}')
    if channels != 3:
        raise ValueError(f'production Procgen RGB channel identity differs: {shape}')
    image_size = height
    if image_size == channels:
        raise ValueError('channel count cannot be used as ResNet image size')
    model_input_shape = (channels, height, width)
    return image_size, model_input_shape, 'HWC_to_CHW'


def run_shape_negative_tests(observed_shape):
    rejected = {}
    cases = {
        'missing_height_width': (3,),
        'channels_first_swapped': (3, observed_shape[0], observed_shape[1]),
        'nonproduction_spatial': (observed_shape[0] // 2, observed_shape[1] // 2, 3),
        'wrong_channel_count': (observed_shape[0], observed_shape[1], 4),
    }
    for name, candidate in cases.items():
        try:
            resolve_production_observation_semantics(candidate, observed_shape)
        except ValueError:
            rejected[name] = True
        else:
            raise AssertionError(f'negative observation-shape case accepted: {name}')
    try:
        image_size, _, _ = resolve_production_observation_semantics(observed_shape, observed_shape)
        if image_size == observed_shape[2]:
            raise AssertionError('channel count accepted as image size')
        rejected['channel_as_image_size'] = True
    except ValueError:
        rejected['channel_as_image_size'] = True
    return rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.trainer).resolve().parent
    sys.path.insert(0, str(root))
    import yaml
    import importlib.util
    spec = importlib.util.spec_from_file_location("task39_trainer", args.trainer)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    from utils.utils import SharedActorCritic, build_resnet

    cfg = yaml.safe_load(Path(args.config).read_text())
    acfg = types.SimpleNamespace(**cfg["algo_config"])
    ncfg = types.SimpleNamespace(**cfg["nets_config"])
    trainer.validate_paper_matched_config(acfg)
    torch.manual_seed(390026)
    device = torch.device("cuda:0")
    # Use the same real Procgen environment and exact observation/model path as
    # train_fn. No mock space, constant image size, reduced network or encoder
    # replacement is permitted.
    env_cfg = types.SimpleNamespace(**cfg['env_config'])
    env_cfg.env_name = args.env_name
    base_env, distribution_mode, start_level, num_levels = args.env_name.split('-')
    start_level, num_levels = int(start_level), int(num_levels)
    venv = trainer.ProcgenEnv(
        num_envs=env_cfg.num_envs, env_name=base_env, num_levels=num_levels,
        start_level=start_level, distribution_mode=distribution_mode, rand_seed=0)
    venv = trainer.VecExtractDictObs(venv, 'rgb')
    venv = trainer.VecMonitor(venv=venv, filename=None)
    observed_shape = tuple(int(v) for v in venv.observation_space.shape)
    image_size, model_input_shape, layout = resolve_production_observation_semantics(observed_shape)
    negative_shape_tests = run_shape_negative_tests(observed_shape)
    act_num = int(venv.action_space.n)
    fn, _ = build_resnet(image_size, ncfg.hidden_size, with_bn=ncfg.with_bn,
                         depths=[8, 16], device=device)
    model = SharedActorCritic(fn, model_input_shape, nets_config=ncfg, n_actions=act_num,
                              dim_actions=None, with_popart=acfg.with_popart,
                              sigma_type=acfg.sigma_type, device=device).to(device).eval()
    venv.close()
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = {n: p.detach() for n, p in named}
    buffers = {n: b.detach() for n, b in model.named_buffers() if b.requires_grad}
    assert list(params) == [n for n, _ in named]
    assert all(params[n].shape == p.shape and params[n].dtype == p.dtype and params[n].device == p.device
               for n, p in named)
    total_parameters = int(sum(p.numel() for _, p in named))
    if total_parameters != 938979:
        raise RuntimeError(f'production parameter manifest drift: {total_parameters} != 938979')

    batch = 512
    obs = torch.randint(0, 256, (batch,) + model_input_shape,
                        device=device, dtype=torch.uint8).float()
    act = torch.arange(batch, device=device) % act_num

    def logp_one(ps, bs, x, a):
        _, logits = functional_call(model, (ps, bs), (x[None],))
        return torch.log_softmax(logits, -1)[0, a]

    def value_one(ps, bs, x):
        value, _ = functional_call(model, (ps, bs), (x[None],))
        return value.reshape(())

    a_tree = vmap(grad(logp_one), in_dims=(None, None, 0, 0), randomness="different")(
        params, buffers, obs, act)
    c_tree = vmap(grad(value_one), in_dims=(None, None, 0), randomness="different")(
        params, buffers, obs)
    a = torch.cat([a_tree[n].reshape(batch, -1) for n, _ in named], 1)
    c = torch.cat([c_tree[n].reshape(batch, -1) for n, _ in named], 1)
    assert a.shape[0] + c.shape[0] == 1024 and a.shape == c.shape
    names = [n for n, _ in named]
    policy_idx = [i for i, n in enumerate(names) if "last_pi_layer" in n]
    value_idx = [i for i, n in enumerate(names) if "last_v_layer" in n]
    assert policy_idx and value_idx
    offsets = np.cumsum([0] + [p.numel() for _, p in named])
    def columns(indices):
        return torch.cat([torch.arange(offsets[i], offsets[i + 1], device=device) for i in indices])
    pcols, vcols = columns(policy_idx), columns(value_idx)
    assert torch.count_nonzero(a[:, vcols]) == 0
    assert torch.count_nonzero(c[:, pcols]) == 0
    shared_mask = torch.ones(a.shape[1], dtype=torch.bool, device=device)
    shared_mask[pcols] = False; shared_mask[vcols] = False
    assert torch.linalg.vector_norm(a[:, shared_mask]) > 0
    assert torch.linalg.vector_norm(c[:, shared_mask]) > 0

    torch.manual_seed(390027)
    bpi = torch.randn(batch, device=device)
    bv = torch.randn(batch, device=device)
    direction, info = trainer.solve_full_shared_joint2b_scale_recovery(a, c, bpi, bv, 0.5)
    h = info["h_bar"]
    gram = info["gram"]
    assert torch.equal(gram[:batch, :batch], info["a_bar"] @ info["a_bar"].t())
    assert torch.equal(gram[batch:, batch:], info["c_bar"] @ info["c_bar"].t())
    assert torch.equal(gram[:batch, batch:], info["a_bar"] @ info["c_bar"].t())
    assert torch.equal(gram[batch:, :batch], info["c_bar"] @ info["a_bar"].t())
    direct = h.t() @ torch.linalg.solve(info["system"], info["b_bar"])
    torch.testing.assert_close(direction.double(), direct, rtol=2e-5, atol=2e-6)
    assert abs(float(info["actor_normalized_mean_gram_diag"]) - 1.0) < 1e-12
    assert abs(float(info["critic_normalized_mean_gram_diag"]) - 1.0) < 1e-12
    assert float(info["relative_residual"]) < 1e-10 and int(info["cholesky_info"]) == 0

    rescale_errors = {}
    cases = {"overall": (7.0, 7.0), "actor_only": (11.0, 1.0),
             "critic_only": (1.0, 13.0), "opposite": (0.125, 16.0)}
    for key, (ca, cc) in cases.items():
        d2, i2 = trainer.solve_full_shared_joint2b_scale_recovery(
            ca * a, cc * c, ca * bpi, cc * bv, 0.5)
        err = torch.linalg.vector_norm(d2.double() - direction.double()) / (
            torch.linalg.vector_norm(direction.double()) + 1e-30)
        rescale_errors[key] = float(err)
        assert float(err) < 3e-6
        torch.testing.assert_close(i2["gram"], gram, rtol=1e-12, atol=1e-12)

    # PopArt affine-scale invariance is exactly the critic-only joint rescaling
    # identity, with unchanged unnormalized prediction movement.
    popart_scale = 3.75
    pop_dir, _ = trainer.solve_full_shared_joint2b_scale_recovery(
        a, popart_scale * c, bpi, popart_scale * bv, 0.5)
    torch.testing.assert_close(pop_dir, direction, rtol=3e-6, atol=2e-6)

    actor_component = info["a_bar"].t() @ info["z"][:batch]
    critic_component = info["c_bar"].t() @ info["z"][batch:]
    shared_actor = torch.linalg.vector_norm(actor_component[shared_mask])
    shared_critic = torch.linalg.vector_norm(critic_component[shared_mask])
    assert shared_actor > 0 and shared_critic > 0
    assert torch.linalg.vector_norm(direction[pcols]) > 0
    assert torch.linalg.vector_norm(direction[vcols]) > 0
    unclipped = torch.linalg.vector_norm(direction)
    clip_scale = min(1.0, 0.5 / (float(unclipped) + 1e-30))

    result = {
        "status": "PRECHECK_PASS", "marker": "GPUH_FULL_SHARED_JOINT2B_SCALE_RECOVERY_PASS",
        "env": args.env_name, "trainer_sha256": sha(args.trainer), "config_sha256": sha(args.config),
        "total_parameters": total_parameters, "parameter_tensors": len(named),
        "observation_space_shape": observed_shape, "observation_layout": layout,
        "resnet_image_size_argument": image_size,
        "model_input_shape": model_input_shape, "action_count": act_num,
        "shape_negative_tests": negative_shape_tests,
        "actor_rows": batch, "critic_rows": batch, "joint_rows": 2 * batch,
        "policy_head_tensors": [names[i] for i in policy_idx],
        "value_head_tensors": [names[i] for i in value_idx],
        "s_pi": float(info["s_pi"]), "s_v": float(info["s_v"]),
        "normalized_actor_mean_diag": float(info["actor_normalized_mean_gram_diag"]),
        "normalized_critic_mean_diag": float(info["critic_normalized_mean_gram_diag"]),
        "relative_damping": 0.5, "relative_residual": float(info["relative_residual"]),
        "cholesky_info": int(info["cholesky_info"]), "rescaling_relative_errors": rescale_errors,
        "raw_actor_spectrum": spectrum(a.double() @ a.double().t()),
        "raw_critic_spectrum": spectrum(c.double() @ c.double().t()),
        "normalized_joint_spectrum": spectrum(gram),
        "cross_frobenius": float(torch.linalg.matrix_norm(gram[:batch, batch:], ord="fro")),
        "shared_actor_contribution_norm": float(shared_actor),
        "shared_critic_contribution_norm": float(shared_critic),
        "policy_head_delta_norm": float(torch.linalg.vector_norm(direction[pcols])),
        "value_head_delta_norm": float(torch.linalg.vector_norm(direction[vcols])),
        "unclipped_delta_norm": float(unclipped), "global_clip_scale": clip_scale,
        "popart_affine_invariance": "PASS", "hard_nonfinite": False,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("GPUH_FULL_SHARED_JOINT2B_SCALE_RECOVERY_PASS")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
