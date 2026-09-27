"""Common full-run driver for four fixed shared profiles; no method mixing."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from collections import deque
import csv
import gzip
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .env import ProcgenAdapter
from .model import SharedActorCritic
from .rollout import Runner
from .update import update
from .large_update import update_large

PROFILES = {
    "normal_127p1": ("dual127p1", 512, 6_004_736),
    "normal_128": ("coeff128", 512, 6_004_736),
    "large8192_127p1": ("dual127p1", 8192, 15_007_744),
    "large8192_128": ("coeff128", 8192, 15_007_744),
}


def build_parser(profile):
    """Fixed scientific profile; only the coefficient128 ray ablation is optional."""
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    method, batch, endpoint = PROFILES[profile]
    parser.set_defaults(method=method)
    parser.set_defaults(residual_ray="off")
    if method == "coeff128":
        parser.add_argument("--residual-ray", choices=["on", "off"], default="on",
                            help="Sample residual-ray scaling; off is a separately labeled variant")
    parser.add_argument("--sampling", choices=["uniform"] if batch == 8192 else ["uniform", "leverage"], default="uniform")
    parser.add_argument("--env", default="bigfish", choices=["bigfish", "bossfight", "starpilot",
                        "caveflyer", "coinrun", "maze", "jumper", "miner"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; must not exist")
    return parser


def main(profile):
    method, batch, endpoint = PROFILES[profile]
    args = build_parser(profile).parse_args()
    rollout_size = 8 * batch
    nsteps = rollout_size // 16
    num_rollouts = endpoint // rollout_size
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly if intended")

    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(profile=profile, coefficient_ray_scale=args.residual_ray == "on", method=args.method, sampling=args.sampling, env=args.env, seed=args.seed,
                  num_envs=16, nsteps=nsteps, rollout=rollout_size, minibatch=batch, epochs=4,
                  minibatches_per_epoch=8, updates_per_rollout=32, num_rollouts=num_rollouts,
                  endpoint=endpoint, depths=[8, 16], hidden=256, shared=True,
                  reduced_budget=128, normalization="full_minibatch_B", damping=0.5,
                  ratio_clamp=[0.1, 10], critic_ratio=False, value_coefficient=1.0,
                  entropy_coefficient=0.0, max_grad_norm=0.5, optimizer="SGD", momentum=0,
                  lr=0.5, lr_bounds=[1e-4, 0.5], kl_band=[0.005, 0.04], lr_factor=1.5,
                  controller="post_minibatch_behavior_KL", gamma=0.999, gae_lambda=0.95,
                  distribution="easy", start_level=0, num_levels=10,
                  reward_normalization=False, popart=True, history=False,
                  torch_version=torch.__version__, device=str(device))
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "status").write_text("RUNNING\n")
    env = None
    try:
        env = ProcgenAdapter(args.env, args.seed, device)
        model = SharedActorCritic(env.action_space.n).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.0)
        runner = Runner(env=env, model=model, nsteps=nsteps, gamma=0.999, lam=0.95,
                        adv_type="gae", device=device, store_obs_on_cpu=batch == 8192)
        episode_buffer = deque(maxlen=100)
        indices = np.arange(rollout_size)
        started = time.perf_counter()
        with (args.out / "progress.csv").open("w", newline="") as progress, gzip.open(
                args.out / "metric_trace.jsonl.gz", "wt") as trace:
            writer = None
            for rollout_index in range(num_rollouts):
                model.eval()
                obs, returns, actions, advantages, old_logits, episodes = runner.run()
                episode_buffer.extend(episodes)
                # Preserve the parent's PopArt/advantage normalization order.
                model.last_v_layer.update(returns)
                returns = model.last_v_layer.normalize(returns)
                advantages = model.last_v_layer.normalize(advantages)
                model.train()
                transition = (rollout_index + 1) * rollout_size
                for epoch_index in range(4):
                    np.random.shuffle(indices)
                    for minibatch_index, start in enumerate(range(0, rollout_size, batch)):
                        ix = indices[start:start + batch]
                        updater = update if batch == 512 else update_large
                        extra = {"sampling": args.sampling} if batch == 512 else {}
                        extra["ray_scale"] = args.residual_ray == "on"
                        metrics = updater(model, optimizer, obs[ix], actions[ix], advantages[ix],
                                         returns[ix], old_logits[ix], method=args.method,
                                         seed=args.seed, **extra,
                                         rollout_index=rollout_index, epoch_index=epoch_index,
                                         minibatch_index=minibatch_index)
                        metrics.update(transition=transition, rollout_index=rollout_index,
                                       epoch_index=epoch_index, minibatch_index=minibatch_index)
                        trace.write(json.dumps(metrics, allow_nan=False) + "\n")
                trace.flush()
                elapsed = time.perf_counter() - started
                row = dict(transition=transition, elapsed=elapsed, fps=transition / elapsed,
                           eprewmean=(np.mean([ep["r"] for ep in episode_buffer])
                                      if episode_buffer else float("nan")),
                           entropy=metrics["entropy"], kl=metrics["kl"], lr=metrics["lr_after"],
                           ordinary_value_mse=metrics["ordinary_value_mse"],
                           grad_norm=metrics["grad_norm"], clip_scale=metrics["clip_scale"],
                           actor_residual=metrics["actor_residual"],
                           critic_residual=metrics["critic_residual"], updates=32)
                if writer is None:
                    writer = csv.DictWriter(progress, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                progress.flush()
                # Atomic checkpoint, created only in this new local run directory.
                # Contains optimizer/PopArt, but NOT Procgen environment/RNG state.
                temporary = args.out / "checkpoint.tmp"
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                rollout_index=rollout_index, transition=transition, config=config), temporary)
                temporary.replace(args.out / "checkpoint.pt")
                print(f"{transition:,} reward={row['eprewmean']:.3f} "
                      f"KL={row['kl']:.4g} LR={row['lr']:.4g} FPS={row['fps']:.0f}", flush=True)
        (args.out / "status").write_text("PASS\n")
    except BaseException:
        (args.out / "status").write_text("FAILED\n")
        raise
    finally:
        if env is not None:
            env.close()
