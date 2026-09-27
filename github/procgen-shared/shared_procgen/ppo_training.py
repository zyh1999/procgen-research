"""Two fixed-LR shared PPO baselines; separate from all RAT profiles."""
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
from .ppo import LR, ADAM_EPS, update_ppo
from .rollout import Runner

PROFILES = {"normal_ppo": (512, 6_004_736),
            "large8192_ppo": (8192, 15_007_744)}


def build_parser(profile):
    batch, endpoint = PROFILES[profile]
    parser = argparse.ArgumentParser(
        description=f"Shared PPO: B{batch}, endpoint {endpoint}, fixed Adam LR=0.0003")
    parser.add_argument("--env", default="bigfish", choices=["bigfish", "bossfight", "starpilot",
                        "caveflyer", "coinrun", "maze", "jumper", "miner"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True, help="New output directory; must not exist")
    return parser


def profile_config(profile, env, seed):
    batch, endpoint = PROFILES[profile]
    return dict(profile=profile, method="ppo_fixed_lr", shared=True, env=env, seed=seed,
                num_envs=16, nsteps=batch // 2, rollout=8 * batch, minibatch=batch,
                epochs=4, minibatches_per_epoch=8, updates_per_rollout=32,
                num_rollouts=endpoint // (8 * batch), endpoint=endpoint,
                depths=[8, 16], hidden=256, optimizer="Adam", lr=LR, adam_eps=ADAM_EPS,
                controller="none_fixed_lr", lr_schedule="constant", ppo_ratio_clip=.2,
                value_coefficient=.5, critic_ratio=False, value_clipping=False,
                entropy_coefficient=0., max_grad_norm=.5, popart=False,
                advantage_normalization="minibatch_center_then_sample_std",
                loss_normalization="mean_B", physical_chunk=128,
                gamma=.999, gae_lambda=.95, distribution="easy", start_level=0,
                num_levels=10, reward_normalization=False)


def main(profile):
    batch, endpoint = PROFILES[profile]
    args = build_parser(profile).parse_args()
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
    config = profile_config(profile, args.env, args.seed)
    config.update(torch_version=torch.__version__, device=str(device))
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "status").write_text("RUNNING\n")
    env = None
    try:
        env = ProcgenAdapter(args.env, args.seed, device)
        model = SharedActorCritic(env.action_space.n, with_popart=False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=ADAM_EPS)
        runner = Runner(env=env, model=model, nsteps=config["nsteps"], gamma=.999,
                        lam=.95, adv_type="gae", device=device, store_obs_on_cpu=True)
        episode_buffer = deque(maxlen=100)
        started = time.perf_counter()
        with (args.out / "progress.csv").open("w", newline="") as progress, gzip.open(
                args.out / "metric_trace.jsonl.gz", "wt") as trace:
            writer = None
            for rollout_index in range(config["num_rollouts"]):
                model.eval()
                obs, returns, actions, advantages, old_logits, episodes = runner.run()
                episode_buffer.extend(episodes)
                # Ordinary PPO targets; no PopArt transformation or RAT reconstruction.
                model.train()
                indices = np.arange(config["rollout"])
                transition = (rollout_index + 1) * config["rollout"]
                updates = 0
                for epoch_index in range(4):
                    np.random.shuffle(indices)
                    for minibatch_index, start in enumerate(range(0, len(indices), batch)):
                        ix = indices[start:start + batch]
                        metrics = update_ppo(model, optimizer, obs[ix], actions[ix],
                                             advantages[ix], returns[ix], old_logits[ix])
                        metrics.update(transition=transition, rollout_index=rollout_index,
                                       epoch_index=epoch_index, minibatch_index=minibatch_index)
                        trace.write(json.dumps(metrics, allow_nan=False) + "\n")
                        updates += 1
                assert updates == 32
                trace.flush()
                elapsed = time.perf_counter() - started
                row = dict(transition=transition, elapsed=elapsed, fps=transition / elapsed,
                           eprewmean=(float(np.mean([ep["r"] for ep in episode_buffer]))
                                      if episode_buffer else float("nan")),
                           entropy=metrics["entropy"], kl=metrics["kl"], lr=LR,
                           ordinary_value_mse=metrics["ordinary_value_mse"],
                           grad_norm=metrics["grad_norm"], clip_scale=metrics["clip_scale"],
                           ratio_clip_fraction=metrics["ratio_clip_fraction"], updates=updates)
                if writer is None:
                    writer = csv.DictWriter(progress, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                progress.flush()
                temporary = args.out / "checkpoint.tmp"
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                rollout_index=rollout_index, transition=transition, config=config), temporary)
                temporary.replace(args.out / "checkpoint.pt")
                print(f"{transition:,} reward={row['eprewmean']:.3f} KL={row['kl']:.4g} "
                      f"LR={LR:.4g} FPS={row['fps']:.0f}", flush=True)
        (args.out / "status").write_text("PASS\n")
    except BaseException:
        (args.out / "status").write_text("FAILED\n")
        raise
    finally:
        if env is not None:
            env.close()
