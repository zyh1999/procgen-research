"""Fixed-LR PPO extraction and driver tests; no Procgen training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from test_algebra import TinyShared
from shared_procgen.model import SharedActorCritic, model_step
from shared_procgen.ppo import LR, ADAM_EPS, update_ppo
from shared_procgen.ppo_training import PROFILES, build_parser, profile_config


class PPOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_chunked_gradient_clip_and_adam_match_dense(self):
        for batch in (512, 8192):
            torch.manual_seed(19)
            reference = TinyShared().double()
            actual = copy.deepcopy(reference)
            obs = torch.randn(batch, 4, dtype=torch.float64)
            actions = torch.arange(batch) % 3
            old = torch.randn(batch, 3, dtype=torch.float64)
            advantage = torch.randn(batch, dtype=torch.float64)
            returns = torch.randn(batch, dtype=torch.float64) * 3
            optimizers = [torch.optim.Adam(m.parameters(), lr=LR, eps=ADAM_EPS)
                          for m in (reference, actual)]
            v, z = reference(obs)
            lp = F.log_softmax(z, -1)
            old_lp = F.log_softmax(old, -1)
            ratio = (lp.gather(1, actions[:, None]) - old_lp.gather(1, actions[:, None])).squeeze(1).exp()
            a = advantage - advantage.mean()
            a = a / (a.std() + 1e-8)
            policy = torch.maximum(-ratio * a, -ratio.clamp(.8, 1.2) * a).mean()
            mse = (v - returns).square().mean()
            (policy + .5 * mse).backward()
            norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .5)
            optimizers[0].step()
            metrics = update_ppo(actual, optimizers[1], obs, actions, advantage, returns, old, chunk=37)
            for p, q in zip(reference.parameters(), actual.parameters()):
                torch.testing.assert_close(p, q, atol=1e-12, rtol=1e-11)
            self.assertAlmostEqual(metrics['grad_norm'], float(norm), places=11)
            self.assertAlmostEqual(metrics['policy_loss'], float(policy.detach()), places=11)
            self.assertAlmostEqual(metrics['value_loss'], .5 * float(mse.detach()), places=11)
            for state in optimizers[1].state.values():
                self.assertEqual(int(state['step']), 1)
            self.assertEqual(metrics['loss_divisor'], batch)
            self.assertEqual(metrics['optimizer_steps'], 1)

    def test_kl_does_not_change_lr(self):
        torch.manual_seed(3)
        model = TinyShared()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=ADAM_EPS)
        obs = torch.randn(512, 4)
        for scale in (0., 4., 1.):
            old = torch.randn(512, 3) * scale
            result = update_ppo(model, optimizer, obs, torch.arange(512) % 3,
                                torch.randn(512), torch.randn(512), old)
            self.assertEqual(optimizer.param_groups[0]['lr'], LR)
            self.assertEqual(result['lr_before'], result['lr_after'])
            self.assertEqual(result['controller'], 'none_fixed_lr')
        optimizer.param_groups[0]['lr'] = .5
        with self.assertRaises(ValueError):
            update_ppo(model, optimizer, obs, torch.arange(512) % 3,
                       torch.randn(512), torch.randn(512), old)

    def test_nonfinite_gradient_rejected_before_step(self):
        model = TinyShared()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=ADAM_EPS)
        initial = copy.deepcopy(model.state_dict())
        with self.assertRaises(RuntimeError):
            update_ppo(model, optimizer, torch.randn(512, 4), torch.arange(512) % 3,
                       torch.randn(512), torch.full((512,), float('inf')),
                       torch.randn(512, 3))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[name])
        self.assertFalse(optimizer.state)

    def test_plain_value_head_and_default_popart_unchanged(self):
        torch.manual_seed(2)
        ppo = SharedActorCritic(15, with_popart=False)
        self.assertFalse(ppo.with_popart)
        self.assertIsInstance(ppo.last_v_layer, torch.nn.Linear)
        obs = torch.randn(2, 3, 64, 64)
        values, logits = ppo(obs)
        actions, step_values, step_logits = model_step(ppo, obs, deterministic=True)
        torch.testing.assert_close(values, step_values)
        torch.testing.assert_close(logits, step_logits)
        torch.testing.assert_close(actions, logits.argmax(-1))
        self.assertTrue(SharedActorCritic(15).with_popart)

    def test_profiles_and_cli(self):
        for profile, (batch, endpoint) in PROFILES.items():
            cfg = profile_config(profile, 'miner', 2)
            self.assertEqual(cfg['num_rollouts'] * cfg['rollout'], endpoint)
            self.assertEqual(cfg['rollout'], 8 * batch)
            self.assertEqual(cfg['updates_per_rollout'], 32)
            self.assertFalse(cfg['popart'])
            self.assertEqual(cfg['lr'], LR)
            self.assertEqual(cfg['entropy_coefficient'], 0.)
            args = build_parser(profile).parse_args(['--out', 'new', '--env', 'miner'])
            self.assertFalse(hasattr(args, 'sampling'))
            self.assertFalse(hasattr(args, 'residual_ray'))

    def test_driver_logging_and_endpoint_with_synthetic_runner(self):
        # Tiny deterministic fixture; no Procgen environment or external job.
        from shared_procgen import ppo_training as t
        class Env:
            action_space = type('Space', (), {'n': 3})()
            def __init__(self, *args):
                pass
            def close(self):
                pass
        class SyntheticRunner:
            def __init__(self, **kwargs):
                pass
            def run(self):
                return (torch.randn(32, 4), torch.randn(32), torch.arange(32) % 3,
                        torch.randn(32), torch.randn(32, 3), [{'r': 1.}])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'new'
            with patch.dict(t.PROFILES, {'test': (4, 64)}), \
                    patch.object(t, 'ProcgenAdapter', Env), \
                    patch.object(t, 'Runner', SyntheticRunner), \
                    patch.object(t, 'SharedActorCritic', lambda *a, **k: TinyShared()), \
                    patch('sys.argv', ['ppo', '--device', 'cpu', '--out', str(out)]):
                t.main('test')
            self.assertEqual((out / 'status').read_text().strip(), 'PASS')
            import csv
            import gzip
            with (out / 'progress.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([int(r['transition']) for r in rows], [32, 64])
            self.assertTrue(all(int(r['updates']) == 32 and float(r['lr']) == LR for r in rows))
            with gzip.open(out / 'metric_trace.jsonl.gz', 'rt') as f:
                events = [json.loads(line) for line in f]
            self.assertEqual(len(events), 64)
            self.assertTrue((out / 'checkpoint.pt').stat().st_size > 0)
