"""Small numerical checks only: these tests do not start RL training."""
import unittest
from collections import OrderedDict
import torch
from torch import nn
from torch.func import functional_call, grad, vmap

from shared_procgen.algebra import LayerwiseIO
from shared_procgen.subcurvature import (solve_fullrhs_named_directions_exactfusion,
                                        solve_dual_coefficients_exactfusion,
                                        fixed_size_inclusion_probabilities, pivotal_sample)
from shared_procgen.model import SharedActorCritic
from shared_procgen.update import adaptive_lr, update


class TinyShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Linear(4, 5)
        self.actor = nn.Linear(5, 3)
        self.critic = nn.Linear(5, 1)

    def forward(self, x):
        z = self.trunk(x).tanh()
        return self.critic(z).squeeze(-1), self.actor(z)


class TinyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 3, padding=1)
        self.actor = nn.Linear(3, 3)
        self.critic = nn.Linear(3, 1)

    def forward(self, x):
        z = self.conv(x).tanh().mean((2, 3))
        return self.critic(z).squeeze(-1), self.actor(z)


def score_fixture(batch=12, convolution=False):
    torch.manual_seed(7)
    model = TinyConv() if convolution else TinyShared()
    obs = torch.randn(batch, 2, 4, 4) if convolution else torch.randn(batch, 4)
    noise = torch.randn(batch)
    actions = torch.arange(batch) % 3
    with LayerwiseIO(model) as io:
        values, logits = model(obs)
    scalar = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1) + 2 * noise * values
    factors = io.factors(scalar)
    io.assert_parameter_coverage()
    def score(params, x, action, xi):
        v, logits = functional_call(model, params, (x.unsqueeze(0),))
        return logits.log_softmax(-1).gather(1, action.reshape(1, 1)).sum() + 2 * xi * v.sum()
    jac = vmap(grad(score), in_dims=(None, 0, 0, 0))(dict(model.named_parameters()), obs, actions, noise)
    h = torch.cat([jac[n].reshape(batch, -1) for n, _ in model.named_parameters()], 1).double()
    return model, factors, h


class AlgebraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fullrhs_against_dense_parameter_solve(self):
        model, factors, h = score_fixture()
        anchors, batch, damping = torch.tensor([0, 3, 7, 10]), len(h), 0.5
        rhs = [OrderedDict((n, torch.randn_like(p)) for n, p in model.named_parameters()) for _ in range(2)]
        ratios = [torch.linspace(0.1, 3, batch), torch.ones(batch)]
        directions, info = solve_fullrhs_named_directions_exactfusion(factors, rhs, ratios, anchors, damping, batch)
        for g, ratio, direction, diag in zip(rhs, ratios, directions, info):
            hs = h[anchors]
            system = hs.T @ (ratio[anchors, None].double() * hs) / batch
            system += damping * torch.eye(h.shape[1], dtype=torch.float64)
            expected = torch.linalg.solve(system, torch.cat([v.flatten() for v in g.values()]).double())
            actual = torch.cat([v.flatten() for v in direction.values()]).double()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
            self.assertLess(diag["reduced_residual"], 1e-12)

    def test_dual_against_dense_galerkin_with_zero_tail(self):
        _, factors, h = score_fixture()
        batch, damping, anchors = len(h), 0.5, torch.tensor([0, 3, 7])
        tail = torch.tensor([i for i in range(batch) if i not in anchors])
        for zero_tail in (False, True):
            targets = torch.stack((torch.linspace(-2, 2, batch), torch.ones(batch)), 1)
            if zero_tail:
                targets[tail] = 0
            ratios = torch.stack((torch.linspace(0.1, 3, batch), torch.ones(batch)), 1)
            actual, info = solve_dual_coefficients_exactfusion(factors, targets, ratios, anchors, damping, batch)
            for col in range(2):
                root = ratios[:, col].double().sqrt()
                a = root[:, None] * (h @ h.T / batch) * root[None, :]
                a += damping * torch.eye(batch, dtype=torch.float64)
                target = root * targets[:, col].double()
                basis = torch.eye(batch, dtype=torch.float64)[:, anchors]
                if not zero_tail:
                    tail_vector = torch.zeros(batch, dtype=torch.float64)
                    tail_vector[tail] = target[tail]
                    basis = torch.cat((basis, tail_vector[:, None]), 1)
                y = basis @ torch.linalg.solve(basis.T @ a @ basis, basis.T @ target)
                expected = y / root
                torch.testing.assert_close(actual[:, col].double(), expected, rtol=2e-5, atol=2e-5)
                self.assertLess(info[col]["combined_reduced_residual"], 1e-12)

    def test_controller_thresholds(self):
        self.assertEqual(adaptive_lr(0.3, 0.04), 0.3)
        self.assertEqual(adaptive_lr(0.3, 0.005), 0.3)
        self.assertAlmostEqual(adaptive_lr(0.3, 0.041), 0.2)
        self.assertEqual(adaptive_lr(0.5, 0.001), 0.5)
        self.assertEqual(adaptive_lr(1e-4, 0.1), 1e-4)
        with self.assertRaises(FloatingPointError):
            adaptive_lr(0.3, float("nan"))

    def test_conv_tail_contractions_against_dense(self):
        _, factors, h = score_fixture(convolution=True)
        batch, anchors = len(h), torch.tensor([0, 3, 7])
        targets = torch.stack((torch.linspace(-2, 2, batch), torch.ones(batch)), 1)
        ratios = torch.stack((torch.linspace(0.1, 3, batch), torch.ones(batch)), 1)
        actual, _ = solve_dual_coefficients_exactfusion(factors, targets, ratios, anchors, 0.5, batch)
        for col in range(2):
            root = ratios[:, col].double().sqrt()
            a = root[:, None] * (h @ h.T / batch) * root[None, :]
            a += 0.5 * torch.eye(batch, dtype=torch.float64)
            b = root * targets[:, col].double()
            tail = b.clone()
            tail[anchors] = 0
            basis = torch.cat((torch.eye(batch, dtype=torch.float64)[:, anchors], tail[:, None]), 1)
            expected = basis @ torch.linalg.solve(basis.T @ a @ basis, basis.T @ b) / root
            torch.testing.assert_close(actual[:, col].double(), expected, rtol=2e-5, atol=2e-5)

    def test_pivotal_fixed_size_and_inclusion_mass(self):
        weights = torch.tensor([0.90, 0.04, 0.02, 0.02, 0.01, 0.01])
        inclusion = fixed_size_inclusion_probabilities(weights, 3)
        self.assertAlmostEqual(float(inclusion.sum()), 3.0)
        self.assertTrue(bool(((inclusion >= 0) & (inclusion <= 1)).all()))
        for seed in range(10):
            torch.manual_seed(seed)
            selected = pivotal_sample(inclusion)
            self.assertEqual(len(selected), 3)
            self.assertEqual(len(selected.unique()), 3)
            self.assertIn(0, selected.tolist())

    def test_normalbatch_update_both_methods(self):
        for method in ("dual127p1", "fullrhs128"):
            torch.manual_seed(19)
            model = TinyShared()
            obs = torch.randn(512, 4)
            old_logits = model(obs)[1].detach()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
            out = update(model, optimizer, obs, torch.arange(512) % 3, torch.randn(512),
                         torch.randn(512), old_logits, method=method)
            self.assertEqual(out["curvature_divisor"], 512)
            self.assertEqual(out["loss_divisor"], 512)
            self.assertFalse(out["critic_ratio"])
            self.assertEqual(out["anchors"], 127 if method == "dual127p1" else 128)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_shared_resnet_shapes_and_popart_invariance(self):
        model = SharedActorCritic(15)
        obs = torch.randn(2, 3, 64, 64)
        values, logits = model(obs)
        before = model.last_v_layer.unnormalize(values).detach()
        model.last_v_layer.update(torch.linspace(-2, 4, 4096))
        after = model.last_v_layer.unnormalize(model(obs)[0]).detach()
        self.assertEqual(logits.shape, (2, 15))
        torch.testing.assert_close(before, after, rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
