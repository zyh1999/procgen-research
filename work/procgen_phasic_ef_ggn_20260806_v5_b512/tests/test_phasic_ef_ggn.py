import math
import unittest
from collections import OrderedDict

import torch
from torch import nn

from phasic_ef_ggn import (
    actor_fisher_vector_product,
    apply_direction,
    cholesky_solve_with_retry,
    compute_critic_ggn_direction,
    compute_full_gradient_anchor_direction,
    compute_per_sample_value_jacobian,
    embed_full_direction,
    fisher_clip_scale,
    flatten_named_tensors,
    partition_named_parameters,
    policy_phase_critic_mse,
    run_auxiliary_critic_ggn_step,
)
from ppg_auxiliary import (
    decode_procgen_observations,
    encode_procgen_observations,
)


torch.set_default_dtype(torch.float64)


class TinySharedActorCritic(nn.Module):
    def __init__(self, obs_dim=3, hidden=4, actions=3):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.actor_head = nn.Linear(hidden, actions)
        self.critic_head = nn.Linear(hidden, 1)
        self.aux_vf_head = nn.Linear(hidden, 1)

    def forward(self, observations, value_head="true"):
        features = self.shared(observations)
        head = self.critic_head if value_head == "true" else self.aux_vf_head
        return head(features).squeeze(-1), self.actor_head(features)


class TinySeparateActorCritic(nn.Module):
    def __init__(self, obs_dim=3, actions=3):
        super().__init__()
        self.actor_head = nn.Linear(obs_dim, actions)
        self.critic_head = nn.Linear(obs_dim, 1)

    def forward(self, observations, value_head="true"):
        return (
            self.critic_head(observations).squeeze(-1),
            self.actor_head(observations),
        )


class PhasicEFGGNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_01_linear_critic_sample_parameter_equivalence(self):
        batch_size, parameter_size = 7, 5
        jacobian = torch.randn(batch_size, parameter_size)
        residual = torch.randn(batch_size)
        damping = 0.03
        beta = torch.linalg.solve(
            jacobian @ jacobian.T
            + batch_size * damping * torch.eye(batch_size),
            residual,
        )
        sample_direction = jacobian.T @ beta
        parameter_direction = torch.linalg.solve(
            jacobian.T @ jacobian
            + batch_size * damping * torch.eye(parameter_size),
            jacobian.T @ residual,
        )
        relative_error = torch.linalg.vector_norm(
            sample_direction - parameter_direction
        ) / torch.linalg.vector_norm(parameter_direction)
        self.assertLess(relative_error.item(), 1e-5)

    def test_02_descent_sign_reduces_linear_mse(self):
        batch_size, parameter_size = 11, 4
        x = torch.randn(batch_size, parameter_size)
        weights = torch.randn(parameter_size)
        targets = torch.randn(batch_size)
        residual = x @ weights - targets
        damping = 0.05
        direction = x.T @ torch.linalg.solve(
            x @ x.T + batch_size * damping * torch.eye(batch_size), residual
        )
        mse_before = 0.5 * residual.square().mean()
        mse_after = 0.5 * (x @ (weights - 0.1 * direction) - targets).square().mean()
        self.assertLess(mse_after.item(), mse_before.item())

    def test_03_fisher_clip_obeys_radius(self):
        dimension = 13
        factor = torch.randn(9, dimension)
        fisher = factor.T @ factor / factor.shape[0]
        direction = torch.randn(dimension)
        q = direction @ fisher @ direction
        radius = 0.07
        scale = fisher_clip_scale(
            q, learning_rate=1.0, fisher_radius=radius, enabled=True
        )
        clipped_q = (scale * direction) @ fisher @ (scale * direction)
        self.assertLessEqual(clipped_q.item(), radius**2 + 1e-10)

    def test_04_no_shared_critic_direction_has_zero_actor_fisher(self):
        model = TinySeparateActorCritic()
        observations = torch.randn(8, 3)
        groups = partition_named_parameters(model)
        aux_names = groups["critic_head"]
        direction, _ = compute_critic_ggn_direction(
            model,
            observations,
            torch.randn(8),
            aux_names,
            damping=0.02,
            linear_solve_dtype=torch.float64,
        )
        full = embed_full_direction(model, direction)
        _, quadratic = actor_fisher_vector_product(
            model, observations, full, groups["actor_head"]
        )
        scale = fisher_clip_scale(
            quadratic, 0.1, target_kl=0.01, enabled=True
        )
        self.assertLess(abs(quadratic.item()), 1e-12)
        self.assertGreater(scale.item(), 1.0 - 1e-12)

    def test_05_shared_critic_direction_has_nonzero_actor_fisher(self):
        model = TinySharedActorCritic()
        observations = torch.randn(9, 3)
        groups = partition_named_parameters(model)
        aux_names = groups["shared"] + groups["critic_head"]
        direction, _ = compute_critic_ggn_direction(
            model,
            observations,
            torch.randn(9),
            aux_names,
            damping=0.02,
            linear_solve_dtype=torch.float64,
        )
        full = embed_full_direction(model, direction)
        _, quadratic = actor_fisher_vector_product(
            model,
            observations,
            full,
            groups["shared"] + groups["actor_head"],
        )
        self.assertGreater(quadratic.item(), 1e-10)
        for name in groups["actor_head"]:
            self.assertEqual(torch.count_nonzero(full[name]).item(), 0)

    def test_06_rank_deficient_cholesky_fallback(self):
        gram = torch.zeros(6, 6)
        rhs = torch.randn(6)
        solution, diagnostics = cholesky_solve_with_retry(
            gram,
            rhs,
            damping=0.0,
            max_retries=4,
            damping_multiplier=10.0,
        )
        self.assertTrue(torch.isfinite(solution).all())
        self.assertEqual(diagnostics["cholesky_success"], 1.0)
        self.assertGreaterEqual(diagnostics["cholesky_retries"], 1.0)
        self.assertGreater(diagnostics["effective_damping"], 0.0)

    def test_07_finite_difference_value_response(self):
        model = TinySharedActorCritic()
        observations = torch.randn(6, 3)
        groups = partition_named_parameters(model)
        aux_names = groups["shared"] + groups["critic_head"]
        jacobian = compute_per_sample_value_jacobian(
            model, observations, aux_names, chunk_size=2
        )
        direction, _ = compute_critic_ggn_direction(
            model,
            observations,
            torch.randn(6),
            aux_names,
            damping=0.03,
            jacobian_chunk_size=2,
            linear_solve_dtype=torch.float64,
        )
        flat_direction = flatten_named_tensors(direction, aux_names)
        predicted_response = jacobian @ flat_direction
        with torch.no_grad():
            values_before = model(observations)[0].clone()

        def response_error(step_size):
            apply_direction(model, direction, step_size, allowed_names=aux_names)
            with torch.no_grad():
                values_after = model(observations)[0].clone()
            reverse = OrderedDict((name, -value) for name, value in direction.items())
            apply_direction(model, reverse, step_size, allowed_names=aux_names)
            finite_response = (values_before - values_after) / step_size
            return torch.linalg.vector_norm(
                finite_response - predicted_response
            ).item()

        error_large = response_error(1e-2)
        error_small = response_error(1e-4)
        self.assertLess(error_small, 0.05 * error_large)

    def test_08_policy_phase_detach_isolates_huge_critic_gradient(self):
        model = TinySharedActorCritic()
        observations = torch.randn(16, 3)
        targets = torch.full((16,), 1.0e6)

        loss, _ = policy_phase_critic_mse(
            model.shared,
            lambda features: model.critic_head(features).squeeze(-1),
            observations,
            targets,
            detach_shared_features=True,
        )
        loss.backward()

        shared_grads = [parameter.grad for parameter in model.shared.parameters()]
        actor_grads = [parameter.grad for parameter in model.actor_head.parameters()]
        critic_grads = [parameter.grad for parameter in model.critic_head.parameters()]
        self.assertTrue(all(gradient is None for gradient in shared_grads))
        self.assertTrue(all(gradient is None for gradient in actor_grads))
        self.assertTrue(all(gradient is not None for gradient in critic_grads))
        critic_norm = torch.linalg.vector_norm(torch.cat([
            gradient.reshape(-1) for gradient in critic_grads
        ]))
        self.assertGreater(critic_norm.item(), 1.0e5)

    def test_09_nonphasic_critic_mse_reaches_shared_but_not_actor_head(self):
        model = TinySharedActorCritic()
        observations = torch.randn(16, 3)
        targets = torch.randn(16)
        loss, _ = policy_phase_critic_mse(
            model.shared,
            lambda features: model.critic_head(features).squeeze(-1),
            observations,
            targets,
            detach_shared_features=False,
        )
        loss.backward()
        self.assertTrue(any(
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in model.shared.parameters()
        ))
        self.assertTrue(all(
            parameter.grad is None for parameter in model.actor_head.parameters()
        ))

    def test_10_full_gradient_anchor_matches_parameter_space_solve(self):
        model = TinySharedActorCritic(obs_dim=3, hidden=2, actions=2)
        observations = torch.randn(7, 3)
        groups = partition_named_parameters(model)
        aux_names = groups["shared"] + groups["aux_critic_head"]
        jacobian = compute_per_sample_value_jacobian(
            model,
            observations,
            aux_names,
            value_head="aux",
        )
        full_gradient = torch.randn(jacobian.shape[1])
        damping = 0.07
        direction, _ = compute_full_gradient_anchor_direction(
            model,
            observations,
            full_gradient,
            aux_names,
            damping,
            value_head="aux",
            linear_solve_dtype=torch.float64,
        )
        sample_direction = flatten_named_tensors(direction, aux_names)
        direct_direction = torch.linalg.solve(
            jacobian.T @ jacobian / observations.shape[0]
            + damping * torch.eye(jacobian.shape[1]),
            full_gradient,
        )
        relative_error = torch.linalg.vector_norm(
            sample_direction - direct_direction
        ) / torch.linalg.vector_norm(direct_direction)
        self.assertLess(relative_error.item(), 1e-8)

    def test_11_procgen_observation_pack_roundtrip(self):
        pixels = torch.randint(0, 256, (9, 3, 8, 8), dtype=torch.int64)
        observations = pixels.to(torch.float64) / 127.5 - 1.0
        packed = encode_procgen_observations(observations)
        restored = decode_procgen_observations(packed, "cpu").to(torch.float64)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertLess(
            float((restored - observations).abs().max().item()), 1e-6
        )

    def test_12_same_batch_step_updates_true_head_only_and_descends(self):
        model = TinySharedActorCritic(obs_dim=3, hidden=5, actions=3)
        observations = torch.randn(12, 3)
        fisher_observations = torch.randn(10, 3)
        targets = torch.randn(12)
        actor_before = {
            name: parameter.detach().clone()
            for name, parameter in model.actor_head.named_parameters()
        }
        aux_before = {
            name: parameter.detach().clone()
            for name, parameter in model.aux_vf_head.named_parameters()
        }
        critic_before = {
            name: parameter.detach().clone()
            for name, parameter in model.critic_head.named_parameters()
        }
        diagnostics = run_auxiliary_critic_ggn_step(
            model,
            observations,
            targets,
            fisher_observations,
            damping=0.03,
            learning_rate=0.05,
            target_kl=0.01,
            fisher_radius=None,
            use_actor_fisher_clip=False,
            jacobian_chunk_size=4,
            cholesky_max_retries=5,
            cholesky_damping_multiplier=10.0,
            linear_solve_dtype=torch.float64,
        )
        self.assertLess(
            diagnostics["critic_mse_after"],
            diagnostics["critic_mse_before"],
        )
        for name, parameter in model.actor_head.named_parameters():
            self.assertTrue(torch.equal(parameter, actor_before[name]))
        for name, parameter in model.aux_vf_head.named_parameters():
            self.assertTrue(torch.equal(parameter, aux_before[name]))
        self.assertTrue(any(
            not torch.equal(parameter, critic_before[name])
            for name, parameter in model.critic_head.named_parameters()
        ))
        self.assertEqual(
            diagnostics["actor_head_direction_max_abs"], 0.0
        )
        self.assertEqual(
            diagnostics["aux_critic_head_direction_max_abs"], 0.0
        )
        self.assertGreater(
            diagnostics["true_critic_head_direction_l2_norm"], 0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
