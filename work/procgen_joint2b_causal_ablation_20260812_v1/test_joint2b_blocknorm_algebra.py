#!/usr/bin/env python3
"""Algebra checks for the objective-preserving strict-2B normalization."""

import numpy as np


def main() -> None:
    rng = np.random.default_rng(7)
    batch, params = 9, 13
    actor_h = rng.normal(size=(batch, params)) * 8.0
    critic_h = rng.normal(size=(batch, params)) * 57.0
    actor_rhs = rng.normal(size=batch)
    critic_rhs = rng.normal(size=batch)
    ratio = np.exp(rng.normal(scale=0.15, size=batch))
    joint_ratio = np.concatenate([ratio, np.ones(batch)])

    actor_med = np.median(np.sum(actor_h**2, axis=1) / batch)
    critic_med = np.median(np.sum(critic_h**2, axis=1) / batch)
    actor_scale = 1.0 / np.sqrt(actor_med)
    critic_scale = 1.0 / np.sqrt(critic_med)

    raw_h = np.concatenate([actor_h, critic_h], axis=0)
    raw_rhs = np.concatenate([actor_rhs, critic_rhs])
    scaled_h = np.concatenate(
        [actor_scale * actor_h, critic_scale * critic_h], axis=0
    )
    scaled_rhs = np.concatenate(
        [actor_rhs / actor_scale, critic_rhs / critic_scale]
    )

    raw_gradient = raw_h.T @ (joint_ratio * raw_rhs) / batch
    scaled_gradient = scaled_h.T @ (joint_ratio * scaled_rhs) / batch
    np.testing.assert_allclose(raw_gradient, scaled_gradient, rtol=1e-12, atol=1e-12)

    damping = 0.03
    gram = scaled_h @ scaled_h.T / batch
    alpha = np.linalg.solve(
        gram * joint_ratio[None, :] + damping * np.eye(2 * batch),
        scaled_rhs,
    )
    dual_direction = scaled_h.T @ (joint_ratio * alpha) / batch

    metric = scaled_h.T @ (joint_ratio[:, None] * scaled_h) / batch
    primal_direction = np.linalg.solve(metric + damping * np.eye(params), scaled_gradient)
    np.testing.assert_allclose(dual_direction, primal_direction, rtol=1e-10, atol=1e-10)

    normalized_actor_med = np.median(
        np.sum((actor_scale * actor_h) ** 2, axis=1) / batch
    )
    normalized_critic_med = np.median(
        np.sum((critic_scale * critic_h) ** 2, axis=1) / batch
    )
    np.testing.assert_allclose(normalized_actor_med, 1.0, rtol=1e-12)
    np.testing.assert_allclose(normalized_critic_med, 1.0, rtol=1e-12)

    # The Schur guard's actor-only reference must use the transformed RHS.
    # With a very large critic-only ridge, the actor-RHS response of the
    # strict joint system must converge to this matched actor-only solve.
    actor_row_diag = np.sum(actor_h**2, axis=1) / batch
    critic_row_diag = np.sum(critic_h**2, axis=1) / batch
    actor_row_scale = 1.0 / np.sqrt(actor_row_diag + 1e-12)
    critic_row_scale = 1.0 / np.sqrt(critic_row_diag + 1e-12)
    row_h_actor = actor_row_scale[:, None] * actor_h
    row_h_critic = critic_row_scale[:, None] * critic_h
    row_actor_rhs = actor_rhs / actor_row_scale
    row_joint_h = np.concatenate([row_h_actor, row_h_critic], axis=0)
    row_joint_ratio = np.concatenate([ratio, np.ones(batch)])
    row_joint_k = row_joint_h @ row_joint_h.T / batch

    actor_k = row_joint_k[:batch, :batch]
    actor_alpha = np.linalg.solve(
        actor_k * ratio[None, :] + damping * np.eye(batch),
        row_actor_rhs,
    )
    actor_direction = row_h_actor.T @ (ratio * actor_alpha) / batch

    joint_alpha_actor_rhs = np.linalg.solve(
        row_joint_k * row_joint_ratio[None, :]
        + np.diag(np.concatenate([
            np.full(batch, damping), np.full(batch, 1e10)
        ])),
        np.concatenate([row_actor_rhs, np.zeros(batch)]),
    )
    joint_direction = (
        row_joint_h.T @ (row_joint_ratio * joint_alpha_actor_rhs) / batch
    )
    np.testing.assert_allclose(
        joint_direction, actor_direction, rtol=1e-8, atol=1e-8
    )

    # The pre-fix reference used the unscaled advantage and therefore does
    # not represent the row-normalized actor-only counterfactual.
    legacy_alpha = np.linalg.solve(
        actor_k * ratio[None, :] + damping * np.eye(batch), actor_rhs
    )
    legacy_direction = row_h_actor.T @ (ratio * legacy_alpha) / batch
    assert not np.allclose(
        legacy_direction, actor_direction, rtol=1e-3, atol=1e-3
    )
    print(
        "PASS: gradient preservation, dual/primal equivalence, block "
        "medians, Schur actor-only RHS identity"
    )


if __name__ == "__main__":
    main()
