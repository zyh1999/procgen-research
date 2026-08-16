#!/usr/bin/env python3
"""Checks for deterministic RAT block-relative actor/critic damping."""

import torch


def damping(actor_median, critic_median):
    actor = max(0.003, 0.10 * actor_median, 0.01 * critic_median)
    critic = max(0.01, 0.10 * critic_median)
    return actor, critic


def main():
    # Once the measured block scales exceed the explicit numerical safety
    # floors, multiplying both kernels by a scalar must multiply both
    # dampings by that scalar and leave dimensionless damping ratios fixed.
    actor_median = 2.0
    critic_median = 9.0
    actor, critic = damping(actor_median, critic_median)
    scale = 17.0
    scaled_actor, scaled_critic = damping(
        scale * actor_median, scale * critic_median
    )
    torch.testing.assert_close(
        torch.tensor(scaled_actor), torch.tensor(scale * actor)
    )
    torch.testing.assert_close(
        torch.tensor(scaled_critic), torch.tensor(scale * critic)
    )

    # At very small scale, the nonzero absolute floors keep both systems
    # invertible; they are intentional safety bounds rather than hidden
    # environment-specific tuning.
    small_actor, small_critic = damping(1e-6, 1e-6)
    assert small_actor == 0.003
    assert small_critic == 0.01

    # A critic block more than ten times larger than the actor activates the
    # shared-trunk actor guard without changing the critic rule.
    guarded_actor, guarded_critic = damping(0.1, 20.0)
    assert guarded_actor == 0.2
    assert guarded_critic == 2.0

    print("actor_damping={:.6g}".format(actor))
    print("critic_damping={:.6g}".format(critic))
    print("TEST=PASS")


if __name__ == "__main__":
    main()
