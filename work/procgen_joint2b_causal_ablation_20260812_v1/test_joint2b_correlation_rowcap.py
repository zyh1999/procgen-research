import importlib.util
from pathlib import Path
import sys

import torch


MODULE_PATH = Path(__file__).with_name(
    "train_shared_joint2b_block_damping.py"
)
if not MODULE_PATH.is_file():
    remote_code = Path(__file__).parent / "code"
    MODULE_PATH = (
        remote_code
        / "train_shared_joint2b_rownorm_correlation_dualanchor_entropy_pi.py"
    )
    if not MODULE_PATH.is_file():
        MODULE_PATH = (
            remote_code
        / "train_shared_joint2b_rownorm_correlation_dualanchor_entropyrhs.py"
        )
    if not MODULE_PATH.is_file():
        MODULE_PATH = (
            remote_code
            / "train_shared_joint2b_rownorm_correlation_dualanchor.py"
        )
        if not MODULE_PATH.is_file():
            MODULE_PATH = (
                remote_code
                / "train_shared_joint2b_rownorm_correlation_fisheranchor05.py"
            )
            if not MODULE_PATH.is_file():
                MODULE_PATH = (
                    remote_code
                    / "train_shared_joint2b_rownorm_correlation_rowcap001.py"
                )
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("joint2b_rowcap", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def solve_with_scale(kernel, rhs, ratio, scale, damping):
    rows = kernel.shape[0]
    normalized = scale[:, None] * kernel * scale[None, :]
    system = normalized * ratio.unsqueeze(0)
    system = system + damping * torch.eye(rows, dtype=kernel.dtype)
    alpha_normalized = torch.linalg.solve(system, scale * rhs)
    return scale * alpha_normalized


def main():
    dtype = torch.float64
    eps = 1e-12
    actor_rows = 4
    raw_diag = torch.tensor(
        [1e-20, 1.0, 4.0, 9.0, 1e-18, 100.0, 121.0, 144.0],
        dtype=dtype,
    )

    uncapped = MODULE.block_capped_correlation_row_scale(
        raw_diag, actor_rows, 0.0, eps
    )
    torch.testing.assert_close(
        uncapped[0], raw_diag.clamp_min(eps).rsqrt(), rtol=0.0, atol=0.0
    )
    assert uncapped[3].item() == 0.0
    assert uncapped[4].item() == 0.0

    scale, effective, floor, actor_fraction, critic_fraction = (
        MODULE.block_capped_correlation_row_scale(
            raw_diag, actor_rows, 0.01, eps
        )
    )
    # torch.median uses the lower middle element for an even-sized block.
    expected_actor_floor = 0.01
    expected_critic_floor = 1.0
    torch.testing.assert_close(
        floor[:actor_rows],
        torch.full((actor_rows,), expected_actor_floor, dtype=dtype),
    )
    torch.testing.assert_close(
        floor[actor_rows:],
        torch.full((4,), expected_critic_floor, dtype=dtype),
    )
    torch.testing.assert_close(effective, torch.maximum(raw_diag, floor))
    torch.testing.assert_close(scale, effective.rsqrt())
    assert actor_fraction.item() == 0.25
    assert critic_fraction.item() == 0.25

    # The capped normalized solve is still exactly representable in raw
    # coordinates with lambda * effective_diagonal damping.
    torch.manual_seed(11)
    h = torch.randn(8, 13, dtype=dtype)
    h[0].mul_(1e-10)
    h[4].mul_(1e-9)
    kernel = h @ h.t() / 8.0
    diag = torch.diagonal(kernel)
    scale, effective, *_ = MODULE.block_capped_correlation_row_scale(
        diag, actor_rows, 0.01, eps
    )
    rhs = torch.randn(8, dtype=dtype)
    ratio = torch.exp(torch.randn(8, dtype=dtype) * 0.2)
    damping = 0.3
    normalized_alpha = solve_with_scale(
        kernel, rhs, ratio, scale, damping
    )
    raw_system = kernel * ratio.unsqueeze(0)
    raw_system = raw_system + torch.diag(damping * effective)
    raw_alpha = torch.linalg.solve(raw_system, rhs)
    torch.testing.assert_close(
        normalized_alpha, raw_alpha, rtol=2e-10, atol=2e-10
    )

    # A categorical-Fisher floor must be an exact no-op above the floor and
    # must cancel global actor-row amplification below it.
    one = MODULE.categorical_fisher_actor_anchor_scale(
        torch.tensor(0.8, dtype=dtype), 0.5
    )
    half = MODULE.categorical_fisher_actor_anchor_scale(
        torch.tensor(0.125, dtype=dtype), 0.5
    )
    disabled = MODULE.categorical_fisher_actor_anchor_scale(
        torch.tensor(0.0, dtype=dtype), 0.0
    )
    torch.testing.assert_close(one, torch.tensor(1.0, dtype=dtype))
    torch.testing.assert_close(half, torch.tensor(0.5, dtype=dtype))
    torch.testing.assert_close(disabled, torch.tensor(1.0, dtype=dtype))

    anchored_scale = scale.clone()
    anchored_scale[:actor_rows].mul_(half)
    anchored_effective = effective.clone()
    anchored_effective[:actor_rows].div_(half.square())
    anchored_alpha = solve_with_scale(
        kernel, rhs, ratio, anchored_scale, damping
    )
    anchored_raw_system = kernel * ratio.unsqueeze(0)
    anchored_raw_system = anchored_raw_system + torch.diag(
        damping * anchored_effective
    )
    anchored_raw_alpha = torch.linalg.solve(anchored_raw_system, rhs)
    torch.testing.assert_close(
        anchored_alpha, anchored_raw_alpha, rtol=2e-10, atol=2e-10
    )

    # A .001 running-highwater anchor with highwater=100 gives a raw actor
    # denominator floor of .1, while leaving the critic block untouched.
    (
        highwater_scale,
        highwater_effective,
        highwater_floor,
        highwater_fraction,
    ) = MODULE.actor_kernel_highwater_row_scale(
        diag,
        effective,
        actor_rows,
        actor_kernel_highwater=100.0,
        anchor_fraction=0.001,
        eps=eps,
    )
    torch.testing.assert_close(
        highwater_floor, torch.tensor(0.1, dtype=dtype)
    )
    torch.testing.assert_close(
        highwater_effective[:actor_rows],
        torch.maximum(
            effective[:actor_rows],
            torch.full((actor_rows,), 0.1, dtype=dtype),
        ),
    )
    torch.testing.assert_close(
        highwater_effective[actor_rows:], effective[actor_rows:]
    )
    torch.testing.assert_close(highwater_scale, highwater_effective.rsqrt())
    expected_highwater_fraction = (diag[:actor_rows] < 0.1).to(dtype).mean()
    torch.testing.assert_close(
        highwater_fraction, expected_highwater_fraction
    )
    highwater_alpha = solve_with_scale(
        kernel, rhs, ratio, highwater_scale, damping
    )
    highwater_raw_system = kernel * ratio.unsqueeze(0)
    highwater_raw_system = highwater_raw_system + torch.diag(
        damping * highwater_effective
    )
    highwater_raw_alpha = torch.linalg.solve(highwater_raw_system, rhs)
    torch.testing.assert_close(
        highwater_alpha, highwater_raw_alpha, rtol=2e-10, atol=2e-10
    )

    for bad_floor in (-0.01, 1.01):
        try:
            MODULE.categorical_fisher_actor_anchor_scale(
                torch.tensor(0.5, dtype=dtype), bad_floor
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid categorical Fisher floor accepted: {bad_floor}"
            )

    for bad_highwater, bad_anchor_fraction, bad_anchor_eps in (
        (-1.0, 0.001, eps),
        (1.0, -0.001, eps),
        (1.0, 1.001, eps),
        (1.0, 0.001, 0.0),
    ):
        try:
            MODULE.actor_kernel_highwater_row_scale(
                diag,
                effective,
                actor_rows,
                bad_highwater,
                bad_anchor_fraction,
                bad_anchor_eps,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "invalid actor-kernel anchor arguments accepted: "
                f"{bad_highwater=}, {bad_anchor_fraction=}, "
                f"{bad_anchor_eps=}"
            )

    # The sampled categorical score target used by the trainer is exactly an
    # entropy ascent gradient in expectation; centering only adds a baseline.
    logits = torch.tensor([1.5, 0.2, -0.7], dtype=dtype, requires_grad=True)
    probabilities = torch.softmax(logits, dim=0)
    log_probabilities = torch.log_softmax(logits, dim=0)
    entropy = -(probabilities * log_probabilities).sum()
    exact_entropy_gradient = torch.autograd.grad(entropy, logits)[0]
    score_rows = torch.eye(3, dtype=dtype) - probabilities.detach()[None, :]
    entropy_targets = -log_probabilities.detach() - 1.0
    entropy_targets = entropy_targets - (
        probabilities.detach() * entropy_targets
    ).sum()
    score_estimate = (
        probabilities.detach()[:, None]
        * score_rows
        * entropy_targets[:, None]
    ).sum(dim=0)
    torch.testing.assert_close(
        score_estimate, exact_entropy_gradient, rtol=2e-12, atol=2e-12
    )

    deficit, coefficient = MODULE.entropy_rhs_controller(
        torch.tensor(2.4, dtype=dtype), 2.5, 0.1, 0.25
    )
    torch.testing.assert_close(deficit, torch.tensor(0.1, dtype=dtype))
    torch.testing.assert_close(coefficient, torch.tensor(0.01, dtype=dtype))
    deficit, coefficient = MODULE.entropy_rhs_controller(
        torch.tensor(0.0, dtype=dtype), 2.5, 0.1, 0.25
    )
    torch.testing.assert_close(deficit, torch.tensor(2.5, dtype=dtype))
    torch.testing.assert_close(coefficient, torch.tensor(0.25, dtype=dtype))
    deficit, coefficient = MODULE.entropy_rhs_controller(
        torch.tensor(2.6, dtype=dtype), 2.5, 0.1, 0.25
    )
    torch.testing.assert_close(deficit, torch.tensor(0.0, dtype=dtype))
    torch.testing.assert_close(coefficient, torch.tensor(0.0, dtype=dtype))
    for bad_controller in ((-1.0, 0.1, 0.25), (2.5, -0.1, 0.25), (2.5, 0.1, 0.0)):
        try:
            MODULE.entropy_rhs_controller(
                torch.tensor(1.0, dtype=dtype), *bad_controller
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid entropy RHS controller accepted: {bad_controller}"
            )

    # The integral state advances once per rollout, then remains frozen
    # across all optimizer minibatches in that rollout.
    deficit, integral, coefficient = MODULE.entropy_rhs_pi_controller(
        torch.tensor(2.0, dtype=dtype), 2.5, 0.1, 0.02, 0.5, 0.2
    )
    torch.testing.assert_close(deficit, torch.tensor(0.5, dtype=dtype))
    torch.testing.assert_close(integral, torch.tensor(0.21, dtype=dtype))
    torch.testing.assert_close(coefficient, torch.tensor(0.26, dtype=dtype))
    _, integral, coefficient = MODULE.entropy_rhs_pi_controller(
        torch.tensor(2.8, dtype=dtype), 2.5, 0.1, 0.02, 0.5, 0.2
    )
    torch.testing.assert_close(integral, torch.tensor(0.194, dtype=dtype))
    torch.testing.assert_close(coefficient, torch.tensor(0.194, dtype=dtype))
    _, frozen_integral, coefficient = MODULE.entropy_rhs_pi_controller(
        torch.tensor(2.0, dtype=dtype), 2.5, 0.1, 0.0, 0.5, 0.21
    )
    torch.testing.assert_close(
        frozen_integral, torch.tensor(0.21, dtype=dtype)
    )
    torch.testing.assert_close(coefficient, torch.tensor(0.26, dtype=dtype))
    _, saturated_integral, saturated_coefficient = (
        MODULE.entropy_rhs_pi_controller(
            torch.tensor(0.0, dtype=dtype), 2.5, 0.25, 0.2, 0.5, 0.49
        )
    )
    torch.testing.assert_close(
        saturated_integral, torch.tensor(0.5, dtype=dtype)
    )
    torch.testing.assert_close(
        saturated_coefficient, torch.tensor(0.5, dtype=dtype)
    )
    for bad_pi in (
        (-1.0, 0.1, 0.02, 0.5, 0.0),
        (2.5, -0.1, 0.02, 0.5, 0.0),
        (2.5, 0.1, -0.02, 0.5, 0.0),
        (2.5, 0.1, 0.02, 0.0, 0.0),
        (2.5, 0.1, 0.02, 0.5, -0.1),
    ):
        try:
            MODULE.entropy_rhs_pi_controller(
                torch.tensor(1.0, dtype=dtype), *bad_pi
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"invalid entropy RHS PI controller accepted: {bad_pi}"
            )

    for bad_rows, bad_fraction, bad_eps in (
        (0, 0.01, eps),
        (9, 0.01, eps),
        (4, -0.01, eps),
        (4, 1.01, eps),
        (4, 0.01, 0.0),
    ):
        try:
            MODULE.block_capped_correlation_row_scale(
                raw_diag, bad_rows, bad_fraction, bad_eps
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "invalid row-cap arguments were accepted: "
                f"{bad_rows=}, {bad_fraction=}, {bad_eps=}"
            )

    print(
        "PASS: dual anchors and categorical entropy-RHS PI controller"
    )


if __name__ == "__main__":
    main()
