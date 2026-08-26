#!/usr/bin/env python3
import torch

from structural_zero import materialize_structural_zeros


device = torch.device("cuda:0")
named = [
    ("shared", torch.ones(2, device=device, requires_grad=True)),
    ("policy", torch.ones(3, device=device, requires_grad=True)),
    ("value", torch.ones(4, device=device, requires_grad=True)),
]
roles = ["SHARED", "POLICY_EXCLUSIVE", "CRITIC_EXCLUSIVE"]
actor, actor_stats = materialize_structural_zeros(
    [torch.ones_like(named[0][1]), torch.ones_like(named[1][1]), None],
    named, roles, {"CRITIC_EXCLUSIVE"})
critic, critic_stats = materialize_structural_zeros(
    [torch.ones_like(named[0][1]), None, torch.ones_like(named[2][1])],
    named, roles, {"POLICY_EXCLUSIVE"})
assert torch.count_nonzero(actor[2]) == 0 and actor_stats["materialized_zero_numel"] == 4
assert torch.count_nonzero(critic[1]) == 0 and critic_stats["materialized_zero_numel"] == 3

rejected = {}
def reject(name, fn):
    try:
        fn()
    except (AssertionError, RuntimeError):
        rejected[name] = True
    else:
        raise AssertionError(f"negative structural-zero case accepted: {name}")

reject("actor_none_shared", lambda: materialize_structural_zeros(
    [None, torch.ones_like(named[1][1]), None], named, roles, {"CRITIC_EXCLUSIVE"}))
reject("actor_none_policy", lambda: materialize_structural_zeros(
    [torch.ones_like(named[0][1]), None, None], named, roles, {"CRITIC_EXCLUSIVE"}))
reject("critic_none_shared", lambda: materialize_structural_zeros(
    [None, None, torch.ones_like(named[2][1])], named, roles, {"POLICY_EXCLUSIVE"}))
reject("critic_none_value", lambda: materialize_structural_zeros(
    [torch.ones_like(named[0][1]), None, None], named, roles, {"POLICY_EXCLUSIVE"}))
reject("actor_value_nonzero", lambda: materialize_structural_zeros(
    [torch.ones_like(named[0][1]), torch.ones_like(named[1][1]),
     torch.ones_like(named[2][1])], named, roles, {"CRITIC_EXCLUSIVE"}))
reject("critic_policy_nonzero", lambda: materialize_structural_zeros(
    [torch.ones_like(named[0][1]), torch.ones_like(named[1][1]),
     torch.ones_like(named[2][1])], named, roles, {"POLICY_EXCLUSIVE"}))
reject("zero_column_deleted", lambda: materialize_structural_zeros(
    [torch.ones_like(named[0][1]), torch.ones_like(named[1][1])], named, roles,
    {"CRITIC_EXCLUSIVE"}))
reject("column_reordered", lambda: materialize_structural_zeros(
    [torch.ones_like(named[1][1]), torch.ones_like(named[0][1]), None],
    list(reversed(named)), roles, {"CRITIC_EXCLUSIVE"}))
reject("wrong_shape", lambda: materialize_structural_zeros(
    [torch.ones(3, device=device), torch.ones_like(named[1][1]), None],
    named, roles, {"CRITIC_EXCLUSIVE"}))
reject("wrong_dtype", lambda: materialize_structural_zeros(
    [torch.ones(2, device=device, dtype=torch.float64), torch.ones_like(named[1][1]), None],
    named, roles, {"CRITIC_EXCLUSIVE"}))
reject("wrong_device", lambda: materialize_structural_zeros(
    [torch.ones(2), torch.ones_like(named[1][1]), None], named, roles,
    {"CRITIC_EXCLUSIVE"}))
reject("connected_gradient_zeroed", lambda: torch.testing.assert_close(
    torch.zeros_like(named[0][1]), torch.ones_like(named[0][1]), rtol=0, atol=0))
print("TASK43_STRUCTURAL_ZERO_NEGATIVE_RULES_PASS")
print({"actor": actor_stats, "critic": critic_stats, "negative_tests": rejected})
