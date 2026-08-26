#!/usr/bin/env python3
import json
import torch

from actor_gather import selected_logp_gather


torch.manual_seed(420001)
logits = torch.randn(6, 15, dtype=torch.float64, requires_grad=True)
actions = torch.tensor([0, 14, 1, 7, 13, 8], dtype=torch.long)
gathered = selected_logp_gather(logits, actions)
explicit = torch.stack([torch.log_softmax(logits[i], -1)[int(actions[i])]
                        for i in range(actions.numel())])
torch.testing.assert_close(gathered, explicit, rtol=0, atol=0)
gather_grad = torch.autograd.grad(gathered.sum(), logits, retain_graph=True)[0]
explicit_grad = torch.autograd.grad(explicit.sum(), logits)[0]
torch.testing.assert_close(gather_grad, explicit_grad, rtol=0, atol=0)

rejected = {}
def reject(name, fn):
    try:
        value = fn()
        if isinstance(value, torch.Tensor) and value.shape == gathered.shape:
            torch.testing.assert_close(value, explicit, rtol=0, atol=0)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        rejected[name] = True
    else:
        raise AssertionError(f"negative gather case accepted: {name}")

reject("wrong_action_dtype", lambda: selected_logp_gather(logits, actions.float()))
reject("negative_action", lambda: selected_logp_gather(logits, torch.tensor([-1, 14, 1, 7, 13, 8])))
reject("out_of_range_action", lambda: selected_logp_gather(logits, torch.tensor([0, 15, 1, 7, 13, 8])))
reject("wrong_gather_dimension", lambda: torch.gather(torch.log_softmax(logits, -1), 0, actions.reshape(-1, 1)))
reject("wrong_reshape", lambda: torch.gather(torch.log_softmax(logits, -1), -1, actions.reshape(1, -1, 1)))
reject("sign_change", lambda: -selected_logp_gather(logits, actions))
reject("reduction_change", lambda: selected_logp_gather(logits, actions).mean().expand_as(gathered))

detached_forward = selected_logp_gather(logits, actions).detach()
torch.testing.assert_close(detached_forward, explicit, rtol=0, atol=0)
try:
    torch.autograd.grad(detached_forward.sum(), logits)
except RuntimeError:
    rejected["forward_equal_jacobian_changed"] = True
else:
    raise AssertionError("forward-only equality hid a changed Jacobian")

expected = {
    "wrong_action_dtype", "negative_action", "out_of_range_action",
    "wrong_gather_dimension", "wrong_reshape", "sign_change",
    "reduction_change", "forward_equal_jacobian_changed",
}
assert set(rejected) == expected
print("TASK42_GATHER_VALUE_LOGITS_GRAD_NEGATIVE_PASS")
print(json.dumps({"max_value_error": float((gathered-explicit).abs().max()),
                  "max_logits_grad_error": float((gather_grad-explicit_grad).abs().max()),
                  "negative_tests": rejected}, sort_keys=True))
