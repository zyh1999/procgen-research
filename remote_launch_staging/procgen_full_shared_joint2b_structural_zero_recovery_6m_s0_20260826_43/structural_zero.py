#!/usr/bin/env python3
"""Role-constrained materialization of structural autograd None gradients."""
import torch


def materialize_structural_zeros(grads, named_parameters, roles, allowed_none_roles):
    if len(grads) != len(named_parameters) or len(roles) != len(named_parameters):
        raise AssertionError("structural-zero collection length mismatch")
    output = []
    statistics = {"none_by_role": {}, "materialized_zero_tensors": 0,
                  "materialized_zero_numel": 0}
    for index, (grad, (name, parameter), role) in enumerate(
            zip(grads, named_parameters, roles)):
        if grad is None:
            if role not in allowed_none_roles:
                raise AssertionError(f"disallowed structural None: {index}:{name}:{role}")
            grad = torch.zeros_like(parameter)
            statistics["none_by_role"][role] = statistics["none_by_role"].get(role, 0) + 1
            statistics["materialized_zero_tensors"] += 1
            statistics["materialized_zero_numel"] += parameter.numel()
        if grad.shape != parameter.shape:
            raise AssertionError(f"structural-zero shape drift: {name}")
        if grad.dtype != parameter.dtype:
            raise AssertionError(f"structural-zero dtype drift: {name}")
        if grad.device != parameter.device:
            raise AssertionError(f"structural-zero device drift: {name}")
        if role in allowed_none_roles and torch.count_nonzero(grad).item() != 0:
            raise AssertionError(f"structurally disconnected role is nonzero: {name}:{role}")
        output.append(grad)
    return output, statistics
