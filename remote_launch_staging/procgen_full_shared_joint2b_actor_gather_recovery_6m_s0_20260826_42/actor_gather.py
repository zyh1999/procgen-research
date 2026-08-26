#!/usr/bin/env python3
"""The sole Task42 preflight-only actor log-prob selection change."""
import torch


def selected_logp_gather(logits, action):
    if action.dtype != torch.long:
        raise TypeError("actor action must have torch.long dtype")
    if action.ndim > logits.ndim - 1:
        raise ValueError("actor action rank exceeds logits batch rank")
    logp_all = torch.log_softmax(logits, dim=-1)
    selected_logp = torch.gather(
        logp_all,
        dim=-1,
        index=action.to(torch.long).reshape(*logp_all.shape[:-1], 1),
    ).squeeze(-1)
    return selected_logp
