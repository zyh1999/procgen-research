#!/usr/bin/env python3
"""Actual-Torch Task26 runtime-spy positive and required negatives."""
import sys
import tempfile
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ast_runtime_call_audit_task26 import RuntimeSemanticSpy


def helper(det, paper):
    det_norm = torch.linalg.vector_norm(det)
    paper_norm = torch.linalg.vector_norm(paper)
    scale = paper_norm / det_norm
    target = det * scale
    cosine = torch.dot(det, paper) / (det_norm * paper_norm)
    return target, det_norm, paper_norm, scale, torch.linalg.vector_norm(target), cosine


def fixture(function=helper):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=1e-6)
    first = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    second = torch.tensor([4.0, 3.0, 2.0, 1.0], device=device)
    namespace = {"head_direction": first, "paper_head_proposal": second}
    module = types.SimpleNamespace(torch=torch, match_head_proposal_norm=function)
    temporary = tempfile.TemporaryDirectory()
    spy = RuntimeSemanticSpy(module, namespace, model, optimizer, Path(temporary.name) / "ledger.json")
    return temporary, module, namespace, model, optimizer, spy


def complete(namespace, result):
    namespace["matched"] = result
    namespace["head_target_proposal"] = result[0]
    namespace["replacement"] = {"head": -result[0]}
    namespace["target_preclip"] = {"head": -result[0]}


def expect_reject(label, action):
    try:
        action()
    except RuntimeError:
        return
    raise RuntimeError("runtime negative was not rejected: " + label)


temporary, module, namespace, model, optimizer, spy = fixture()
try:
    result = module.match_head_proposal_norm(namespace["head_direction"], namespace["paper_head_proposal"])
    complete(namespace, result)
    ledger = spy.finalize()
    if ledger["wrapped_observed_call_count"] != 1:
        raise RuntimeError("positive spy count mismatch")
finally:
    spy.restore()
    temporary.cleanup()


temporary, module, namespace, model, optimizer, spy = fixture()
try:
    expect_reject("no-call", spy.finalize)
finally:
    spy.restore(); temporary.cleanup()


temporary, module, namespace, model, optimizer, spy = fixture()
try:
    module.match_head_proposal_norm(namespace["head_direction"], namespace["paper_head_proposal"])
    expect_reject("duplicate-call", lambda: module.match_head_proposal_norm(namespace["head_direction"], namespace["paper_head_proposal"]))
finally:
    spy.restore(); temporary.cleanup()


temporary, module, namespace, model, optimizer, spy = fixture()
try:
    expect_reject("wrong-identity", lambda: module.match_head_proposal_norm(namespace["head_direction"].clone(), namespace["paper_head_proposal"]))
finally:
    spy.restore(); temporary.cleanup()


def rng_mutation(det, paper):
    torch.rand(1, device=det.device)
    return helper(det, paper)


temporary, module, namespace, model, optimizer, spy = fixture(rng_mutation)
try:
    expect_reject("rng-mutation", lambda: module.match_head_proposal_norm(namespace["head_direction"], namespace["paper_head_proposal"]))
finally:
    spy.restore(); temporary.cleanup()


temporary, module, namespace, model, optimizer, spy = fixture()
original = spy.original
def parameter_mutation(det, paper):
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    return original(det, paper)
spy.original = parameter_mutation
try:
    expect_reject("parameter-mutation", lambda: module.match_head_proposal_norm(namespace["head_direction"], namespace["paper_head_proposal"]))
finally:
    spy.restore(); temporary.cleanup()


def wrong_return(det, paper):
    result = list(helper(det, paper))
    result[4] = result[4] + 1.0
    return tuple(result)


temporary, module, namespace, model, optimizer, spy = fixture(wrong_return)
try:
    expect_reject("return-mutation", lambda: module.match_head_proposal_norm(namespace["head_direction"], namespace["paper_head_proposal"]))
finally:
    spy.restore(); temporary.cleanup()

print(sys.version)
print("torch_version=" + torch.__version__)
print("TASK26_RUNTIME_SPY_POSITIVE_NEGATIVE_TESTS_PASS")
