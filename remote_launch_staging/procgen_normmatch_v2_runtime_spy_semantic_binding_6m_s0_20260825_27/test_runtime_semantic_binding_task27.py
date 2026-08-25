#!/usr/bin/env python3
"""Actual-Torch Task27 direct-object runtime binding regressions."""
import importlib.util
import os
import tempfile
import types
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("task27_preflight", HERE / "gpuh_preflight_normmatch_v2_task27.py")
task27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task27)


def helper(det, paper):
    det_norm = torch.linalg.vector_norm(det)
    paper_norm = torch.linalg.vector_norm(paper)
    scale = paper_norm / det_norm
    target = det * scale
    cosine = torch.dot(det, paper) / (det_norm * paper_norm)
    return target, det_norm, paper_norm, scale, torch.linalg.vector_norm(target), cosine


def fixture(function=helper):
    device = torch.device(os.environ.get("TASK27_TEST_DEVICE", "cpu"))
    model = torch.nn.Linear(4, 2).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=1e-6)
    det = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    paper = torch.tensor([4.0, 3.0, 2.0, 1.0], device=device)
    module = types.SimpleNamespace(torch=torch, match_head_proposal_norm=function)
    temporary = tempfile.NamedTemporaryFile(prefix="task27-binding-", suffix=".json")
    spy = task27.SemanticBoundRuntimeSpy(
        module, det, paper, model, optimizer, Path(temporary.name),
        {"call": {"normalized_ast_sha256": "test"}}, {"result": "test"},
    )
    return temporary, module, det, paper, model, optimizer, spy


def complete(spy, result):
    namespace = {"matched": result, "target_head_proposal": result[0]}
    return spy.finalize(namespace)


def reject(label, action):
    try:
        action()
    except RuntimeError:
        return
    raise RuntimeError("Task27 runtime negative not rejected: " + label)


temporary, module, det, paper, model, optimizer, spy = fixture()
try:
    result = module.match_head_proposal_norm(det, paper)
    ledger = complete(spy, result)
    if ledger["call_count"] != 1 or not ledger["direct_object_capture_no_name_lookup"]:
        raise RuntimeError("Task27 positive mapping ledger mismatch")
finally:
    spy.restore(); temporary.close()

for label, substitute in (
    ("clone", lambda value: value.clone()),
    ("detach", lambda value: value.detach()),
    ("view", lambda value: value.view_as(value)),
    ("cast", lambda value: value.double()),
    ("recomputed-equal", lambda value: value + torch.zeros_like(value)),
):
    temporary, module, det, paper, model, optimizer, spy = fixture()
    try:
        reject(label, lambda substitute=substitute: module.match_head_proposal_norm(substitute(det), paper))
    finally:
        spy.restore(); temporary.close()

temporary, module, det, paper, model, optimizer, spy = fixture()
try:
    reject("reversed", lambda: module.match_head_proposal_norm(paper, det))
finally:
    spy.restore(); temporary.close()

temporary, module, det, paper, model, optimizer, spy = fixture()
try:
    reject("missing", lambda: spy.finalize({}))
finally:
    spy.restore(); temporary.close()

temporary, module, det, paper, model, optimizer, spy = fixture()
try:
    module.match_head_proposal_norm(det, paper)
    reject("duplicate", lambda: module.match_head_proposal_norm(det, paper))
finally:
    spy.restore(); temporary.close()


def rng_mutation(det, paper):
    torch.rand(1, device=det.device)
    return helper(det, paper)


temporary, module, det, paper, model, optimizer, spy = fixture(rng_mutation)
try:
    reject("RNG mutation", lambda: module.match_head_proposal_norm(det, paper))
finally:
    spy.restore(); temporary.close()

temporary, module, det, paper, model, optimizer, spy = fixture()
base = spy.original
def parameter_mutation(det, paper):
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    return base(det, paper)
spy.original = parameter_mutation
try:
    reject("parameter mutation", lambda: module.match_head_proposal_norm(det, paper))
finally:
    spy.restore(); temporary.close()

temporary, module, det, paper, model, optimizer, spy = fixture()
base = spy.original
def optimizer_mutation(det, paper):
    optimizer.param_groups[0]["lr"] = 0.25
    return base(det, paper)
spy.original = optimizer_mutation
try:
    reject("optimizer mutation", lambda: module.match_head_proposal_norm(det, paper))
finally:
    spy.restore(); temporary.close()


def input_mutation(det, paper):
    det.add_(1.0)
    return helper(det, paper)


temporary, module, det, paper, model, optimizer, spy = fixture(input_mutation)
try:
    reject("input mutation", lambda: module.match_head_proposal_norm(det, paper))
finally:
    spy.restore(); temporary.close()


def output_mutation(det, paper):
    result = list(helper(det, paper))
    result[4] = result[4] + 1.0
    return tuple(result)


temporary, module, det, paper, model, optimizer, spy = fixture(output_mutation)
try:
    reject("output mutation", lambda: module.match_head_proposal_norm(det, paper))
finally:
    spy.restore(); temporary.close()

print(torch.__version__)
print("TASK27_RUNTIME_DIRECT_OBJECT_POSITIVE_NEGATIVE_PASS")
