#!/usr/bin/env python3
"""Task27 static semantic-role/dataflow positive and negative tests."""
import ast
import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREFLIGHT = Path(os.environ.get(
    "TASK27_BASE_PREFLIGHT",
    ROOT / "procgen_paper_hybrid_head_normmatch_detggn_6m_s0_20260825_14/gpuh_preflight_normmatch_v2.py",
))
TRAINER = Path(os.environ.get(
    "TASK27_TRAINER",
    ROOT / "procgen_paper_hybrid_head_normmatch_detggn_6m_s0_20260825_14/train_shared_paper_hybrid_head_detggn_papernorm_v2.py",
))
spec = importlib.util.spec_from_file_location("task27_preflight", HERE / "gpuh_preflight_normmatch_v2_task27.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reject(label, action):
    try:
        action()
    except RuntimeError:
        return
    raise RuntimeError("Task27 static negative not rejected: " + label)


trainer_ledger = module.validate_trainer_path(TRAINER)
if trainer_ledger["call"]["args"] != ["head_direction", "paper_head_proposal"]:
    raise RuntimeError("trainer semantic role ledger changed")
source = PREFLIGHT.read_text()
ledger = module.validate_preflight_dataflow(source, str(PREFLIGHT))
if ledger["boundary_args"] != ["det_proposal", "paper_head_proposal"]:
    raise RuntimeError("preflight semantic boundary ledger changed")

reject("wrong deterministic formula", lambda: module.validate_preflight_dataflow(
    source.replace("weighted_rows, head_solution[0], obs.shape[0], 257", "weighted_rows, head_solution[0], obs.shape[0], 256"),
    "wrong-formula",
))
reject("wrong damping", lambda: module.validate_preflight_dataflow(
    source.replace("weighted_rows, head_rhs, obs.shape[0], 0.5, 257, 1e-18", "weighted_rows, head_rhs, obs.shape[0], 0.1, 257, 1e-18"),
    "wrong-damping",
))
reject("reversed boundary", lambda: module.validate_preflight_dataflow(
    source.replace("det_proposal, paper_head_proposal", "paper_head_proposal, det_proposal"),
    "reversed-boundary",
))
reject("recomputed boundary", lambda: module.validate_preflight_dataflow(
    source.replace("matched = module.match_head_proposal_norm(det_proposal, paper_head_proposal)", "matched = module.match_head_proposal_norm(det_proposal.clone(), paper_head_proposal)"),
    "recomputed-boundary",
))

tree = ast.parse(source)
transformer = module.Task27Transformer()
transformer.visit(tree)
if transformer.assert_replacements != 1 or transformer.boundary_insertions != 1:
    raise RuntimeError("Task27 transformer did not make the exact bounded replacement")

text = (HERE / "gpuh_preflight_normmatch_v2_task27.py").read_text()
for forbidden in ("locals()", "namespace.get(EXPECTED_ARGS", "value equality lookup"):
    if forbidden in text:
        raise RuntimeError("forbidden runtime lookup remains: " + forbidden)

print("TASK27_STATIC_SEMANTIC_BINDING_POSITIVE_NEGATIVE_PASS")
