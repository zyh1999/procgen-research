#!/usr/bin/env python3
"""Task27 preflight: explicit semantic-role binding at the real one-step boundary."""
import ast
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import MappingProxyType

HERE = Path(__file__).resolve().parent
audit_path = HERE / "ast_runtime_call_audit_task26.py"
if not audit_path.exists():
    audit_path = HERE.parent / "procgen_normmatch_v2_ast_call_audit_6m_s0_20260825_26/ast_runtime_call_audit_task26.py"
audit_ns = runpy.run_path(str(audit_path))
validate_trainer_path = audit_ns["validate_trainer_path"]
_clone_tree = audit_ns["_clone_tree"]
_equal_tree = audit_ns["_equal_tree"]

FROZEN_PREFLIGHT_SHA256 = "b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc"
FROZEN_TRAINER_SHA256 = "0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b"
OLD_SUBSTRING = "match_head_proposal_norm(head_direction, paper_head_proposal)"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tensor_metadata(torch, tensor):
    raw = tensor.detach().contiguous().cpu().numpy().tobytes()
    storage = tensor.untyped_storage()
    return {
        "object_identity": hex(id(tensor)),
        "data_pointer": tensor.data_ptr(),
        "storage_data_pointer": storage.data_ptr(),
        "storage_offset": tensor.storage_offset(),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "version_counter": tensor._version,
        "requires_grad": tensor.requires_grad,
        "finite": bool(torch.isfinite(tensor).all()),
        "value_sha256": hashlib.sha256(raw).hexdigest(),
        "value_numel": tensor.numel(),
        "value_l2": float(torch.linalg.vector_norm(tensor)),
        "value_sum": float(tensor.sum()),
    }


def validate_preflight_dataflow(source, label):
    tree = ast.parse(source, filename=label)
    assignments = {
        target.id: node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    solve = assignments.get("head_solution")
    deterministic = assignments.get("det_proposal")
    paper = assignments.get("paper_head_proposal")
    matched = assignments.get("matched")
    if any(item is None for item in (solve, deterministic, paper, matched)):
        raise RuntimeError("frozen preflight proposal dataflow assignment missing")
    def is_obs_shape_zero(node):
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "obs"
            and node.value.attr == "shape"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == 0
        )
    if not (
        isinstance(solve.value, ast.Call)
        and isinstance(solve.value.func, ast.Attribute)
        and isinstance(solve.value.func.value, ast.Name)
        and solve.value.func.value.id == "module"
        and solve.value.func.attr == "solve_head_critic_b_fp64"
        and len(solve.value.args) == 6
        and isinstance(solve.value.args[0], ast.Name) and solve.value.args[0].id == "weighted_rows"
        and isinstance(solve.value.args[1], ast.Name) and solve.value.args[1].id == "head_rhs"
        and is_obs_shape_zero(solve.value.args[2])
        and isinstance(solve.value.args[3], ast.Constant) and solve.value.args[3].value == 0.5
        and isinstance(solve.value.args[4], ast.Constant) and solve.value.args[4].value == 257
        and isinstance(solve.value.args[5], ast.Constant) and solve.value.args[5].value == 1e-18
    ):
        raise RuntimeError("frozen FP64 deterministic head solve contract changed")
    if not (
        isinstance(deterministic.value, ast.Call)
        and isinstance(deterministic.value.func, ast.Attribute)
        and isinstance(deterministic.value.func.value, ast.Name)
        and deterministic.value.func.value.id == "module"
        and deterministic.value.func.attr == "chunked_transpose_mv_fp64"
        and len(deterministic.value.args) == 4
        and isinstance(deterministic.value.args[0], ast.Name) and deterministic.value.args[0].id == "weighted_rows"
        and isinstance(deterministic.value.args[1], ast.Subscript)
        and isinstance(deterministic.value.args[1].value, ast.Name)
        and deterministic.value.args[1].value.id == "head_solution"
        and is_obs_shape_zero(deterministic.value.args[2])
        and isinstance(deterministic.value.args[3], ast.Constant) and deterministic.value.args[3].value == 257
    ):
        raise RuntimeError("frozen deterministic proposal construction changed")
    if not (
        isinstance(paper.value, ast.UnaryOp) and isinstance(paper.value.op, ast.USub)
        and isinstance(paper.value.operand, ast.Call)
        and isinstance(paper.value.operand.func, ast.Attribute)
        and paper.value.operand.func.attr == "cat"
    ):
        raise RuntimeError("frozen counterfactual Paper proposal construction changed")
    if not (
        isinstance(matched.value, ast.Call)
        and isinstance(matched.value.func, ast.Attribute)
        and isinstance(matched.value.func.value, ast.Name)
        and matched.value.func.value.id == "module"
        and matched.value.func.attr == "match_head_proposal_norm"
        and len(matched.value.args) == 2
        and not matched.value.keywords
        and [item.id if isinstance(item, ast.Name) else None for item in matched.value.args]
        == ["det_proposal", "paper_head_proposal"]
    ):
        raise RuntimeError("frozen preflight norm-match boundary changed")
    records = {}
    for key, node in (
        ("fp64_solve", solve), ("deterministic_proposal", deterministic),
        ("paper_proposal", paper), ("normmatch_call", matched),
    ):
        dump = ast.dump(node, annotate_fields=True, include_attributes=False)
        records[key] = {
            "span": [node.lineno, node.col_offset, node.end_lineno, node.end_col_offset],
            "normalized_ast_sha256": hashlib.sha256(dump.encode()).hexdigest(),
        }
    return {
        "result": "TASK27_PREFLIGHT_DATAFLOW_PASS",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "deterministic_formula": "chunked_transpose_mv_fp64(weighted_rows, head_solution[0], obs.shape[0], 257)",
        "solver": "solve_head_critic_b_fp64(weighted_rows, head_rhs, obs.shape[0], 0.5, 257, 1e-18)",
        "counterfactual_paper_formula": "-torch.cat(paper_full[head_names])",
        "boundary_args": ["det_proposal", "paper_head_proposal"],
        "records": records,
    }


class SemanticBoundRuntimeSpy:
    def __init__(self, module, expected_det, expected_paper, model, optimizer, ledger_path, ast_ledger, dataflow_ledger):
        self.module = module
        self.expected_det = expected_det
        self.expected_paper = expected_paper
        self.model = model
        self.optimizer = optimizer
        self.ledger_path = Path(ledger_path)
        self.ast_ledger = ast_ledger
        self.dataflow_ledger = dataflow_ledger
        self.original = module.match_head_proposal_norm
        self.calls = 0
        self.return_value = None
        self.inputs_before = None
        self.mapping = MappingProxyType({
            "deterministic_head_proposal": MappingProxyType({
                "trainer_ast_name": "head_direction",
                "preflight_semantic_role": "det_proposal",
                "preflight_object": expected_det,
            }),
            "counterfactual_paper_head_proposal": MappingProxyType({
                "trainer_ast_name": "paper_head_proposal",
                "preflight_semantic_role": "paper_head_proposal",
                "preflight_object": expected_paper,
            }),
        })
        self.module.match_head_proposal_norm = self

    def restore(self):
        if self.module.match_head_proposal_norm is self:
            self.module.match_head_proposal_norm = self.original

    def _snapshot(self, torch):
        return {
            "cpu_rng": torch.get_rng_state().clone(),
            "cuda_rng": [item.clone() for item in torch.cuda.get_rng_state_all()],
            "parameters": {name: item.detach().clone() for name, item in self.model.named_parameters()},
            "optimizer": _clone_tree(self.optimizer.state_dict()),
        }

    def _assert_state(self, torch, first, second):
        if not torch.equal(first["cpu_rng"], second["cpu_rng"]):
            raise RuntimeError("semantic spy changed CPU RNG")
        if not _equal_tree(torch, first["cuda_rng"], second["cuda_rng"]):
            raise RuntimeError("semantic spy changed CUDA RNG")
        if not _equal_tree(torch, first["parameters"], second["parameters"]):
            raise RuntimeError("semantic spy changed model parameters")
        if not _equal_tree(torch, first["optimizer"], second["optimizer"]):
            raise RuntimeError("semantic spy changed optimizer state")

    def __call__(self, *args, **kwargs):
        torch = self.module.torch
        if kwargs or len(args) != 2:
            raise RuntimeError("semantic spy call signature changed")
        if self.calls != 0:
            raise RuntimeError("semantic spy observed a duplicate call")
        actual_det, actual_paper = args
        if actual_det is not self.expected_det:
            raise RuntimeError("deterministic proposal is not the captured preflight object")
        if actual_paper is not self.expected_paper:
            raise RuntimeError("Paper proposal is not the captured preflight object")
        if actual_det.data_ptr() != self.expected_det.data_ptr() or actual_paper.data_ptr() != self.expected_paper.data_ptr():
            raise RuntimeError("semantic proposal storage identity changed")
        if actual_det.shape != actual_paper.shape or actual_det.dtype != actual_paper.dtype or actual_det.device != actual_paper.device:
            raise RuntimeError("semantic proposal shape/dtype/device mismatch")
        if not torch.isfinite(actual_det).all() or not torch.isfinite(actual_paper).all():
            raise RuntimeError("semantic proposal is nonfinite")
        self.inputs_before = (actual_det.detach().clone(), actual_paper.detach().clone())
        metadata_before = (_tensor_metadata(torch, actual_det), _tensor_metadata(torch, actual_paper))
        state_before = self._snapshot(torch)
        result = self.original(actual_det, actual_paper)
        state_after = self._snapshot(torch)
        self._assert_state(torch, state_before, state_after)
        if not torch.equal(self.inputs_before[0], actual_det) or not torch.equal(self.inputs_before[1], actual_paper):
            raise RuntimeError("semantic spy input mutation")
        if not isinstance(result, tuple) or len(result) != 6 or not all(torch.is_tensor(item) and torch.isfinite(item).all() for item in result):
            raise RuntimeError("semantic spy return contract changed")
        if not torch.allclose(result[4], result[2], rtol=2e-6, atol=2e-8):
            raise RuntimeError("semantic spy norm-match return failed")
        self.calls = 1
        self.return_value = result
        self.metadata_before = metadata_before
        return result

    def finalize(self, namespace):
        torch = self.module.torch
        if self.calls != 1 or self.return_value is None:
            raise RuntimeError("semantic spy expected exactly one call")
        if namespace.get("matched") is not self.return_value or namespace.get("target_head_proposal") is not self.return_value[0]:
            raise RuntimeError("semantic spy return did not flow through one-step update")
        metadata_after = (_tensor_metadata(torch, self.expected_det), _tensor_metadata(torch, self.expected_paper))
        if metadata_after != self.metadata_before:
            raise RuntimeError("semantic proposal metadata/value changed")
        self.restore()
        state_before = self._snapshot(torch)
        unwrapped = self.original(self.expected_det, self.expected_paper)
        state_after = self._snapshot(torch)
        self._assert_state(torch, state_before, state_after)
        if not _equal_tree(torch, self.return_value, unwrapped):
            raise RuntimeError("semantic wrapped/unwrapped output mismatch")
        ledger = {
            "result": "TASK27_RUNTIME_SEMANTIC_BINDING_PASS",
            "semantic_mapping": {
                key: {
                    "trainer_ast_name": item["trainer_ast_name"],
                    "preflight_semantic_role": item["preflight_semantic_role"],
                    "tensor": _tensor_metadata(torch, item["preflight_object"]),
                }
                for key, item in self.mapping.items()
            },
            "mapping_immutable": True,
            "direct_object_capture_no_name_lookup": True,
            "actual_det_is_expected_det": True,
            "actual_paper_is_expected_paper": True,
            "call_count": self.calls,
            "inputs_unchanged": True,
            "norm_match_return": True,
            "wrapped_unwrapped_rng_outputs_parameters_optimizer_telemetry_bit_identical": True,
            "science_spy_or_hook_residue": False,
            "trainer_ast_call_sha256": self.ast_ledger["call"]["normalized_ast_sha256"],
            "preflight_dataflow": self.dataflow_ledger,
        }
        self.ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        return ledger


def prepare_runtime_binding(module, det_proposal, paper_head_proposal, model, optimizer, ledger_path, ast_ledger, dataflow_ledger):
    return SemanticBoundRuntimeSpy(
        module, det_proposal, paper_head_proposal, model, optimizer,
        ledger_path, ast_ledger, dataflow_ledger,
    )


class Task27Transformer(ast.NodeTransformer):
    def __init__(self):
        self.assert_replacements = 0
        self.boundary_insertions = 0

    def visit_Assert(self, node):
        self.generic_visit(node)
        test = node.test
        if (
            isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.In)
            and len(test.comparators) == 1 and isinstance(test.left, ast.Constant)
            and test.left.value == OLD_SUBSTRING and isinstance(test.comparators[0], ast.Name)
            and test.comparators[0].id == "trainer_text"
        ):
            self.assert_replacements += 1
            return ast.copy_location(ast.parse(
                "__task27_ast_ledger = task27_validate_trainer_path("
                "trainer, evidence_dir / 'ast_call_ledger.json')"
            ).body[0], node)
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (
            len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "matched" and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "match_head_proposal_norm"
            and [item.id if isinstance(item, ast.Name) else None for item in node.value.args]
            == ["det_proposal", "paper_head_proposal"] and not node.value.keywords
        ):
            self.boundary_insertions += 1
            setup = ast.parse(
                "__task27_runtime_spy = task27_prepare_runtime_binding("
                "module, det_proposal, paper_head_proposal, model, production_optimizer, "
                "evidence_dir / 'runtime_semantic_binding_ledger.json', "
                "__task27_ast_ledger, __task27_preflight_dataflow_ledger)"
            ).body[0]
            return [ast.copy_location(setup, node), node]
        return node


def main():
    explicit = Path(sys.argv[1]).resolve(strict=True)
    if sha256(explicit) == FROZEN_TRAINER_SHA256:
        base = HERE / "gpuh_preflight_normmatch_v2_task25_frozen.py"
        forwarded = sys.argv[1:]
    else:
        base = explicit
        forwarded = sys.argv[2:]
    if sha256(base) != FROZEN_PREFLIGHT_SHA256:
        raise RuntimeError("frozen Task14 preflight identity mismatch")
    if not forwarded or sha256(forwarded[0]) != FROZEN_TRAINER_SHA256:
        raise RuntimeError("frozen V2 trainer identity mismatch")
    source = base.read_text()
    dataflow = validate_preflight_dataflow(source, str(base))
    tree = ast.parse(source, filename=str(base))
    transformer = Task27Transformer()
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    if transformer.assert_replacements != 1 or transformer.boundary_insertions != 1:
        raise RuntimeError("Task27 expected exactly one assertion replacement and boundary insertion")
    namespace = {
        "__name__": "__main__", "__file__": str(base),
        "task27_validate_trainer_path": validate_trainer_path,
        "task27_prepare_runtime_binding": prepare_runtime_binding,
        "__task27_preflight_dataflow_ledger": dataflow,
    }
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(base)] + forwarded
        exec(compile(tree, str(base), "exec"), namespace)
        spy = namespace.get("__task27_runtime_spy")
        if spy is None:
            raise RuntimeError("Task27 runtime semantic spy was not installed")
        ledger = spy.finalize(namespace)
    finally:
        candidate = namespace.get("__task27_runtime_spy")
        if candidate is not None:
            candidate.restore()
        sys.argv = old_argv
    print("TASK27_RUNTIME_SEMANTIC_BINDING_PASS")
    print("task27_call_count=" + str(ledger["call_count"]))
    print("task27_wrapped_unwrapped_bit_identical=PASS")


if __name__ == "__main__":
    main()
