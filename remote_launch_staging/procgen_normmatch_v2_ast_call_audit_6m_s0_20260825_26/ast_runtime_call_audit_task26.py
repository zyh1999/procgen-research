#!/usr/bin/env python3
"""Task26 exact AST call-flow and side-effect-free runtime-spy audit."""
import ast
import copy
import hashlib
import json
from pathlib import Path


HELPER = "match_head_proposal_norm"
EXPECTED_ARGS = ("head_direction", "paper_head_proposal")
EXPECTED_RESULT = "head_target_proposal"


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _span(node):
    return {
        "lineno": node.lineno,
        "col_offset": node.col_offset,
        "end_lineno": node.end_lineno,
        "end_col_offset": node.end_col_offset,
    }


def _parents(tree):
    result = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            result[child] = parent
    return result


def _bound_names(node):
    names = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
            names.add(item.id)
        elif isinstance(item, ast.arg):
            names.add(item.arg)
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if item is not node:
                names.add(item.name)
        elif isinstance(item, ast.alias):
            names.add(item.asname or item.name.split(".")[0])
    return names


def _ancestor_chain(node, parents):
    chain = []
    current = node
    while current in parents:
        current = parents[current]
        chain.append(current)
    return chain


def _literal_false(test):
    try:
        return ast.literal_eval(test) is False
    except (ValueError, TypeError, SyntaxError):
        return False


def _name_loads(node, name):
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id == name
    ]


def validate_trainer_source(source, source_label="<trainer>"):
    """Prove the frozen helper definition, exact live call, and update flow."""
    tree = ast.parse(source, filename=source_label)
    parents = _parents(tree)
    definitions = [
        item for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == HELPER
    ]
    if len(definitions) != 1:
        raise RuntimeError("expected exactly one module-level norm-match definition")
    definition = definitions[0]
    if isinstance(definition, ast.AsyncFunctionDef):
        raise RuntimeError("norm-match definition must be synchronous")

    calls = [
        item for item in ast.walk(tree)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == HELPER
    ]
    if len(calls) != 1:
        raise RuntimeError("expected exactly one direct norm-match call")
    call = calls[0]
    if len(call.args) != 2 or call.keywords:
        raise RuntimeError("norm-match call must have exactly two positional arguments")
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        raise RuntimeError("starred norm-match arguments are forbidden")
    actual_args = tuple(
        argument.id if isinstance(argument, ast.Name) and isinstance(argument.ctx, ast.Load) else None
        for argument in call.args
    )
    if actual_args != EXPECTED_ARGS:
        raise RuntimeError("norm-match argument identity/order mismatch: " + repr(actual_args))

    chain = _ancestor_chain(call, parents)
    enclosing_functions = [
        item for item in chain
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
    ]
    if (
        len(enclosing_functions) != 2
        or not all(isinstance(item, ast.FunctionDef) for item in enclosing_functions)
        or [item.name for item in enclosing_functions] != ["Advantage_Update", "learn"]
    ):
        raise RuntimeError("norm-match call is not in the exact production minibatch update path")
    update_function = enclosing_functions[0]
    if any(HELPER in _bound_names(scope) for scope in enclosing_functions):
        raise RuntimeError("norm-match helper is shadowed in the production scope")
    for ancestor in chain:
        if isinstance(ancestor, ast.If) and _literal_false(ancestor.test):
            raise RuntimeError("norm-match call is in a statically dead branch")
        if isinstance(ancestor, ast.If) and any(
            name.id in {"PYTEST_CURRENT_TEST", "TESTING", "UNIT_TEST"}
            for name in ast.walk(ancestor.test) if isinstance(name, ast.Name)
        ):
            raise RuntimeError("norm-match call is in a test-only branch")

    parent = parents.get(call)
    if not isinstance(parent, (ast.Assign, ast.AnnAssign)):
        raise RuntimeError("norm-match return is not assigned")
    target = parent.targets[0] if isinstance(parent, ast.Assign) else parent.target
    if not isinstance(target, (ast.Tuple, ast.List)) or not target.elts:
        raise RuntimeError("norm-match return is not unpacked into the frozen update tuple")
    target_names = tuple(
        item.id if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store) else None
        for item in target.elts
    )
    expected_targets = (
        "head_target_proposal", "head_det_proposal_l2", "head_paper_proposal_l2",
        "head_normmatch_scale", "head_target_proposal_l2", "head_proposal_cosine",
    )
    if target_names != expected_targets:
        raise RuntimeError("norm-match return target tuple changed")

    later_loads = [
        item for item in _name_loads(update_function, EXPECTED_RESULT)
        if item.lineno > call.end_lineno
    ]
    if not later_loads:
        raise RuntimeError("norm-match return is unused")
    append_flows = []
    for load in later_loads:
        load_chain = _ancestor_chain(load, parents)
        append = next((
            item for item in load_chain
            if isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "append"
            and isinstance(item.func.value, ast.Name)
            and item.func.value.id == "target_preclip_grads"
        ), None)
        if append is not None:
            append_flows.append((load, append))
    if len(append_flows) != 1:
        raise RuntimeError("norm-match return does not flow exactly once into target head update")
    load, append = append_flows[0]
    if not any(
        isinstance(item, ast.UnaryOp) and isinstance(item.op, ast.USub)
        for item in _ancestor_chain(load, parents)
        if item is not append
    ):
        raise RuntimeError("frozen head proposal-to-gradient sign flow changed")

    definition_dump = _dump(definition)
    call_dump = _dump(call)
    assign_dump = _dump(parent)
    flow_dump = _dump(append)
    return {
        "result": "TASK26_AST_CALL_AUDIT_PASS",
        "source_label": source_label,
        "source_sha256": _sha_bytes(source.encode()),
        "definition": {
            "name": HELPER,
            "span": _span(definition),
            "normalized_ast": definition_dump,
            "normalized_ast_sha256": _sha_bytes(definition_dump.encode()),
        },
        "call": {
            "callee": HELPER,
            "args": list(actual_args),
            "keywords": 0,
            "starargs": 0,
            "containing_function": update_function.name,
            "outer_production_function": enclosing_functions[1].name,
            "span": _span(call),
            "normalized_ast": call_dump,
            "normalized_ast_sha256": _sha_bytes(call_dump.encode()),
        },
        "return_flow": {
            "assignment_targets": list(target_names),
            "assignment_span": _span(parent),
            "assignment_normalized_ast_sha256": _sha_bytes(assign_dump.encode()),
            "head_update_append_span": _span(append),
            "head_update_append_normalized_ast_sha256": _sha_bytes(flow_dump.encode()),
            "target": "target_preclip_grads.append(-head_target_proposal[...].view_as(parameter))",
        },
        "format_independent": True,
        "unshadowed_module_definition": True,
        "live_production_minibatch_path": True,
    }


def validate_trainer_path(path, ledger_path=None):
    path = Path(path)
    ledger = validate_trainer_source(path.read_text(), str(path))
    if ledger_path is not None:
        Path(ledger_path).write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


def _clone_tree(value):
    if hasattr(value, "detach") and hasattr(value, "clone"):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _equal_tree(torch, first, second):
    if torch.is_tensor(first) or torch.is_tensor(second):
        return torch.is_tensor(first) and torch.is_tensor(second) and torch.equal(first, second)
    if type(first) is not type(second):
        return False
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(_equal_tree(torch, first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(_equal_tree(torch, a, b) for a, b in zip(first, second))
    return first == second


class RuntimeSemanticSpy:
    """One-call wrapper for the actual preflight update, with full state audit."""

    def __init__(self, module, namespace, model, optimizer, ledger_path):
        self.module = module
        self.namespace = namespace
        self.model = model
        self.optimizer = optimizer
        self.ledger_path = Path(ledger_path)
        self.original = module.match_head_proposal_norm
        self.calls = []
        self.return_value = None
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

    def _assert_snapshot_equal(self, torch, first, second):
        if not torch.equal(first["cpu_rng"], second["cpu_rng"]):
            raise RuntimeError("runtime wrapper changed CPU RNG")
        if not _equal_tree(torch, first["cuda_rng"], second["cuda_rng"]):
            raise RuntimeError("runtime wrapper changed CUDA RNG")
        if not _equal_tree(torch, first["parameters"], second["parameters"]):
            raise RuntimeError("runtime wrapper changed model parameters")
        if not _equal_tree(torch, first["optimizer"], second["optimizer"]):
            raise RuntimeError("runtime wrapper changed optimizer state")

    def __call__(self, *args, **kwargs):
        torch = self.module.torch
        if kwargs or len(args) != 2:
            raise RuntimeError("runtime norm-match call signature changed")
        if self.calls:
            raise RuntimeError("runtime norm-match call duplicated")
        expected_first = self.namespace.get(EXPECTED_ARGS[0])
        expected_second = self.namespace.get(EXPECTED_ARGS[1])
        if args[0] is not expected_first or args[1] is not expected_second:
            raise RuntimeError("runtime norm-match argument object identity mismatch")
        for index, tensor in enumerate(args):
            if not torch.is_tensor(tensor) or tensor.ndim != 1 or not torch.isfinite(tensor).all():
                raise RuntimeError("runtime norm-match tensor contract failed")
            if tensor.shape != expected_first.shape or tensor.dtype != expected_first.dtype or tensor.device != expected_first.device:
                raise RuntimeError("runtime norm-match shape/dtype/device mismatch")
        input_before = [item.detach().clone() for item in args]
        state_before = self._snapshot(torch)
        result = self.original(*args)
        state_after = self._snapshot(torch)
        self._assert_snapshot_equal(torch, state_before, state_after)
        if not all(torch.equal(before, after) for before, after in zip(input_before, args)):
            raise RuntimeError("runtime wrapper mutated an input tensor")
        if not isinstance(result, tuple) or len(result) != 6:
            raise RuntimeError("runtime norm-match return contract changed")
        if not all(torch.is_tensor(item) and torch.isfinite(item).all() for item in result):
            raise RuntimeError("runtime norm-match returned a nonfinite/non-tensor value")
        if not torch.allclose(result[4], result[2], rtol=2e-6, atol=2e-8):
            raise RuntimeError("runtime norm-match return does not match Paper proposal norm")
        self.calls.append({
            "first_object_identity": hex(id(args[0])),
            "second_object_identity": hex(id(args[1])),
            "shape": list(args[0].shape),
            "dtype": str(args[0].dtype),
            "device": str(args[0].device),
            "finite": True,
        })
        self.return_value = result
        return result

    def finalize(self):
        torch = self.module.torch
        if len(self.calls) != 1 or self.return_value is None:
            raise RuntimeError("runtime expected exactly one wrapped update call")
        if self.namespace.get("matched") is not self.return_value:
            raise RuntimeError("wrapped return did not remain in actual one-step control flow")
        if self.namespace.get(EXPECTED_RESULT) is not self.return_value[0]:
            raise RuntimeError("wrapped target proposal identity was not preserved")
        args = (self.namespace[EXPECTED_ARGS[0]], self.namespace[EXPECTED_ARGS[1]])
        self.restore()
        state_before = self._snapshot(torch)
        unwrapped = self.original(*args)
        state_after = self._snapshot(torch)
        self._assert_snapshot_equal(torch, state_before, state_after)
        if not _equal_tree(torch, self.return_value, unwrapped):
            raise RuntimeError("wrapped and unwrapped norm-match results are not bit-identical")
        replacement = self.namespace.get("replacement")
        target_preclip = self.namespace.get("target_preclip")
        if not isinstance(replacement, dict) or not isinstance(target_preclip, dict):
            raise RuntimeError("actual head-update control flow did not complete")
        ledger = {
            "result": "TASK26_RUNTIME_IDENTITY_AUDIT_PASS",
            "wrapped_expected_call_count": 1,
            "wrapped_observed_call_count": len(self.calls),
            "args": list(EXPECTED_ARGS),
            "call": self.calls[0],
            "argument_identity_equal": True,
            "input_tensors_unchanged": True,
            "rng_unchanged": True,
            "model_parameters_unchanged": True,
            "optimizer_state_unchanged": True,
            "control_flow_completed": True,
            "return_used_by_head_update": True,
            "norm_match_return": "PASS",
            "wrapped_unwrapped_bit_identical": True,
        }
        self.ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        return ledger


def prepare_replacement(trainer, ast_ledger_path, module, namespace, runtime_ledger_path, optimizer, model):
    ledger = validate_trainer_path(trainer, ast_ledger_path)
    spy = RuntimeSemanticSpy(module, namespace, model, optimizer, runtime_ledger_path)
    return ledger, spy
