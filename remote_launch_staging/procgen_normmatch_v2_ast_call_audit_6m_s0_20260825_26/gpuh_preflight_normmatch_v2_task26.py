#!/usr/bin/env python3
"""Task26 versioned preflight: replace one frozen text Assert by AST/runtime audit."""
import ast
import hashlib
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
audit_namespace = runpy.run_path(str(HERE / "ast_runtime_call_audit_task26.py"))
prepare_replacement = audit_namespace["prepare_replacement"]

FROZEN_PREFLIGHT_SHA256 = "b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc"
FROZEN_TRAINER_SHA256 = "0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b"
OLD_SUBSTRING = "match_head_proposal_norm(head_direction, paper_head_proposal)"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReplaceFrozenSubstringAssert(ast.NodeTransformer):
    def __init__(self):
        self.replacements = 0

    def visit_Assert(self, node):
        self.generic_visit(node)
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.In)
            and len(test.comparators) == 1
            and isinstance(test.left, ast.Constant)
            and test.left.value == OLD_SUBSTRING
            and isinstance(test.comparators[0], ast.Name)
            and test.comparators[0].id == "trainer_text"
        ):
            self.replacements += 1
            replacement = ast.parse(
                "__task26_ast_ledger, __task26_runtime_spy = "
                "task26_prepare_replacement("
                "trainer, evidence_dir / 'ast_call_ledger.json', module, globals(), "
                "evidence_dir / 'runtime_identity_ledger.json', production_optimizer, model)"
            ).body[0]
            return ast.copy_location(replacement, node)
        return node


explicit_candidate = Path(sys.argv[1]).resolve(strict=True)
if sha256(explicit_candidate) == FROZEN_TRAINER_SHA256:
    # Hermetic deployment spelling: this versioned file occupies the frozen
    # preflight name and the byte-identical predecessor is retained beside it.
    base_preflight = HERE / "gpuh_preflight_normmatch_v2_task25_frozen.py"
    forwarded_argv = sys.argv[1:]
else:
    # Repository/local spelling used by regressions and explicit audits.
    base_preflight = explicit_candidate
    forwarded_argv = sys.argv[2:]
if sha256(base_preflight) != FROZEN_PREFLIGHT_SHA256:
    raise RuntimeError("frozen Task14 preflight identity mismatch")
if len(forwarded_argv) < 1 or sha256(forwarded_argv[0]) != FROZEN_TRAINER_SHA256:
    raise RuntimeError("frozen V2 trainer identity mismatch")

source = base_preflight.read_text()
tree = ast.parse(source, filename=str(base_preflight))
transformer = ReplaceFrozenSubstringAssert()
tree = transformer.visit(tree)
ast.fix_missing_locations(tree)
if transformer.replacements != 1:
    raise RuntimeError("expected exactly one frozen substring Assert replacement")

namespace = {
    "__name__": "__main__",
    "__file__": str(base_preflight),
    "task26_prepare_replacement": prepare_replacement,
}
old_argv = sys.argv[:]
try:
    sys.argv = [str(base_preflight)] + forwarded_argv
    exec(compile(tree, str(base_preflight), "exec"), namespace)
    spy = namespace.get("__task26_runtime_spy")
    if spy is None:
        raise RuntimeError("Task26 runtime spy was not installed")
    runtime_ledger = spy.finalize()
finally:
    candidate = namespace.get("__task26_runtime_spy")
    if candidate is not None:
        candidate.restore()
    sys.argv = old_argv

print("TASK26_AST_CALL_AUDIT_PASS")
print("TASK26_RUNTIME_IDENTITY_AUDIT_PASS")
print("task26_wrapped_expected_call_count=" + str(runtime_ledger["wrapped_expected_call_count"]))
print("task26_wrapped_unwrapped_bit_identical=PASS")
