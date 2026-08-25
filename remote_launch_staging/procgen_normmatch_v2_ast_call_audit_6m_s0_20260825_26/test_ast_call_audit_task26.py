#!/usr/bin/env python3
"""Task26 AST positive, formatting, and required negative regressions."""
import ast
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ast_runtime_call_audit_task26 import HELPER, validate_trainer_source

TRAINER = ROOT / "procgen_paper_hybrid_head_normmatch_detggn_6m_s0_20260825_14/train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
SOURCE = TRAINER.read_text()


def expect_reject(label, source):
    try:
        validate_trainer_source(source, "negative:" + label)
    except RuntimeError:
        return
    raise RuntimeError("negative AST case was not rejected: " + label)


def find(tree):
    call = next(
        item for item in ast.walk(tree)
        if isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == HELPER
    )
    parent = next(
        item for item in ast.walk(tree)
        if isinstance(item, ast.Assign) and call in ast.walk(item)
    )
    learn = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "learn")
    return call, parent, learn


def rendered(mutator):
    tree = ast.parse(SOURCE)
    call, parent, learn = find(tree)
    mutator(tree, call, parent, learn)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


positive = validate_trainer_source(SOURCE, str(TRAINER))
if positive["call"]["args"] != ["head_direction", "paper_head_proposal"]:
    raise RuntimeError("positive call ledger mismatch")

# Reformatting through AST unparse intentionally changes whitespace and wrapping.
reformatted = ast.unparse(ast.parse(SOURCE))
validate_trainer_source(reformatted, "reformatted-trainer")


def wrong_name_with_string(tree, call, parent, learn):
    call.func.id = "other_helper"
    learn.body.insert(0, ast.Expr(value=ast.Constant(value="match_head_proposal_norm(head_direction, paper_head_proposal)")))


expect_reject("string-only", rendered(wrong_name_with_string))


def reverse_args(tree, call, parent, learn):
    call.args = list(reversed(call.args))


expect_reject("reversed-args", rendered(reverse_args))


def missing_arg(tree, call, parent, learn):
    call.args = call.args[:1]


expect_reject("missing-arg", rendered(missing_arg))


def extra_arg(tree, call, parent, learn):
    call.args.append(ast.Name(id="head_direction", ctx=ast.Load()))


expect_reject("extra-arg", rendered(extra_arg))


def keyword_arg(tree, call, parent, learn):
    call.args = []
    call.keywords = [
        ast.keyword(arg="det_proposal", value=ast.Name(id="head_direction", ctx=ast.Load())),
        ast.keyword(arg="paper_proposal", value=ast.Name(id="paper_head_proposal", ctx=ast.Load())),
    ]


expect_reject("keywords", rendered(keyword_arg))


def attribute_call(tree, call, parent, learn):
    call.func = ast.Attribute(value=ast.Name(id="module", ctx=ast.Load()), attr=HELPER, ctx=ast.Load())


expect_reject("attribute-call", rendered(attribute_call))


def shadowed(tree, call, parent, learn):
    insertion = learn.body.index(next(item for item in learn.body if parent in ast.walk(item)))
    learn.body.insert(insertion, ast.Assign(targets=[ast.Name(id=HELPER, ctx=ast.Store())], value=ast.Name(id="other_helper", ctx=ast.Load())))


expect_reject("shadowed-callee", rendered(shadowed))


def dead_branch(tree, call, parent, learn):
    for node in ast.walk(learn):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and parent in value:
                value[value.index(parent)] = ast.If(test=ast.Constant(value=False), body=[parent], orelse=[])
                return
    raise AssertionError("parent not found")


expect_reject("dead-branch", rendered(dead_branch))


def test_only(tree, call, parent, learn):
    learn.name = "test_learn"


expect_reject("test-only-function", rendered(test_only))


def unused_return(tree, call, parent, learn):
    for node in ast.walk(learn):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and parent in value:
                value[value.index(parent)] = ast.Expr(value=call)
                return
    raise AssertionError("parent not found")


expect_reject("unused-return", rendered(unused_return))

print("TASK26_AST_POSITIVE_FORMAT_NEGATIVE_TESTS_PASS")

