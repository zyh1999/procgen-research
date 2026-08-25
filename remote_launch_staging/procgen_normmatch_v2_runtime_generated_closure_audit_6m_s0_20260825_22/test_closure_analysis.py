#!/usr/bin/env python3
"""Static negative contract for the Task22 closure analysis."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
probe = (HERE / "runtime_generated_closure_probe.py").read_text()
analysis = (HERE / "analyze_closure_reproductions.py").read_text()
ast.parse(probe)
ast.parse(analysis)
compile(probe, "runtime_generated_closure_probe.py", "exec")
compile(analysis, "analyze_closure_reproductions.py", "exec")

required_probe = [
    "sys.addaudithook", "traceback.extract_stack", "runpy.run_path",
    "os.lstat", "regular_file", "symlink", "distribution_path",
    "record_hash", "ast_parse", "compile", "forbidden",
]
for token in required_probe:
    assert token in probe, token
required_analysis = [
    "normalized_closure_equal", "no physical artifact",
    "no create/write/rename/delete lifecycle", "no module spec",
    "no loader identity", "formal_clean_room_audit_permitted",
]
for token in required_analysis:
    assert token in analysis, token
assert "APPROVED_RUNTIME_GENERATED_THIRDPARTY_MODULE" not in probe
assert "origin.name" not in probe
print("TASK22_CLOSURE_STATIC_NEGATIVE_CONTRACT_PASS")
