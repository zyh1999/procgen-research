#!/usr/bin/env python3
"""Validate the explicit policy identity, then record the empty cwd."""
import os
import sys

support_namespace = {}
support_path = os.environ.get("NORMMATCH_V2_POLICY_NAMESPACE_SUPPORT")
if not support_path:
    raise RuntimeError("missing explicit policy namespace support path")
exec(compile(open(support_path, "rb").read(), support_path, "exec"), support_namespace)
namespace, policy_ledger = support_namespace["load_explicit_policy"](
    sys.argv[1], ledger_path=sys.argv[3]
)
record = namespace["snapshot_empty_directory"](sys.argv[1], "before_interpreter")
record["recorded_at_ns"] = __import__("time").time_ns()
record["origin_policy"] = policy_ledger
namespace["write_json"](sys.argv[2], record)
print("TASK19_EXPLICIT_POLICY_PRESTART_PASS")
print("DESIGNATED_EMPTY_PRESTART_PASS")
