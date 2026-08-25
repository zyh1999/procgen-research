#!/usr/bin/env python3
"""Deployment-only L40S compatibility adapter for frozen Task33 preflight."""
import hashlib
import sys
from pathlib import Path


original = Path(__file__).resolve().with_name("gpuh_preflight.py")
expected = "38e588b1c801840280f10c5330701712ad2098f773a75fae09da5ae8902043b9"
source = original.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == expected

hardware_name = 'assert "H100" in props.name or "H200" in props.name, props.name'
hardware_memory = "assert props.total_memory >= 70_000_000_000, props.total_memory"
assert source.count(hardware_name) == 1
assert source.count(hardware_memory) == 1
adapted = source.replace(
    hardware_name,
    'assert "L40S" in props.name, props.name',
).replace(
    hardware_memory,
    "assert props.total_memory >= 40_000_000_000, props.total_memory",
)
assert adapted.count('assert "L40S" in props.name, props.name') == 1
assert adapted.count("assert props.total_memory >= 40_000_000_000, props.total_memory") == 1
namespace = {
    "__name__": "__main__",
    "__file__": str(original),
    "__package__": None,
}
exec(compile(adapted, str(original), "exec"), namespace, namespace)
print("GPUL_TASK33_DEPLOYMENT_COMPATIBILITY_PASS")
