#!/usr/bin/env python3
"""Execute the frozen Task41 preflight with only its actor selection replaced."""
import argparse
import hashlib
import sys
from pathlib import Path


TASK41_PREFLIGHT_SHA256 = "cd87d3bbc9030df2eed536cea1b0acf8c88bf3f6b858757e69ced040ca0f9660"
OLD = "        return torch.log_softmax(logits, -1)[0, action]\n"
NEW = """        if action.dtype != torch.long:
            raise TypeError(\"actor action must have torch.long dtype\")
        logp_all = torch.log_softmax(logits, dim=-1)
        selected_logp = torch.gather(
            logp_all,
            dim=-1,
            index=action.to(torch.long).reshape(*logp_all.shape[:-1], 1),
        ).squeeze(-1)
        return selected_logp.reshape(())
"""


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--task41-preflight", required=True)
known, remaining = parser.parse_known_args()
source_path = Path(known.task41_preflight).resolve()
source_bytes = source_path.read_bytes()
if hashlib.sha256(source_bytes).hexdigest() != TASK41_PREFLIGHT_SHA256:
    raise AssertionError("frozen Task41 preflight hash drift")
source = source_bytes.decode()
if source.count(OLD) != 1:
    raise AssertionError("Task41 actor-index expression identity drift")
patched = source.replace(OLD, NEW)
if patched.count(NEW) != 1 or OLD in patched:
    raise AssertionError("Task42 gather replacement was not exact and unique")
sys.path.insert(0, str(source_path.parent))
sys.argv = [str(source_path)] + remaining
namespace = {"__name__": "__main__", "__file__": str(source_path),
             "__package__": None, "__cached__": None}
exec(compile(patched, str(source_path) + "[TASK42_GATHER_ONLY]", "exec"), namespace)
