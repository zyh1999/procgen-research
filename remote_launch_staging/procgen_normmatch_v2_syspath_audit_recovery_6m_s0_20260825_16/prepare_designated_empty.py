#!/usr/bin/env python3
"""Record the designated empty cwd before the audited interpreter starts."""
import os
import sys

namespace = {}
exec(compile(open(sys.argv[1]).read(), sys.argv[1], "exec"), namespace)
record = namespace["snapshot_empty_directory"](sys.argv[2], "before_interpreter")
record["recorded_at_ns"] = __import__("time").time_ns()
namespace["write_json"](sys.argv[3], record)
print("DESIGNATED_EMPTY_PRESTART_PASS")
