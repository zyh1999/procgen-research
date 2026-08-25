#!/usr/bin/env python3
"""Positive and negative gates for explicit Task18 policy path loading."""
import importlib.util
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK18 = HERE.parent / "procgen_normmatch_v2_torch_generated_origin_audit_6m_s0_20260825_18"
TASK17 = HERE.parent / "procgen_normmatch_v2_interpreter_path_audit_6m_s0_20260825_17"
POLICY = TASK18 / "origin_safety.py"
BASELINE = TASK18 / "provenance/reproduction_1.json"

support_spec = importlib.util.spec_from_file_location("task19_policy_support", HERE / "policy_namespace_support.py")
support = importlib.util.module_from_spec(support_spec)
support_spec.loader.exec_module(support)

old_env = dict(os.environ)
try:
    os.environ["TASK17_ORIGIN_SAFETY_BASE"] = str((TASK17 / "origin_safety.py").resolve())
    os.environ["TORCH_GENERATED_PROVENANCE_BASELINE"] = str(BASELINE.resolve())
    with tempfile.TemporaryDirectory(prefix="task19_namespace_") as temporary:
        empty = Path(temporary) / "empty"
        empty.mkdir()

        # Positive: caller starts bare; the explicit validated policy is loaded.
        os.environ[support.POLICY_PATH_ENV] = str(POLICY.resolve())
        namespace, ledger = support.load_explicit_policy(empty)
        assert ledger["result"] == "EXPLICIT_ORIGIN_POLICY_PATH_PASS"
        assert ledger["sha256"] == support.EXPECTED_POLICY_SHA256
        assert namespace["snapshot_empty_directory"](empty, "bare")["entries"] == []

        # A normal module load and explicit bare loader expose identical policy API/results.
        module_spec = importlib.util.spec_from_file_location("task18_policy_normal", POLICY)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        public = sorted(name for name, value in namespace.items() if callable(value) and not name.startswith("_"))
        normal_public = sorted(name for name in dir(module) if callable(getattr(module, name)) and not name.startswith("_"))
        assert public == normal_public
        bare_record = namespace["snapshot_empty_directory"](empty, "same")
        normal_record = module.snapshot_empty_directory(empty, "same")
        assert bare_record == normal_record

        # Negative: required variable missing and nonexistent path.
        del os.environ[support.POLICY_PATH_ENV]
        for value in (None, str((Path(temporary) / "missing.py").resolve())):
            if value is not None:
                os.environ[support.POLICY_PATH_ENV] = value
            try:
                support.load_explicit_policy(empty)
            except (RuntimeError, FileNotFoundError):
                pass
            else:
                raise AssertionError("missing/nonexistent explicit path was accepted")
            os.environ.pop(support.POLICY_PATH_ENV, None)

        # Negative: symlink and wrong SHA.
        link = Path(temporary) / "policy_link.py"
        link.symlink_to(POLICY)
        altered = Path(temporary) / "altered.py"
        altered.write_bytes(POLICY.read_bytes() + b"\n# altered\n")
        altered.chmod(0o644)
        for value in (link, altered):
            os.environ[support.POLICY_PATH_ENV] = str(value)
            try:
                support.load_explicit_policy(empty)
            except RuntimeError:
                pass
            else:
                raise AssertionError("symlink/hash-mismatched policy was accepted")

        # The Task18 eager default still fails under a truly bare exec; support must inject metadata.
        os.environ["DEMO_PATH"] = "/explicit/value"
        eager = b"import os\nfrom pathlib import Path\nVALUE=os.environ.get('DEMO_PATH', Path(__file__))\n"
        try:
            exec(compile(eager, "eager_demo.py", "exec"), {})
        except NameError:
            pass
        else:
            raise AssertionError("eager file fallback unexpectedly worked in a bare namespace")
finally:
    os.environ.clear()
    os.environ.update(old_env)

print("TASK19_POLICY_NAMESPACE_SUPPORT_TESTS_PASS")
