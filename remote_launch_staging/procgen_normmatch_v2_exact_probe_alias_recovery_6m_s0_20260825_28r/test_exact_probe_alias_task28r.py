#!/usr/bin/env python3
"""Task28R exact probe alias positive and required negative tests."""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("task28r", HERE / "exact_probe_alias_extension_task28r.py")
task28r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task28r)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(data):
    return hashlib.sha1(("blob " + str(len(data)) + "\0").encode() + data).hexdigest()


def reject(label, action):
    try:
        action()
    except RuntimeError:
        return
    raise RuntimeError("Task28R negative not rejected: " + label)


def identity(path, module_key="task28r_exact_probe_test"):
    value = os.stat(path)
    data = Path(path).read_bytes()
    return {
        "frozen_git_commit": "test-commit", "repository_relative_path": "frozen/runtime_closure_probe_task23.py",
        "git_mode": "100644", "git_blob": blob(data), "sha256": sha(data),
        "basename": Path(path).name, "raw_path": str(path),
        "resolved_path": str(Path(path).resolve(strict=True)), "module_key": module_key,
        "loader_module": "_frozen_importlib_external", "loader_name": "SourceFileLoader",
        "package": None, "spec": None, "device": value.st_dev, "inode": value.st_ino,
        "uid": value.st_uid, "gid": value.st_gid, "mode": stat.S_IMODE(value.st_mode), "size": value.st_size,
    }


class Fixture:
    def __init__(self, root, process_role="closure-audit-entrypoint", content=b"probe\n", module_key="task28r_exact_probe_test"):
        self.root = Path(root)
        self.probe = self.root / "runtime_closure_probe_task23.py"
        self.probe.write_bytes(content)
        self.launcher = self.root / "science.sbatch"
        self.launcher.write_text("python trainer.py --seed 0\n")
        self.expected = identity(self.probe, module_key)
        self.module = types.ModuleType(module_key)
        self.loader = importlib.machinery.SourceFileLoader(module_key, str(self.probe))
        self.module.__file__ = str(self.probe)
        self.module.__loader__ = self.loader
        self.module.__spec__ = None
        self.module.__package__ = None
        self.previous = sys.modules.get(module_key)
        sys.modules[module_key] = self.module
        self.old_argv = sys.argv[:]
        sys.argv = [str(self.probe)]
        self.session = task28r.ExactFrozenProbeAlias(
            expected=self.expected, process_role=process_role,
            science_launcher=self.launcher,
        )

    def finish(self):
        self.session.close()
        sys.argv = self.old_argv
        if self.previous is None:
            sys.modules.pop(self.expected["module_key"], None)
        else:
            sys.modules[self.expected["module_key"]] = self.previous

    def approve_finalize(self, manifest=None):
        task28r.SCIENCE_LAUNCHER_SHA256 = sha(self.launcher.read_bytes())
        result = self.session.approve(str(self.probe))
        ledger = self.session.finalize(self.root / "bundle", {"files": [], "repository_local_import_closure": []} if manifest is None else manifest)
        return result, ledger


with tempfile.TemporaryDirectory(prefix="task28r-") as root:
    root = Path(root); (root / "bundle").mkdir()
    fixture = Fixture(root)
    try:
        result, ledger = fixture.approve_finalize()
        if result != task28r.CLASSIFICATION or not ledger["fd_path_identity_and_sha_stable"]:
            raise RuntimeError("Task28R positive ledger mismatch")
    finally:
        fixture.finish()

def case(mutator, process_role="closure-audit-entrypoint"):
    with tempfile.TemporaryDirectory(prefix="task28r-neg-") as root:
        root = Path(root); (root / "bundle").mkdir()
        fixture = Fixture(root, process_role=process_role)
        try:
            mutator(fixture)
        finally:
            fixture.finish()

reject("wrong SHA", lambda: case(lambda f: (f.session.expected.__setitem__("sha256", "0" * 64), f.session.approve(str(f.probe)))))
reject("wrong Git blob", lambda: case(lambda f: (f.session.expected.__setitem__("git_blob", "0" * 40), f.session.approve(str(f.probe)))))

def different_inode(f):
    other = f.root / "other.py"; other.write_bytes(f.probe.read_bytes())
    f.session.expected["raw_path"] = str(other); f.session.expected["resolved_path"] = str(other.resolve())
    f.module.__file__ = str(other)
    f.module.__loader__ = importlib.machinery.SourceFileLoader(f.expected["module_key"], str(other))
    sys.argv = [str(other)]
    f.session.approve(str(other))
reject("same bytes different inode", lambda: case(different_inode))

def symlink_file(f):
    target = f.root / "target.py"; target.write_bytes(f.probe.read_bytes())
    f.probe.unlink(); f.probe.symlink_to(target)
    f.session.approve(str(f.probe))
reject("symlink final component", lambda: case(symlink_file))

def wrong_module(f):
    f.session.expected["module_key"] = "wrong"; f.session.approve(str(f.probe))
reject("module key", lambda: case(wrong_module))

def wrong_loader(f):
    f.module.__loader__ = object(); f.session.approve(str(f.probe))
reject("loader", lambda: case(wrong_loader))

def wrong_spec(f):
    f.module.__spec__ = object(); f.session.approve(str(f.probe))
reject("spec", lambda: case(wrong_spec))

def wrong_package(f):
    f.module.__package__ = "bundle"; f.session.approve(str(f.probe))
reject("package", lambda: case(wrong_package))
reject("trainer masquerade", lambda: case(lambda f: f.session.approve(str(f.probe)), process_role="formal-scientific-trainer"))

def other_file(f):
    other = f.root / "not_the_probe.py"; other.write_bytes(f.probe.read_bytes())
    f.session.approve(str(other))
reject("same directory other file", lambda: case(other_file))

def arbitrary_origin(f):
    alias = f.root / "alias.py"; os.link(f.probe, alias)
    f.session.approve(str(alias))
reject("unapproved storage spelling", lambda: case(arbitrary_origin))

def replace_after_open(f):
    def replace(raw, resolved, descriptor):
        replacement = f.root / "replacement"; replacement.write_bytes(f.probe.read_bytes())
        os.replace(replacement, f.probe)
    f.session.after_open_hook = replace
    f.session.approve(str(f.probe)); f.session.finalize(f.root / "bundle", {"files": [], "repository_local_import_closure": []})
reject("pre/post replacement", lambda: case(replace_after_open))

def science_leak(f):
    task28r.SCIENCE_LAUNCHER_SHA256 = sha(("python trainer.py " + f.expected["basename"] + "\n").encode())
    f.launcher.write_text("python trainer.py " + f.expected["basename"] + "\n")
    f.session.approve(str(f.probe)); f.session.finalize(f.root / "bundle", {"files": [], "repository_local_import_closure": []})
reject("probe leak to formal science", lambda: case(science_leak))

git_ledger = json.loads((HERE / "probe_git_identity.json").read_text())
repo = HERE.parents[1]
if (repo / ".git").exists() or subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-dir"], capture_output=True).returncode == 0:
    path = git_ledger["repository_relative_path"]
    commit = git_ledger["frozen_git_commit"]
    tree = subprocess.check_output(["git", "-C", str(repo), "ls-tree", commit, path], text=True).strip().split()
    data = subprocess.check_output(["git", "-C", str(repo), "show", commit + ":" + path])
    if tree[:3] != [git_ledger["git_mode"], "blob", git_ledger["git_blob"]] or sha(data) != git_ledger["sha256"] or len(data) != git_ledger["size"]:
        raise RuntimeError("Task28R frozen Git identity ledger mismatch")

print("TASK28R_EXACT_PROBE_ALIAS_POSITIVE_NEGATIVE_PASS")
