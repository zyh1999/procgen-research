#!/usr/bin/env python3
"""Run the frozen Task34R audit with manifest-resolved bundle paths.

The frozen audit bytes are never changed on disk.  This adapter validates the
trainer/config identities from the immutable Task35R manifest, replaces only
the two stale path expressions in the in-memory AST, executes the audit, and
then revalidates the files.
"""

import ast
import hashlib
import json
import os
import stat
import sys
from pathlib import Path


EXPECTED_MANIFEST_SHA = "287a744078b10054d107974125bac6b5fac43fd944b6200ec54720cd2695c9af"
EXPECTED_AUDIT_SHA = "9d4929685bd8368f861770f155c5fcc0b7f86b91e9cfc92481987b0dc8ec2723"
TRAINER_REPO_PATH = (
    "remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_6m_s0_20260825_34r/"
    "train_shared_det_standard_mse_ggn_head_cvlm_v1.py"
)
TRAINER_BUNDLE_PATH = "code/train_shared_det_standard_mse_ggn_head_cvlm_v1.py"
TRAINER_SHA = "ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc"
TRAINER_BLOB = "f480eb3b509fb693a45d3264ae44aa383cc5f2e6"
TRAINER_SIZE = 74577
TRAINER_MODE = "100755"
CONFIG_REPO_PATH = (
    "remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_6m_s0_20260825_34r/"
    "adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml"
)
CONFIG_BUNDLE_PATH = "code/configs/adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml"
CONFIG_SHA = "52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132"
CONFIG_BLOB = "d561f9390836b166f955ca33f2787d33f0f3a474"
CONFIG_SIZE = 926
CONFIG_MODE = "100644"
AUDIT_BUNDLE_PATH = "frozen/audit_task34r.py"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def git_blob_bytes(data):
    return hashlib.sha1(("blob %d\0" % len(data)).encode() + data).hexdigest()


def _inside(path, directory):
    return os.path.commonpath((str(path), str(directory))) == str(directory)


def file_identity(path):
    raw = path.read_bytes()
    info = os.lstat(str(path))
    return {
        "path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "gid": info.st_gid,
        "mode": format(stat.S_IMODE(info.st_mode), "04o"),
        "size": info.st_size,
        "sha256": sha256_bytes(raw),
        "git_blob": git_blob_bytes(raw),
        "regular": stat.S_ISREG(info.st_mode),
        "symlink": stat.S_ISLNK(info.st_mode),
    }


def resolve_manifest_member(
    bundle_root,
    manifest,
    repository_path,
    bundle_path,
    expected_sha,
    expected_blob,
    expected_size,
    expected_mode,
):
    entries = [
        entry for entry in manifest.get("files", [])
        if entry.get("repository_path") == repository_path
    ]
    if len(entries) != 1:
        raise RuntimeError("manifest repository identity missing or duplicated: " + repository_path)
    entry = entries[0]
    expected_entry = {
        "bundle_path": bundle_path,
        "repository_path": repository_path,
        "sha256": expected_sha,
        "git_blob": expected_blob,
        "size": expected_size,
        "mode": expected_mode,
    }
    for key, value in expected_entry.items():
        if entry.get(key) != value:
            raise RuntimeError("manifest identity mismatch %s: %r != %r" % (key, entry.get(key), value))

    lexical_root = bundle_root.absolute()
    lexical_target = lexical_root.joinpath(*Path(bundle_path).parts)
    resolved_root = lexical_root.resolve(strict=True)
    resolved_target = lexical_target.resolve(strict=True)
    if not _inside(resolved_target, resolved_root / "code"):
        raise RuntimeError("manifest target escaped verified bundle/code")

    current = lexical_root
    for component in Path(bundle_path).parts:
        current = current / component
        info = os.lstat(str(current))
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError("symlink component rejected: " + str(current))
    identity = file_identity(lexical_target)
    if not identity["regular"] or identity["symlink"]:
        raise RuntimeError("target is not a regular non-symlink file")
    if identity["sha256"] != expected_sha:
        raise RuntimeError("target SHA256 mismatch")
    if identity["git_blob"] != expected_blob:
        raise RuntimeError("target Git blob mismatch")
    if identity["size"] != expected_size:
        raise RuntimeError("target size mismatch")
    if identity["mode"] != format(int(expected_mode[-3:], 8), "04o"):
        raise RuntimeError("target mode mismatch")
    return lexical_target, entry, identity


def verify_manifest(bundle_root):
    manifest_path = bundle_root / "BUNDLE_MANIFEST.json"
    manifest_bytes = manifest_path.read_bytes()
    if sha256_bytes(manifest_bytes) != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("immutable manifest SHA256 mismatch")
    manifest = json.loads(manifest_bytes)
    repository_paths = [entry.get("repository_path") for entry in manifest.get("files", [])]
    if len(repository_paths) != len(set(repository_paths)):
        raise RuntimeError("duplicate manifest repository path")
    return manifest_path, manifest


def verify_audit_source(audit_path):
    identity = file_identity(audit_path)
    if not identity["regular"] or identity["symlink"]:
        raise RuntimeError("audit is not a regular non-symlink file")
    if identity["sha256"] != EXPECTED_AUDIT_SHA:
        raise RuntimeError("frozen audit numerical logic SHA256 mismatch")
    return identity


class ExplicitPathBinding(ast.NodeTransformer):
    """Replace only stale target path values in the frozen ``paths`` dict."""

    def __init__(self):
        self.replacements = []

    def visit_Assign(self, node):
        self.generic_visit(node)
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "paths"
            and isinstance(node.value, ast.Dict)
        ):
            return node
        for index, key in enumerate(node.value.keys):
            if not isinstance(key, ast.Constant):
                continue
            if key.value == "target_trainer":
                old = ast.dump(node.value.values[index], include_attributes=False)
                expected = ast.dump(
                    ast.parse('root / "train_shared_det_standard_mse_ggn_head_cvlm_v1.py"', mode="eval").body,
                    include_attributes=False,
                )
                if old != expected:
                    raise RuntimeError("unexpected frozen target trainer path expression")
                node.value.values[index] = ast.copy_location(
                    ast.Name(id="_task36_explicit_trainer", ctx=ast.Load()),
                    node.value.values[index],
                )
                self.replacements.append("target_trainer")
            elif key.value == "target_config":
                old = ast.dump(node.value.values[index], include_attributes=False)
                expected = ast.dump(
                    ast.parse('root / "adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml"', mode="eval").body,
                    include_attributes=False,
                )
                if old != expected:
                    raise RuntimeError("unexpected frozen target config path expression")
                node.value.values[index] = ast.copy_location(
                    ast.Name(id="_task36_explicit_config", ctx=ast.Load()),
                    node.value.values[index],
                )
                self.replacements.append("target_config")
        return node


def run_frozen_audit(audit_path, trainer_path, config_path):
    source = audit_path.read_text()
    tree = ast.parse(source, filename=str(audit_path))
    transformer = ExplicitPathBinding()
    transformed = transformer.visit(tree)
    if transformer.replacements != ["target_trainer", "target_config"]:
        raise RuntimeError("path adapter did not make exactly two ordered replacements")
    ast.fix_missing_locations(transformed)
    namespace = {
        "__name__": "__main__",
        "__file__": str(audit_path),
        "_task36_explicit_trainer": trainer_path,
        "_task36_explicit_config": config_path,
    }
    exec(compile(transformed, str(audit_path), "exec"), namespace, namespace)
    return transformer.replacements


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: audit_path_adapter_task36.py BUNDLE_ROOT OUTPUT_LEDGER IDENTITY_LEDGER")
    bundle_root = Path(sys.argv[1]).absolute()
    output_ledger = Path(sys.argv[2]).absolute()
    identity_ledger = Path(sys.argv[3]).absolute()
    manifest_path, manifest = verify_manifest(bundle_root)
    trainer_path, trainer_entry, trainer_before = resolve_manifest_member(
        bundle_root, manifest, TRAINER_REPO_PATH, TRAINER_BUNDLE_PATH,
        TRAINER_SHA, TRAINER_BLOB, TRAINER_SIZE, TRAINER_MODE,
    )
    config_path, config_entry, config_before = resolve_manifest_member(
        bundle_root, manifest, CONFIG_REPO_PATH, CONFIG_BUNDLE_PATH,
        CONFIG_SHA, CONFIG_BLOB, CONFIG_SIZE, CONFIG_MODE,
    )
    audit_path = bundle_root / AUDIT_BUNDLE_PATH
    audit_before = verify_audit_source(audit_path)
    old_output = os.environ.get("TASK34R_AUDIT_OUTPUT")
    os.environ["TASK34R_AUDIT_OUTPUT"] = str(output_ledger)
    try:
        replacements = run_frozen_audit(audit_path, trainer_path, config_path)
    finally:
        if old_output is None:
            os.environ.pop("TASK34R_AUDIT_OUTPUT", None)
        else:
            os.environ["TASK34R_AUDIT_OUTPUT"] = old_output
    trainer_after = file_identity(trainer_path)
    config_after = file_identity(config_path)
    audit_after = verify_audit_source(audit_path)
    if trainer_after != trainer_before or config_after != config_before or audit_after != audit_before:
        raise RuntimeError("frozen input identity changed during audit execution")
    ledger = {
        "task_id": "PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-AUDIT-PATH-RECOVERY-20260826-36",
        "manifest": {"path": str(manifest_path), "sha256": EXPECTED_MANIFEST_SHA},
        "trainer_manifest_entry": trainer_entry,
        "trainer_pre": trainer_before,
        "trainer_post": trainer_after,
        "config_manifest_entry": config_entry,
        "config_pre": config_before,
        "config_post": config_after,
        "audit_pre": audit_before,
        "audit_post": audit_after,
        "in_memory_path_replacements": replacements,
        "audit_math_source_modified": False,
        "ambient_fallback": False,
    }
    identity_ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print("TASK36_AUDIT_PATH_ADAPTER_PASS")
    print("trainer=" + str(trainer_path))
    print("trainer_sha256=" + trainer_after["sha256"])
    print("audit_sha256=" + audit_after["sha256"])


if __name__ == "__main__":
    main()
