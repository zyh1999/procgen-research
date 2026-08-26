#!/usr/bin/env python3
"""Verify and safely extract the Task35R content-addressed source bundle."""

import hashlib
import json
import os
import stat
import sys
import tarfile
from pathlib import Path, PurePosixPath


EXPECTED_SOURCE_COMMIT = "55984df39bf883685583f22894edd5eb615f95ea"
EXPECTED_TASK = "PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-HERMETIC-PREFLIGHT-20260826-35R"
EXPECTED_METHOD = "DET_STANDARD_MSE_GGN_HEAD_CVLM_V1"
REQUIRED = {
    "code/train_shared_det_standard_mse_ggn_head_cvlm_v1.py",
    "code/configs/adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml",
    "code/utils/logger.py",
    "code/utils/runners.py",
    "code/utils/utils.py",
    "code/vec_env/__init__.py",
    "frozen/gpuh_preflight.py",
    "frozen/audit_task34r.py",
    "frozen/standard_mse_ggn_head_cvlm_6m_gpuh.sbatch",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def main():
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: verify ARCHIVE EXPECTED_ARCHIVE_SHA EXPECTED_MANIFEST_SHA DESTINATION"
        )
    archive_path = Path(sys.argv[1]).resolve()
    expected_archive_sha = sys.argv[2]
    expected_manifest_sha = sys.argv[3]
    destination = Path(sys.argv[4]).resolve()
    archive_bytes = archive_path.read_bytes()
    if sha(archive_bytes) != expected_archive_sha:
        raise RuntimeError("bundle archive SHA256 mismatch")
    with tarfile.open(archive_path, "r") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate archive member")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise RuntimeError(f"unsafe archive member: {member.name}")
        manifest_member = archive.extractfile("BUNDLE_MANIFEST.json")
        if manifest_member is None:
            raise RuntimeError("bundle manifest missing")
        manifest_bytes = manifest_member.read()
        if sha(manifest_bytes) != expected_manifest_sha:
            raise RuntimeError("bundle manifest SHA256 mismatch")
        manifest = json.loads(manifest_bytes)
        if manifest["task_id"] != EXPECTED_TASK:
            raise RuntimeError("wrong task identity")
        if manifest["method"] != EXPECTED_METHOD:
            raise RuntimeError("wrong method identity")
        if manifest["source_commit"] != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError("wrong frozen source commit")
        entries = {entry["bundle_path"]: entry for entry in manifest["files"]}
        if len(entries) != len(manifest["files"]):
            raise RuntimeError("duplicate manifest path")
        if set(names) != {*entries, "BUNDLE_MANIFEST.json"}:
            raise RuntimeError("archive/manifest file-set mismatch")
        if not REQUIRED.issubset(entries):
            raise RuntimeError(f"required bundle entries missing: {sorted(REQUIRED - entries.keys())}")
        closure = set(manifest["repository_local_import_closure"])
        if not {
            "code/utils/logger.py",
            "code/utils/runners.py",
            "code/utils/utils.py",
            "code/vec_env/__init__.py",
        }.issubset(closure):
            raise RuntimeError("required local import closure missing")
        for name, entry in entries.items():
            data = archive.extractfile(name).read()
            member = archive.getmember(name)
            expected_mode = int(entry["mode"][-3:], 8)
            if len(data) != entry["size"] or sha(data) != entry["sha256"]:
                raise RuntimeError(f"manifest content mismatch: {name}")
            if stat.S_IMODE(member.mode) != expected_mode:
                raise RuntimeError(f"manifest mode mismatch: {name}")
            if not entry["git_blob"] or not entry["repository_path"]:
                raise RuntimeError(f"Git provenance incomplete: {name}")
            if git_blob(data) != entry["git_blob"]:
                raise RuntimeError(f"Git blob/content mismatch: {name}")
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError("destination must be absent or empty")
        destination.mkdir(parents=True, exist_ok=True)
        for name, entry in entries.items():
            data = archive.extractfile(name).read()
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            os.chmod(target, int(entry["mode"][-3:], 8))
        (destination / "BUNDLE_MANIFEST.json").write_bytes(manifest_bytes)

    for entry in manifest["files"]:
        target = destination / entry["bundle_path"]
        actual = os.lstat(target)
        if not stat.S_ISREG(actual.st_mode) or stat.S_ISLNK(actual.st_mode):
            raise RuntimeError(f"post-extract non-regular file: {entry['bundle_path']}")
        if sha(target.read_bytes()) != entry["sha256"]:
            raise RuntimeError(f"post-extract mismatch: {entry['bundle_path']}")
        if stat.S_IMODE(actual.st_mode) != int(entry["mode"][-3:], 8):
            raise RuntimeError(f"post-extract mode mismatch: {entry['bundle_path']}")
    print("TASK35R_HERMETIC_BUNDLE_VERIFY_PASS")
    print(f"bundle_sha256={expected_archive_sha}")
    print(f"manifest_sha256={expected_manifest_sha}")
    print(f"files={len(entries)} destination={destination}")


if __name__ == "__main__":
    main()
