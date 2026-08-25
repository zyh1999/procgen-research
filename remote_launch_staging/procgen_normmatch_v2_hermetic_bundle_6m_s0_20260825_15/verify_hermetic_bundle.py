#!/usr/bin/env python3
"""Verify and safely extract the content-addressed hermetic source bundle."""
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    archive_path = Path(sys.argv[1]).resolve()
    expected_archive_sha = sys.argv[2]
    destination = Path(sys.argv[3]).resolve()
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
        manifest_bytes = archive.extractfile("BUNDLE_MANIFEST.json").read()
        manifest = json.loads(manifest_bytes)
        entries = {entry["bundle_path"]: entry for entry in manifest["files"]}
        if set(names) != {*entries, "BUNDLE_MANIFEST.json"}:
            raise RuntimeError("archive/manifest file-set mismatch")
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError("destination must be absent or empty")
        destination.mkdir(parents=True, exist_ok=True)
        for name, entry in entries.items():
            data = archive.extractfile(name).read()
            if len(data) != entry["size"] or sha(data) != entry["sha256"]:
                raise RuntimeError(f"manifest mismatch: {name}")
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            os.chmod(target, 0o644)
        (destination / "BUNDLE_MANIFEST.json").write_bytes(manifest_bytes)
    for entry in manifest["files"]:
        if sha((destination / entry["bundle_path"]).read_bytes()) != entry["sha256"]:
            raise RuntimeError(f"post-extract mismatch: {entry['bundle_path']}")
    print("HERMETIC_BUNDLE_VERIFY_PASS")
    print(f"bundle_sha256={expected_archive_sha}")
    print(f"manifest_sha256={sha(manifest_bytes)}")
    print(f"files={len(entries)} destination={destination}")


if __name__ == "__main__":
    main()
