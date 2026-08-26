#!/usr/bin/env python3
"""Local deterministic and negative gates for Task35R deployment recovery."""

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_hermetic_bundle_task35r.py"
VERIFIER = HERE / "verify_hermetic_bundle_task35r.py"
SMOKE = HERE / "hermetic_import_smoke_task35r.py"
OLD = HERE.parent / "procgen_standard_mse_ggn_head_cvlm_6m_s0_20260825_34r"
TRAINER_SHA = "ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc"
CONFIG_SHA = "52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs)


def archive_payload(path):
    with tarfile.open(path, "r") as archive:
        return [(member, archive.extractfile(member).read()) for member in archive.getmembers()]


def write_archive(path, records):
    with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for original, data in records:
            info = tarfile.TarInfo(original.name)
            info.size = len(data)
            info.mode = original.mode
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))


def expect_reject(command, fragment, env=None):
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if result.returncode == 0:
        raise AssertionError(f"negative gate unexpectedly passed: {command}")
    combined = result.stdout + result.stderr
    if fragment not in combined:
        raise AssertionError(f"negative gate missing {fragment!r}: {combined}")


def main():
    for path in (
        BUILDER,
        VERIFIER,
        SMOKE,
        HERE / "audit_launcher_equivalence_task35r.py",
        HERE / "standard_mse_ggn_head_cvlm_hermetic_preflight_gpuh.sbatch",
    ):
        compile(path.read_text(), str(path), "exec") if path.suffix == ".py" else None
    if sha((OLD / "train_shared_det_standard_mse_ggn_head_cvlm_v1.py").read_bytes()) != TRAINER_SHA:
        raise AssertionError("frozen Task34R trainer changed")
    if sha((OLD / "adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml").read_bytes()) != CONFIG_SHA:
        raise AssertionError("frozen Task34R config changed")

    with tempfile.TemporaryDirectory(prefix="task35r_local_gate_") as temporary:
        root = Path(temporary)
        first, second = root / "first", root / "second"
        run(sys.executable, str(BUILDER), str(first))
        run(sys.executable, str(BUILDER), str(second))
        first_manifest = (first / "BUNDLE_MANIFEST.json").read_bytes()
        second_manifest = (second / "BUNDLE_MANIFEST.json").read_bytes()
        if first_manifest != second_manifest:
            raise AssertionError("independent manifests differ")
        first_archive = next(first.glob("task35r_source_*.tar"))
        second_archive = next(second.glob("task35r_source_*.tar"))
        if first_archive.read_bytes() != second_archive.read_bytes():
            raise AssertionError("independent archives differ")
        archive_sha = sha(first_archive.read_bytes())
        manifest_sha = sha(first_manifest)
        extracted = root / "extracted"
        run(
            sys.executable,
            str(VERIFIER),
            str(first_archive),
            archive_sha,
            manifest_sha,
            str(extracted),
        )
        manifest = json.loads(first_manifest)
        for entry in manifest["files"]:
            for field in ("repository_path", "git_blob", "sha256", "size", "mode"):
                if field not in entry:
                    raise AssertionError(f"manifest field absent: {field}")

        records = archive_payload(first_archive)
        missing_archive = root / "missing_utils.tar"
        write_archive(
            missing_archive,
            [(member, data) for member, data in records if member.name != "code/utils/logger.py"],
        )
        expect_reject(
            [sys.executable, str(VERIFIER), str(missing_archive), sha(missing_archive.read_bytes()), manifest_sha, str(root / "missing_out")],
            "archive/manifest file-set mismatch",
        )

        wrong_hash_archive = root / "wrong_hash.tar"
        write_archive(
            wrong_hash_archive,
            [(member, data + b"\n# tamper\n" if member.name == "code/utils/logger.py" else data) for member, data in records],
        )
        expect_reject(
            [sys.executable, str(VERIFIER), str(wrong_hash_archive), sha(wrong_hash_archive.read_bytes()), manifest_sha, str(root / "wrong_hash_out")],
            "manifest content mismatch",
        )

        bad_blob_manifest = json.loads(first_manifest)
        for entry in bad_blob_manifest["files"]:
            if entry["bundle_path"] == "code/utils/logger.py":
                entry["git_blob"] = "0" * 40
        bad_blob_bytes = (json.dumps(bad_blob_manifest, indent=2, sort_keys=True) + "\n").encode()
        bad_blob_archive = root / "wrong_blob.tar"
        write_archive(
            bad_blob_archive,
            [(member, bad_blob_bytes if member.name == "BUNDLE_MANIFEST.json" else data) for member, data in records],
        )
        expect_reject(
            [sys.executable, str(VERIFIER), str(bad_blob_archive), sha(bad_blob_archive.read_bytes()), sha(bad_blob_bytes), str(root / "wrong_blob_out")],
            "Git blob/content mismatch",
        )

        ambient = root / "ambient"
        (ambient / "utils").mkdir(parents=True)
        (ambient / "utils/logger.py").write_text("AMBIENT = True\n")
        empty_cwd = root / "empty_cwd"
        empty_cwd.mkdir()
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ambient)
        expect_reject(
            [sys.executable, str(SMOKE), str(extracted), str(root / "ambient_origins.json")],
            "ambient repository-local candidate paths forbidden",
            env=environment,
        )

    print("TASK35R_LOCAL_HERMETIC_TESTS_PASS")
    print(f"bundle_sha256={archive_sha}")
    print(f"manifest_sha256={manifest_sha}")
    print("negative_missing_utils=REJECT")
    print("negative_wrong_hash=REJECT")
    print("negative_different_git_blob=REJECT")
    print("negative_ambient_path_fallback=REJECT")
    evidence = HERE / "evidence_local/hermetic_tests.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({
        "result": "TASK35R_LOCAL_HERMETIC_TESTS_PASS",
        "bundle_sha256": archive_sha,
        "manifest_sha256": manifest_sha,
        "independent_builds_byte_identical": True,
        "manifest_fields": ["repository_path", "git_blob", "sha256", "size", "mode"],
        "negative_missing_utils": "REJECT",
        "negative_wrong_hash": "REJECT",
        "negative_different_git_blob": "REJECT",
        "negative_ambient_path_fallback": "REJECT",
        "frozen_trainer_sha256": TRAINER_SHA,
        "frozen_config_sha256": CONFIG_SHA,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
