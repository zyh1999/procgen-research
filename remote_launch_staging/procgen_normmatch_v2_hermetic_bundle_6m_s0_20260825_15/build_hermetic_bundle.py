#!/usr/bin/env python3
"""Build a deterministic source bundle exclusively from frozen Git blobs."""
import ast
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

SOURCE_COMMIT = "c2470eac175ee7a05904a73fa2d93bac1f643cf7"
TASK = "PROCGEN-NORMMATCH-V2-HERMETIC-BUNDLE-AND-6M-S0-20260825-15"
METHOD = "PAPER_MATCHED_HYBRID_HEAD_DETGGN_PAPERNORM_V2"
TASK14 = "remote_launch_staging/procgen_paper_hybrid_head_normmatch_detggn_6m_s0_20260825_14"
V1 = "remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08"
PAPER_SOURCE = "work/procgen_paper_2b5affd_6m_bede_20260722/source"

SPECS = [
    ("code/train_shared_paper_hybrid_head_detggn_papernorm_v2.py", f"{TASK14}/train_shared_paper_hybrid_head_detggn_papernorm_v2.py"),
    ("code/configs/adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml", f"{TASK14}/adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml"),
    ("frozen/gpuh_preflight_normmatch_v2.py", f"{TASK14}/gpuh_preflight_normmatch_v2.py"),
    ("frozen/test_hybrid_head_normmatch_v2.py", f"{TASK14}/test_hybrid_head_normmatch_v2.py"),
    ("frozen/audit_normmatch_v2.py", f"{TASK14}/audit_normmatch_v2.py"),
    ("frozen/stage_monitor.py", f"{TASK14}/stage_monitor.py"),
    ("frozen/normmatch_v2_6m_gpuh.sbatch", f"{TASK14}/normmatch_v2_6m_gpuh.sbatch"),
    ("frozen/normmatch_v2_preflight_gpuh.sbatch", f"{TASK14}/normmatch_v2_preflight_gpuh.sbatch"),
    ("frozen/SCIENTIFIC_SHA256SUMS", f"{TASK14}/SCIENTIFIC_SHA256SUMS"),
    ("frozen/reference_hybrid_v1.py", f"{V1}/train_shared_paper_hybrid_head_detggn_v1.py"),
]

LOCAL_CLOSURE = [
    "utils/logger.py", "utils/runners.py", "utils/utils.py",
    "utils/vision_transformers.py", "utils/vit.py", "utils/resnet.py",
    "utils/convnet.py", "utils/popart.py", "utils/running_mean_std.py",
    "utils/transformer.py", "utils/rope.py", "utils/seq_running_mean_std.py",
    "utils/monitor.py", "vec_env/__init__.py", "vec_env/vec_env.py",
    "vec_env/shmem_vec_env.py", "vec_env/subproc_vec_env.py",
    "vec_env/vec_monitor.py", "vec_env/vec_normalize.py",
    "vec_env/vec_remove_dict_obs.py", "vec_env/dummy_vec_env.py",
    "vec_env/util.py",
]
SPECS.extend((f"code/{path}", f"{PAPER_SOURCE}/{path}") for path in LOCAL_CLOSURE)

EXPECTED = {
    "code/train_shared_paper_hybrid_head_detggn_papernorm_v2.py": "0e2c2e26a3ec388cb9df626b4bdae83bff5409a9bbb1febd5c6e2c23a9ddc46b",
    "code/configs/adv_resnet_shared_paper_hybrid_head_detggn_papernorm_v2_6m.yaml": "9497be42db0bac8abb504721677ca6608d9f698f101587980c1a726c1dd81fda",
    "frozen/gpuh_preflight_normmatch_v2.py": "b3dd8b496c478c2289091fb1147b0b0f9256d2fcea669770caa67fded4696afc",
    "frozen/test_hybrid_head_normmatch_v2.py": "f7125681770213974a92d7664250810c201968dc34bb06ce3c365eb4fa59e23c",
    "frozen/audit_normmatch_v2.py": "6dcd75bba65e1575e937afc23cc0e122c194c84187acf3f75b2fc60c23cd98c2",
    "frozen/stage_monitor.py": "536b87201191f81a44fc3aa6564565653572df523080b0952b11d6347152572e",
    "frozen/normmatch_v2_6m_gpuh.sbatch": "85e12886ce5cf81fd98647aa5163319a50174a39210cbeea1ccfde015aaf9d19",
    "frozen/normmatch_v2_preflight_gpuh.sbatch": "ee2634e386b8422d5adbae6d782d80e8620195d05519625be745706a9f901caa",
    "frozen/reference_hybrid_v1.py": "7bcf9bb6f25a6e40206bb2b08404423992ecdf088cf0f64f806c4a8e7a521e54",
}


def git_bytes(path):
    return subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{path}"])


def git_blob(path):
    return subprocess.check_output(
        ["git", "rev-parse", f"{SOURCE_COMMIT}:{path}"], text=True
    ).strip()


def module_name(bundle_path):
    path = Path(bundle_path)
    if path.parts[0] != "code" or path.suffix != ".py":
        return None
    rel = path.relative_to("code")
    if rel.name == "__init__.py":
        return ".".join(rel.parent.parts)
    return ".".join(rel.with_suffix("").parts)


def raw_imports(data):
    tree = ast.parse(data.decode("utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append("." * node.level + (node.module or ""))
    return sorted(set(found))


def local_edges(bundle_path, imports, module_to_path):
    current = module_name(bundle_path)
    if current is None:
        return []
    is_package = Path(bundle_path).name == "__init__.py"
    package = current if is_package else current.rpartition(".")[0]
    edges = set()
    for imported in imports:
        if imported.startswith("."):
            try:
                candidate = importlib.util.resolve_name(imported, package)
            except (ImportError, ValueError):
                continue
        else:
            candidate = imported
        while candidate:
            if candidate in module_to_path:
                edges.add(module_to_path[candidate])
                break
            candidate = candidate.rpartition(".")[0]
    return sorted(edges)


def main():
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    for bundle_path, source_path in SPECS:
        if bundle_path in payload:
            raise RuntimeError(f"duplicate bundle path: {bundle_path}")
        payload[bundle_path] = (source_path, git_bytes(source_path))

    module_to_path = {
        name: path for path in payload if (name := module_name(path)) is not None
    }
    entries = []
    for bundle_path in sorted(payload):
        source_path, data = payload[bundle_path]
        sha = hashlib.sha256(data).hexdigest()
        if bundle_path in EXPECTED and sha != EXPECTED[bundle_path]:
            raise RuntimeError(f"frozen SHA mismatch for {bundle_path}: {sha}")
        imports = raw_imports(data) if bundle_path.endswith(".py") else []
        entries.append({
            "bundle_path": bundle_path,
            "source_repository_path": source_path,
            "source_commit": SOURCE_COMMIT,
            "git_blob_sha1": git_blob(source_path),
            "sha256": sha,
            "size": len(data),
            "imports": imports,
            "repository_local_dependencies": local_edges(bundle_path, imports, module_to_path),
        })

    by_path = {entry["bundle_path"]: entry for entry in entries}
    root = "code/train_shared_paper_hybrid_head_detggn_papernorm_v2.py"
    closure, stack = set(), [root]
    while stack:
        path = stack.pop()
        if path in closure:
            continue
        closure.add(path)
        stack.extend(by_path[path]["repository_local_dependencies"])
    expected_closure = {root, *(f"code/{path}" for path in LOCAL_CLOSURE)}
    missing = expected_closure - closure
    if missing:
        raise RuntimeError(f"declared local closure not reachable: {sorted(missing)}")

    manifest = {
        "format_version": 1,
        "task_id": TASK,
        "method": METHOD,
        "source_commit": SOURCE_COMMIT,
        "source_policy": "bytes read only from named frozen Git blobs",
        "entrypoint": root,
        "repository_local_import_closure": sorted(closure),
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    temp_tar = output_dir / ".bundle.tmp.tar"
    with tarfile.open(temp_tar, "w", format=tarfile.PAX_FORMAT) as archive:
        for path, data in sorted((p, d) for p, (_, d) in payload.items()):
            info = tarfile.TarInfo(path)
            info.size = len(data)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
        info = tarfile.TarInfo("BUNDLE_MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        archive.addfile(info, io.BytesIO(manifest_bytes))
    archive_sha = hashlib.sha256(temp_tar.read_bytes()).hexdigest()
    archive_name = f"normmatch_v2_source_{archive_sha}.tar"
    archive_path = output_dir / archive_name
    temp_tar.replace(archive_path)
    manifest_path = output_dir / "BUNDLE_MANIFEST.json"
    manifest_path.write_bytes(manifest_bytes)
    (output_dir / "BUNDLE_SHA256SUMS").write_text(
        f"{archive_sha}  {archive_name}\n{manifest_sha}  BUNDLE_MANIFEST.json\n"
    )
    print(f"BUNDLE_BUILD_PASS archive={archive_path}")
    print(f"bundle_sha256={archive_sha}")
    print(f"manifest_sha256={manifest_sha}")
    print(f"files={len(entries)} closure={len(closure)}")


if __name__ == "__main__":
    main()
