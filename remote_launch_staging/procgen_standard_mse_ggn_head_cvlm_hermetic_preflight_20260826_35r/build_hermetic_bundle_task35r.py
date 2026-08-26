#!/usr/bin/env python3
"""Build Task35R's deterministic source bundle from frozen Git objects only."""

import ast
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path


SOURCE_COMMIT = "55984df39bf883685583f22894edd5eb615f95ea"
TASK_ID = "PROCGEN-STANDARD-MSE-GGN-HEAD-CVLM-HERMETIC-PREFLIGHT-20260826-35R"
METHOD = "DET_STANDARD_MSE_GGN_HEAD_CVLM_V1"
TASK34R = "remote_launch_staging/procgen_standard_mse_ggn_head_cvlm_6m_s0_20260825_34r"
PAPER_SOURCE = "work/procgen_paper_2b5affd_6m_bede_20260722/source"

LOCAL_CLOSURE = [
    "utils/logger.py",
    "utils/runners.py",
    "utils/utils.py",
    "utils/vision_transformers.py",
    "utils/vit.py",
    "utils/resnet.py",
    "utils/convnet.py",
    "utils/popart.py",
    "utils/running_mean_std.py",
    "utils/transformer.py",
    "utils/rope.py",
    "utils/seq_running_mean_std.py",
    "utils/monitor.py",
    "vec_env/__init__.py",
    "vec_env/vec_env.py",
    "vec_env/shmem_vec_env.py",
    "vec_env/subproc_vec_env.py",
    "vec_env/vec_monitor.py",
    "vec_env/vec_normalize.py",
    "vec_env/vec_remove_dict_obs.py",
    "vec_env/dummy_vec_env.py",
    "vec_env/util.py",
]

SPECS = [
    (
        "code/train_shared_det_standard_mse_ggn_head_cvlm_v1.py",
        f"{TASK34R}/train_shared_det_standard_mse_ggn_head_cvlm_v1.py",
    ),
    (
        "code/configs/adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml",
        f"{TASK34R}/adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml",
    ),
    ("frozen/gpuh_preflight.py", f"{TASK34R}/gpuh_preflight.py"),
    ("frozen/audit_task34r.py", f"{TASK34R}/audit_task34r.py"),
    (
        "frozen/standard_mse_ggn_head_cvlm_6m_gpuh.sbatch",
        f"{TASK34R}/standard_mse_ggn_head_cvlm_6m_gpuh.sbatch",
    ),
    (
        "audit_sources/task07/train_shared_paper_separateb_detggn_v1.py",
        "remote_launch_staging/procgen_paper_separateb_detggn_6m_s0_20260824_07/"
        "train_shared_paper_separateb_detggn_v1.py",
    ),
    (
        "audit_sources/task13/train_shared_paper_hybrid_head_detggn_v1.py",
        "remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/"
        "train_shared_paper_hybrid_head_detggn_v1.py",
    ),
    (
        "audit_sources/task13/adv_resnet_shared_paper_hybrid_head_detggn_v1_6m.yaml",
        "remote_launch_staging/procgen_paper_hybrid_head_detggn_6m_s0_20260824_08/"
        "adv_resnet_shared_paper_hybrid_head_detggn_v1_6m.yaml",
    ),
    (
        "audit_sources/task32/train_shared_det_actor_weighted_gae_ggn_head_v1.py",
        "remote_launch_staging/procgen_actor_weighted_gae_ggn_head_6m_s0_20260825_32/"
        "train_shared_det_actor_weighted_gae_ggn_head_v1.py",
    ),
]
SPECS.extend((f"code/{path}", f"{PAPER_SOURCE}/{path}") for path in LOCAL_CLOSURE)

EXPECTED_SHA256 = {
    "code/train_shared_det_standard_mse_ggn_head_cvlm_v1.py":
        "ca53efd549ae58738c5c215c84dfe5f342c0f801d3eb0f7fb61a2507f31e69fc",
    "code/configs/adv_resnet_shared_det_standard_mse_ggn_head_cvlm_v1_6m.yaml":
        "52c133559825947cd233184f2468c4aa715c6c419274bb78f7f715d542718132",
    "frozen/gpuh_preflight.py":
        "2baac759c26c4fb663ac24ff68a6de8b640bb9bd0c443ec8f65106de3d36759a",
    "frozen/audit_task34r.py":
        "9d4929685bd8368f861770f155c5fcc0b7f86b91e9cfc92481987b0dc8ec2723",
    "frozen/standard_mse_ggn_head_cvlm_6m_gpuh.sbatch":
        "6dffce265e55f87fa5be848ec5bd9940fcecb6203f621a1454fb4f6a9c74caca",
}


def git_bytes(repository_path):
    return subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{repository_path}"])


def git_blob(repository_path):
    return subprocess.check_output(
        ["git", "rev-parse", f"{SOURCE_COMMIT}:{repository_path}"], text=True
    ).strip()


def git_mode(repository_path):
    output = subprocess.check_output(
        ["git", "ls-tree", SOURCE_COMMIT, "--", repository_path], text=True
    ).strip()
    fields = output.split(None, 3)
    if len(fields) != 4 or fields[1] != "blob" or fields[3] != repository_path:
        raise RuntimeError(f"not one frozen Git blob: {repository_path}: {output!r}")
    return fields[0]


def module_name(bundle_path):
    path = Path(bundle_path)
    if path.parts[0] != "code" or path.suffix != ".py":
        return None
    relative = path.relative_to("code")
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts)
    return ".".join(relative.with_suffix("").parts)


def raw_imports(data):
    tree = ast.parse(data.decode("utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append("." * node.level + (node.module or ""))
    return sorted(set(imports))


def local_edges(bundle_path, imports, module_to_path):
    current = module_name(bundle_path)
    if current is None:
        return []
    package = current if Path(bundle_path).name == "__init__.py" else current.rpartition(".")[0]
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
    source_records = {}
    for bundle_path, repository_path in SPECS:
        if bundle_path in payload:
            raise RuntimeError(f"duplicate bundle path: {bundle_path}")
        data = git_bytes(repository_path)
        payload[bundle_path] = data
        source_records[bundle_path] = {
            "repository_path": repository_path,
            "git_blob": git_blob(repository_path),
            "git_mode": git_mode(repository_path),
        }

    module_to_path = {
        name: path for path in payload if (name := module_name(path)) is not None
    }
    entries = []
    for bundle_path in sorted(payload):
        data = payload[bundle_path]
        digest = hashlib.sha256(data).hexdigest()
        expected = EXPECTED_SHA256.get(bundle_path)
        if expected is not None and digest != expected:
            raise RuntimeError(f"frozen SHA mismatch: {bundle_path}: {digest} != {expected}")
        imports = raw_imports(data) if bundle_path.endswith(".py") else []
        source = source_records[bundle_path]
        entries.append({
            "bundle_path": bundle_path,
            "repository_path": source["repository_path"],
            "source_commit": SOURCE_COMMIT,
            "git_blob": source["git_blob"],
            "mode": source["git_mode"],
            "sha256": digest,
            "size": len(data),
            "imports": imports,
            "repository_local_dependencies": local_edges(
                bundle_path, imports, module_to_path
            ),
        })

    by_path = {entry["bundle_path"]: entry for entry in entries}
    entrypoint = "code/train_shared_det_standard_mse_ggn_head_cvlm_v1.py"
    closure, stack = set(), [entrypoint]
    while stack:
        path = stack.pop()
        if path in closure:
            continue
        closure.add(path)
        stack.extend(by_path[path]["repository_local_dependencies"])
    declared_local = {entrypoint, *(f"code/{path}" for path in LOCAL_CLOSURE)}
    missing = closure - declared_local
    if missing:
        raise RuntimeError(f"computed local closure absent from declaration: {sorted(missing)}")
    required = {
        entrypoint,
        "code/utils/logger.py",
        "code/utils/runners.py",
        "code/utils/utils.py",
        "code/vec_env/__init__.py",
    }
    if not required.issubset(closure):
        raise RuntimeError(f"required reachable local modules missing: {sorted(required - closure)}")

    manifest = {
        "format_version": 1,
        "task_id": TASK_ID,
        "method": METHOD,
        "source_commit": SOURCE_COMMIT,
        "source_policy": "all payload bytes read from named frozen Git blobs",
        "repository_import_root": "code",
        "entrypoint": entrypoint,
        "repository_local_import_closure": sorted(closure),
        "included_repository_local_files": sorted(declared_local),
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    temporary = output_dir / ".task35r_bundle.tmp.tar"
    with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
        for path in sorted(payload):
            data = payload[path]
            mode = int(source_records[path]["git_mode"][-3:], 8)
            info = tarfile.TarInfo(path)
            info.size = len(data)
            info.mode = mode
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
    archive_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
    archive_name = f"task35r_source_{archive_sha}.tar"
    archive_path = output_dir / archive_name
    temporary.replace(archive_path)
    (output_dir / "BUNDLE_MANIFEST.json").write_bytes(manifest_bytes)
    (output_dir / "BUNDLE_SHA256SUMS").write_text(
        f"{archive_sha}  {archive_name}\n{manifest_sha}  BUNDLE_MANIFEST.json\n"
    )
    print(f"TASK35R_HERMETIC_BUNDLE_BUILD_PASS archive={archive_path}")
    print(f"bundle_sha256={archive_sha}")
    print(f"manifest_sha256={manifest_sha}")
    print(f"files={len(entries)} closure={len(closure)}")


if __name__ == "__main__":
    main()
