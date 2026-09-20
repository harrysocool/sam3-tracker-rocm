#!/usr/bin/env python3
"""Write reproducible identity metadata for a SAM3 ROCm artifact root."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

ARTIFACT_SUBDIRS = (
    "backbone_detector",
    "detector_modules",
    "detr_decoder_fixed",
    "tracker_modules",
)
EXCLUDED_ROOT_FILES = {
    "ARTIFACT_MANIFEST.json",
    "ARTIFACT_MANIFEST.sha256",
    "SHA256SUMS",
}
BUILD_PROVENANCE_FILENAME = "BUILD_PROVENANCE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ec_power_mode() -> tuple[str | None, str]:
    override = os.environ.get("SAM3_EC_POWER_MODE", "").strip().lower()
    if override:
        return override, "SAM3_EC_POWER_MODE"
    path = Path(
        os.environ.get(
            "SAM3_EC_POWER_MODE_PATH",
            "/sys/class/ec_su_axb35/apu/power_mode",
        )
    )
    try:
        return path.read_text(encoding="ascii").strip().lower(), str(path)
    except OSError:
        return None, str(path)


def runtime_metadata() -> dict:
    import migraphx
    import onnxruntime
    import torch

    return {
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "onnxruntime": onnxruntime.__version__,
        "migraphx": getattr(migraphx, "__version__", "unknown"),
        "migraphx_module": str(Path(migraphx.__file__).resolve()),
        "providers": onnxruntime.get_available_providers(),
    }


def source_version() -> str:
    version_path = Path(__file__).resolve().parent.parent / "VERSION"
    return (
        version_path.read_text(encoding="utf-8").strip()
        if version_path.is_file() else "unknown"
    )


def current_build_provenance(
    checkpoint: Path,
    *,
    imgsz: int,
    ptr_tokens: int,
    max_spatial_slots: int,
) -> dict:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source_commit = os.environ.get("SAM3_SOURCE_COMMIT", "").strip()
    source_dirty = os.environ.get("SAM3_SOURCE_DIRTY", "unknown").strip()
    image_id = os.environ.get("SAM3_DOCKER_IMAGE_ID", "").strip()
    image_ref = os.environ.get("SAM3_DOCKER_IMAGE_REF", "").strip()
    if not source_commit:
        raise RuntimeError("SAM3_SOURCE_COMMIT was not supplied by the launcher")
    if source_dirty not in {"0", "1"}:
        raise RuntimeError("SAM3_SOURCE_DIRTY must be 0 or 1")
    if not image_id:
        raise RuntimeError("SAM3_DOCKER_IMAGE_ID was not supplied by the launcher")
    mode, mode_source = read_ec_power_mode()
    return {
        "schema": 1,
        "source": {
            "commit": source_commit,
            "dirty": source_dirty == "1",
            "version": source_version(),
        },
        "checkpoint": {
            "size": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        },
        "build": {
            "image_ref": image_ref,
            "image_id": image_id,
            "imgsz": imgsz,
            "ptr_tokens": ptr_tokens,
            "max_spatial_slots": max_spatial_slots,
            "ec_power_mode": mode,
            "ec_power_mode_source": mode_source,
        },
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _clear_generated_artifacts(root: Path) -> None:
    for name in (*ARTIFACT_SUBDIRS, BUILD_PROVENANCE_FILENAME,
                 *EXCLUDED_ROOT_FILES):
        path = root / name
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def prepare_build_root(
    root: Path,
    checkpoint: Path,
    *,
    imgsz: int,
    ptr_tokens: int,
    max_spatial_slots: int,
    reset_full_build: bool,
) -> dict:
    """Initialize or validate the immutable inputs for a resumable build."""
    root.mkdir(parents=True, exist_ok=True)
    expected = current_build_provenance(
        checkpoint,
        imgsz=imgsz,
        ptr_tokens=ptr_tokens,
        max_spatial_slots=max_spatial_slots,
    )
    provenance_path = root / BUILD_PROVENANCE_FILENAME
    try:
        recorded = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        recorded = None

    populated = any(root.iterdir())
    if reset_full_build:
        _clear_generated_artifacts(root)
        root.mkdir(parents=True, exist_ok=True)
    elif populated and recorded != expected:
        raise RuntimeError(
            "artifact root is non-empty but its build provenance does not "
            f"match the current source/checkpoint/image: {root}. Select a new "
            "directory, or use --force with --steps all for an explicit reset."
        )

    _write_json_atomic(provenance_path, expected)
    return expected


def invalidate_artifact_manifest(root: Path) -> None:
    for name in EXCLUDED_ROOT_FILES:
        (root / name).unlink(missing_ok=True)


def collect_files(root: Path) -> list[dict]:
    records = []
    seen = set()
    provenance = root / BUILD_PROVENANCE_FILENAME
    if provenance.is_file():
        records.append(
            {
                "path": provenance.name,
                "size": provenance.stat().st_size,
                "sha256": sha256(provenance),
            }
        )
        seen.add(provenance.name)
    for subdir in ARTIFACT_SUBDIRS:
        logical_root = root / subdir
        if not logical_root.exists():
            continue
        physical_root = logical_root.resolve()
        for path in sorted(physical_root.rglob("*")):
            if not path.is_file():
                continue
            logical = Path(subdir) / path.relative_to(physical_root)
            logical_name = str(logical)
            if logical.name in EXCLUDED_ROOT_FILES or logical_name in seen:
                continue
            seen.add(logical_name)
            records.append(
                {
                    "path": logical_name,
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return records


def validate_required_artifacts(
    root: Path, *, ptr_tokens: int, max_spatial_slots: int
) -> dict:
    required = [
        root / "backbone_detector/single_simplified.onnx",
        root / "backbone_detector/tuned.mxr",
        root / "backbone_detector/tuned_gpuio.mxr",
        root / "detector_modules/detr_encoder_simplified.onnx",
        root / "detr_decoder_fixed/detr_decoder_fixed_simplified.onnx",
        root / "detr_decoder_fixed/direct_gpuio.mxr",
        root / "detr_decoder_fixed/direct_gpuio.mxr.sha256",
    ]
    required.extend(
        root / "tracker_modules"
        / f"memory_attention_fixed_S{slot}_P{ptr_tokens}.onnx"
        for slot in range(1, max_spatial_slots + 1)
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "artifact root is incomplete: " + ", ".join(missing)
        )

    cache = root / "tracker_modules/ort_cache_mem_attn"
    mxrs = sorted(cache.glob("*.mxr"))
    if len(mxrs) != max_spatial_slots:
        raise RuntimeError(
            "memory-attention cache coverage mismatch: "
            f"expected {max_spatial_slots}, found {len(mxrs)} in {cache}"
        )
    policy_path = cache / "compile_policy.json"
    try:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"missing or invalid memory compile policy: {policy_path}"
        ) from exc
    if policy.get("migraphx_skip_benchmarking") is not False:
        raise RuntimeError("memory compile policy did not record full benchmarking")
    if policy.get("ptr_tokens") != ptr_tokens:
        raise RuntimeError("memory compile policy pointer-token mismatch")
    if policy.get("max_spatial_slots") != max_spatial_slots:
        raise RuntimeError("memory compile policy slot-count mismatch")
    expected_ops = {
        f"S{slot}": "" if slot == 8 else "attention"
        for slot in range(1, max_spatial_slots + 1)
    }
    if policy.get("specific_ops_by_slot") != expected_ops:
        raise RuntimeError("memory compile policy specific-op mismatch")

    expected_onnx = {
        f"S{slot}": sha256(
            root / "tracker_modules"
            / f"memory_attention_fixed_S{slot}_P{ptr_tokens}.onnx"
        )
        for slot in range(1, max_spatial_slots + 1)
    }
    if policy.get("onnx_sha256_by_slot") != expected_onnx:
        raise RuntimeError("memory compile policy ONNX hash mismatch")
    recorded_cache = {
        row.get("name"): row.get("size")
        for row in policy.get("cache_files", []) if isinstance(row, dict)
    }
    actual_cache = {path.name: path.stat().st_size for path in mxrs}
    if recorded_cache != actual_cache:
        raise RuntimeError("memory compile policy cache-file mismatch")

    decoder = root / "detr_decoder_fixed/direct_gpuio.mxr"
    sidecar = decoder.with_suffix(decoder.suffix + ".sha256")
    try:
        recorded_digest, recorded_name = sidecar.read_text(
            encoding="ascii"
        ).split()
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"invalid fixed-decoder checksum: {sidecar}") from exc
    if recorded_name != decoder.name or recorded_digest != sha256(decoder):
        raise RuntimeError("fixed-decoder checksum mismatch")
    return policy


def build_manifest(
    root: Path,
    checkpoint: Path,
    *,
    imgsz: int,
    ptr_tokens: int,
    max_spatial_slots: int,
) -> dict:
    root = root.resolve()
    checkpoint = checkpoint.resolve()
    expected_provenance = current_build_provenance(
        checkpoint,
        imgsz=imgsz,
        ptr_tokens=ptr_tokens,
        max_spatial_slots=max_spatial_slots,
    )
    provenance_path = root / BUILD_PROVENANCE_FILENAME
    try:
        recorded_provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"missing or invalid build provenance: {provenance_path}"
        ) from exc
    if recorded_provenance != expected_provenance:
        raise RuntimeError(
            "artifact files were not built from the current "
            "source/checkpoint/image identity"
        )
    policy = validate_required_artifacts(
        root,
        ptr_tokens=ptr_tokens,
        max_spatial_slots=max_spatial_slots,
    )
    stack = runtime_metadata()
    if not stack["providers"] or stack["providers"][0] != "MIGraphXExecutionProvider":
        raise RuntimeError(
            "artifact manifest requires MIGraphXExecutionProvider as primary: "
            f"{stack['providers']}"
        )

    return {
        "schema": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(root),
        "source": recorded_provenance["source"],
        "checkpoint": {
            "path": str(checkpoint),
            **recorded_provenance["checkpoint"],
        },
        "build": {
            **recorded_provenance["build"],
            "platform": platform.platform(),
        },
        "build_provenance": recorded_provenance,
        "stack": stack,
        "memory_attention": policy,
        "files": collect_files(root),
    }


def write_manifest(root: Path, manifest: dict) -> None:
    manifest_path = root / "ARTIFACT_MANIFEST.json"
    sums_path = root / "SHA256SUMS"
    manifest_hash_path = root / "ARTIFACT_MANIFEST.sha256"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sums_path.write_text(
        "".join(
            f"{row['sha256']}  {row['path']}\n"
            for row in manifest["files"]
        ),
        encoding="ascii",
    )
    manifest_hash_path.write_text(
        f"{sha256(manifest_path)}  {manifest_path.name}\n",
        encoding="ascii",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--ptr-tokens", type=int, required=True)
    parser.add_argument("--max-spatial-slots", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest = build_manifest(
        root,
        args.checkpoint,
        imgsz=args.imgsz,
        ptr_tokens=args.ptr_tokens,
        max_spatial_slots=args.max_spatial_slots,
    )
    write_manifest(root, manifest)
    print(
        f"wrote artifact manifest for {len(manifest['files'])} files: "
        f"{root / 'ARTIFACT_MANIFEST.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
