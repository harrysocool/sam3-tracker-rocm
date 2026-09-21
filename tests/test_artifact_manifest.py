"""CPU-only tests for artifact provenance and checksums."""

import json
from pathlib import Path

import pytest

from export import write_artifact_manifest as artifact_manifest
from export.tracker_modules import compile_memory_attention as memory_compiler


def _fixture(root: Path, checkpoint: Path) -> None:
    checkpoint.write_bytes(b"checkpoint")
    artifact_manifest.prepare_build_root(
        root,
        checkpoint,
        imgsz=504,
        ptr_tokens=64,
        max_spatial_slots=10,
        reset_full_build=False,
    )
    required = (
        "backbone_detector/single_simplified.onnx",
        "backbone_detector/tuned.mxr",
        "backbone_detector/tuned_gpuio.mxr",
        "detector_modules/detr_encoder_simplified.onnx",
        "detr_decoder_fixed/detr_decoder_fixed_simplified.onnx",
        "detr_decoder_fixed/direct_gpuio.mxr",
        "detr_decoder_fixed/direct_gpuio.mxr.sha256",
    )
    for name in required:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    detector_cache = root / "detector_modules/ort_cache/detr.mxr"
    detector_cache.parent.mkdir(parents=True, exist_ok=True)
    detector_cache.write_bytes(b"detr-cache")
    decoder = root / "detr_decoder_fixed/direct_gpuio.mxr"
    decoder.with_suffix(decoder.suffix + ".sha256").write_text(
        f"{artifact_manifest.sha256(decoder)}  {decoder.name}\n"
    )
    for slot in range(1, 11):
        path = memory_compiler.graph_path(root, slot, 64)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"S{slot}".encode())
    cache = memory_compiler.cache_path(root)
    cache.mkdir(parents=True, exist_ok=True)
    policy = memory_compiler.expected_policy(root, 64, 10)
    files = []
    for slot in range(1, 11):
        path = cache / f"S{slot}.mxr"
        path.write_bytes(f"MXR{slot}".encode())
        files.append({"name": path.name, "size": path.stat().st_size})
    policy["cache_files"] = files
    (cache / memory_compiler.POLICY_FILENAME).write_text(json.dumps(policy))


def _identity_env(monkeypatch) -> None:
    monkeypatch.setenv("SAM3_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("SAM3_SOURCE_DIRTY", "0")
    monkeypatch.setenv("SAM3_DOCKER_IMAGE_REF", "sam3:test")
    monkeypatch.setenv("SAM3_DOCKER_IMAGE_ID", "sha256:" + "b" * 64)
    monkeypatch.setenv("SAM3_EC_POWER_MODE", "performance")
    monkeypatch.setenv("SAM3_BUILD_HOST_ID", "test-evo-x2")


def test_manifest_records_complete_identity(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    checkpoint = tmp_path / "model.safetensors"
    _identity_env(monkeypatch)
    _fixture(root, checkpoint)
    monkeypatch.setattr(
        artifact_manifest,
        "runtime_metadata",
        lambda: {
            "torch": "test", "torch_hip": "test", "onnxruntime": "1.24.2",
            "migraphx": "2.17", "migraphx_module": "/test/migraphx.so",
            "providers": ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
            "gpu_name": "test-gpu", "gpu_arch": "gfx1151",
            "gpu_arch_detail": "gfx1151:sramecc+:xnack-",
        },
    )

    manifest = artifact_manifest.build_manifest(
        root, checkpoint, imgsz=504, ptr_tokens=64, max_spatial_slots=10
    )
    artifact_manifest.write_manifest(root, manifest)

    assert manifest["schema"] == 2
    assert manifest["source"]["commit"] == "a" * 40
    assert manifest["source"]["dirty"] is False
    assert manifest["checkpoint"]["sha256"] == artifact_manifest.sha256(checkpoint)
    assert manifest["build"]["image_id"] == "sha256:" + "b" * 64
    assert manifest["build"]["ec_power_mode"] == "performance"
    assert manifest["hardware"]["build_host_id"] == "test-evo-x2"
    assert manifest["hardware"]["gpu_arch"] == "gfx1151"
    paths = {row["path"] for row in manifest["files"]}
    assert artifact_manifest.BUILD_PROVENANCE_FILENAME in paths
    assert "tracker_modules/ort_cache_mem_attn/compile_policy.json" in paths
    assert len([path for path in paths if path.endswith(".mxr")]) == 14
    assert (root / "ARTIFACT_MANIFEST.json").is_file()
    assert (root / "ARTIFACT_MANIFEST.sha256").is_file()
    assert (root / "SHA256SUMS").is_file()


def test_manifest_rejects_missing_memory_cache(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    checkpoint = tmp_path / "model.safetensors"
    _identity_env(monkeypatch)
    _fixture(root, checkpoint)
    (memory_compiler.cache_path(root) / "S10.mxr").unlink()
    with pytest.raises(RuntimeError, match="expected 10, found 9"):
        artifact_manifest.validate_required_artifacts(
            root, ptr_tokens=64, max_spatial_slots=10
        )


def test_manifest_rejects_missing_detr_cache(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    checkpoint = tmp_path / "model.safetensors"
    _identity_env(monkeypatch)
    _fixture(root, checkpoint)
    (root / "detector_modules/ort_cache/detr.mxr").unlink()
    with pytest.raises(RuntimeError, match="DETR encoder cache coverage"):
        artifact_manifest.validate_required_artifacts(
            root, ptr_tokens=64, max_spatial_slots=10
        )


def test_nonempty_root_without_provenance_is_rejected(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    legacy = root / "legacy.bin"
    legacy.write_bytes(b"preserve")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    _identity_env(monkeypatch)

    with pytest.raises(RuntimeError, match="build provenance does not match"):
        artifact_manifest.prepare_build_root(
            root,
            checkpoint,
            imgsz=504,
            ptr_tokens=64,
            max_spatial_slots=10,
            reset_full_build=False,
        )

    assert legacy.read_bytes() == b"preserve"


def test_resume_rejects_changed_checkpoint(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"first")
    _identity_env(monkeypatch)
    artifact_manifest.prepare_build_root(
        root,
        checkpoint,
        imgsz=504,
        ptr_tokens=64,
        max_spatial_slots=10,
        reset_full_build=False,
    )
    checkpoint.write_bytes(b"second")

    with pytest.raises(RuntimeError, match="build provenance does not match"):
        artifact_manifest.prepare_build_root(
            root,
            checkpoint,
            imgsz=504,
            ptr_tokens=64,
            max_spatial_slots=10,
            reset_full_build=False,
        )


def test_explicit_full_force_resets_generated_artifacts(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    generated = root / "backbone_detector/old.mxr"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"old")
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    _identity_env(monkeypatch)

    artifact_manifest.prepare_build_root(
        root,
        checkpoint,
        imgsz=504,
        ptr_tokens=64,
        max_spatial_slots=10,
        reset_full_build=True,
    )

    assert not generated.exists()
    assert (root / artifact_manifest.BUILD_PROVENANCE_FILENAME).is_file()
