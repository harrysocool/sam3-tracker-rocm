"""CPU-only tests for artifact provenance and checksums."""

import json
from pathlib import Path

import pytest

from export import write_artifact_manifest as artifact_manifest
from export.tracker_modules import compile_memory_attention as memory_compiler


def _fixture(root: Path, checkpoint: Path) -> None:
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
    checkpoint.write_bytes(b"checkpoint")


def test_manifest_records_complete_identity(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    checkpoint = tmp_path / "model.safetensors"
    _fixture(root, checkpoint)
    monkeypatch.setenv("SAM3_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("SAM3_SOURCE_DIRTY", "0")
    monkeypatch.setenv("SAM3_DOCKER_IMAGE_REF", "sam3:test")
    monkeypatch.setenv("SAM3_DOCKER_IMAGE_ID", "sha256:" + "b" * 64)
    monkeypatch.setenv("SAM3_EC_POWER_MODE", "performance")
    monkeypatch.setattr(
        artifact_manifest,
        "runtime_metadata",
        lambda: {
            "torch": "test", "torch_hip": "test", "onnxruntime": "1.24.2",
            "migraphx": "2.17", "migraphx_module": "/test/migraphx.so",
            "providers": ["MIGraphXExecutionProvider", "CPUExecutionProvider"],
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
    paths = {row["path"] for row in manifest["files"]}
    assert "tracker_modules/ort_cache_mem_attn/compile_policy.json" in paths
    assert len([path for path in paths if path.endswith(".mxr")]) == 13
    assert (root / "ARTIFACT_MANIFEST.json").is_file()
    assert (root / "ARTIFACT_MANIFEST.sha256").is_file()
    assert (root / "SHA256SUMS").is_file()


def test_manifest_rejects_missing_memory_cache(tmp_path):
    root = tmp_path / "artifacts"
    checkpoint = tmp_path / "model.safetensors"
    _fixture(root, checkpoint)
    (memory_compiler.cache_path(root) / "S10.mxr").unlink()
    with pytest.raises(RuntimeError, match="expected 10, found 9"):
        artifact_manifest.validate_required_artifacts(
            root, ptr_tokens=64, max_spatial_slots=10
        )
