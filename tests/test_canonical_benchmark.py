"""CPU-only guards for the canonical latest-frame benchmark contract."""

import json

import pytest

from eval.benchmarks import benchmark_latest_frame_canonical as benchmark


def _write_manifest(tmp_path, *, dirty=False, extra_files=None):
    files = {"backbone_detector/tuned_gpuio.mxr": b"backbone"}
    files.update(extra_files or {})
    rows = []
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        rows.append(
            {
                "path": name,
                "size": len(content),
                "sha256": benchmark._sha256(path),
            }
        )
    manifest = {
        "schema": 2,
        "source": {"commit": "a" * 40, "dirty": dirty},
        "files": rows,
    }
    manifest_path = tmp_path / "ARTIFACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "ARTIFACT_MANIFEST.sha256").write_text(
        f"{benchmark._sha256(manifest_path)}  {manifest_path.name}\n"
    )
    (tmp_path / "SHA256SUMS").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows)
    )
    return manifest


def test_manifest_supplies_backbone_identity(tmp_path):
    manifest = _write_manifest(tmp_path)
    loaded, digest = benchmark._load_artifact_identity(tmp_path)
    assert loaded == manifest
    assert digest == manifest["files"][0]["sha256"]


def test_dirty_artifact_manifest_is_rejected(tmp_path):
    _write_manifest(tmp_path, dirty=True)
    with pytest.raises(RuntimeError, match="dirty source"):
        benchmark._load_artifact_identity(tmp_path)


def test_non_backbone_artifact_tampering_is_rejected(tmp_path):
    _write_manifest(
        tmp_path,
        extra_files={"tracker_modules/ort_cache_mem_attn/S8.mxr": b"correct"},
    )
    (tmp_path / "tracker_modules/ort_cache_mem_attn/S8.mxr").write_bytes(
        b"changed"
    )
    with pytest.raises(RuntimeError, match="artifact SHA256 mismatch"):
        benchmark._load_artifact_identity(tmp_path)


def test_unrecorded_artifact_is_rejected(tmp_path):
    _write_manifest(tmp_path)
    extra = tmp_path / "tracker_modules/unrecorded.mxr"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"extra")
    with pytest.raises(RuntimeError, match="unrecorded"):
        benchmark._load_artifact_identity(tmp_path)


def test_manifest_checksum_mismatch_is_rejected(tmp_path):
    _write_manifest(tmp_path)
    (tmp_path / "ARTIFACT_MANIFEST.json").write_text("{}")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        benchmark._load_artifact_identity(tmp_path)


@pytest.mark.parametrize(
    "name",
    ["BENCH_NUM_MASKMEM", "BENCH_MAX_COND_FRAMES", "BENCH_KEEP_RECENT"],
)
def test_memory_override_is_rejected(name):
    with pytest.raises(RuntimeError, match=name):
        benchmark._reject_noncanonical_environment({name: "1"})


def test_execution_identity_checks_checkpoint_image_source_and_ec(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"weights")
    manifest = {
        "source": {"commit": "a" * 40},
        "checkpoint": {"sha256": benchmark._sha256(weights)},
        "build": {"image_id": "sha256:" + "b" * 64},
    }
    env = {
        "SAM3_SOURCE_COMMIT": "a" * 40,
        "SAM3_SOURCE_DIRTY": "0",
        "SAM3_DOCKER_IMAGE_ID": "sha256:" + "b" * 64,
        "SAM3_EC_POWER_MODE": "performance",
    }
    benchmark._validate_execution_identity(manifest, checkpoint, env)


def test_execution_identity_rejects_wrong_image(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"weights")
    manifest = {
        "source": {"commit": "a" * 40},
        "checkpoint": {"sha256": benchmark._sha256(weights)},
        "build": {"image_id": "sha256:" + "b" * 64},
    }
    env = {
        "SAM3_SOURCE_COMMIT": "a" * 40,
        "SAM3_SOURCE_DIRTY": "0",
        "SAM3_DOCKER_IMAGE_ID": "sha256:" + "c" * 64,
        "SAM3_EC_POWER_MODE": "performance",
    }
    with pytest.raises(RuntimeError, match="container image"):
        benchmark._validate_execution_identity(manifest, checkpoint, env)
