"""CPU-only guards for the canonical latest-frame benchmark contract."""

import json

import pytest

from eval.benchmarks import benchmark_latest_frame_canonical as benchmark


def test_manifest_supplies_backbone_identity(tmp_path):
    manifest = {
        "schema": 2,
        "source": {"commit": "a" * 40, "dirty": False},
        "files": [
            {
                "path": "backbone_detector/tuned_gpuio.mxr",
                "sha256": "b" * 64,
            }
        ],
    }
    (tmp_path / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest))
    loaded, digest = benchmark._load_artifact_identity(tmp_path)
    assert loaded == manifest
    assert digest == "b" * 64


def test_dirty_artifact_manifest_is_rejected(tmp_path):
    manifest = {"schema": 2, "source": {"dirty": True}, "files": []}
    (tmp_path / "ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="dirty source"):
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
