"""CPU-only guards for the canonical latest-frame benchmark contract."""

import json
import sys

import pytest

from eval.benchmarks import benchmark_latest_frame_canonical as benchmark


def test_canonical_profiles_are_fully_pinned():
    assert benchmark.PROFILE_CONFIGS == {
        "canonical-250": {
            "loops": 5, "capture_fps": 24.0, "warm_outputs": 5,
            "tail_outputs": 20, "expected_arrivals": 250,
        },
        "soak-1000": {
            "loops": 20, "capture_fps": 24.0, "warm_outputs": 5,
            "tail_outputs": 50, "expected_arrivals": 1000,
        },
    }
    profile = benchmark._resolve_profile("canonical-250")
    assert profile["prompt"] == "swan"
    assert profile["video_sha256"] == benchmark.CANONICAL_VIDEO_SHA256


def test_canonical_profile_rejects_changed_video(tmp_path):
    changed = tmp_path / "blackswan.mp4"
    changed.write_bytes(b"not-the-canonical-input")
    with pytest.raises(RuntimeError, match="input video SHA256 mismatch"):
        benchmark._resolve_profile("canonical-250", changed)


def test_power_policy_records_complete_attestation():
    policy = benchmark._power_policy(
        {
            "SAM3_STAPM_LIMIT_W": "120",
            "SAM3_FAST_PPT_LIMIT_W": "140",
            "SAM3_SLOW_PPT_LIMIT_W": "120",
        }
    )
    assert policy == {
        "stapm_limit_w": 120.0,
        "fast_ppt_limit_w": 140.0,
        "slow_ppt_limit_w": 120.0,
        "complete": True,
        "source": "environment_attestation",
    }


def test_power_policy_marks_missing_values_incomplete():
    policy = benchmark._power_policy({"SAM3_SLOW_PPT_LIMIT_W": "120"})
    assert policy["slow_ppt_limit_w"] == 120.0
    assert policy["complete"] is False
    assert policy["source"] == "not_fully_reported"


def test_power_policy_rejects_invalid_values():
    with pytest.raises(ValueError, match="SAM3_SLOW_PPT_LIMIT_W"):
        benchmark._power_policy({"SAM3_SLOW_PPT_LIMIT_W": "invalid"})


def test_execution_hardware_records_safe_host_fields(monkeypatch, tmp_path):
    for name, value in {
        "sys_vendor": "GMKtec",
        "product_name": "NucBox_EVO-X2",
        "bios_version": "test-bios",
    }.items():
        (tmp_path / name).write_text(value)
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: False)
    hardware = benchmark._execution_hardware(
        {"SAM3_BUILD_HOST_ID": "harry-evo-x2"}, tmp_path
    )
    assert hardware["build_host_id"] == "harry-evo-x2"
    assert hardware["system_vendor"] == "GMKtec"
    assert hardware["product_name"] == "NucBox_EVO-X2"
    assert hardware["bios_version"] == "test-bios"
    assert hardware["gpu_arch"] is None


def test_legacy_custom_timing_arguments_are_rejected(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_latest_frame_canonical.py",
            "--checkpoint", "model",
            "--onnx-dir", "artifacts",
            "--profile", "canonical-250",
            "--out", "result.json",
            "--loops", "1",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        benchmark._parse_args()
    assert caught.value.code == 2


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
