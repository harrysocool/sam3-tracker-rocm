"""Docker integration coverage for the release launcher portability path."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docker/rocm714/run.sh"
BASH = shutil.which("bash")
DEFAULT_IMAGE = "sam3-gpu714-ort1242-mgx217-gfx1151:0.2.0-rc4-local"


def _require_release_container(image: str) -> None:
    if not BASH:
        pytest.skip("bash is required")
    if not (Path("/dev/kfd").exists() and Path("/dev/dri").is_dir()):
        pytest.skip("the release wrapper requires ROCm device nodes")
    if not shutil.which("docker"):
        pytest.skip("docker is required")
    inspected = subprocess.run(
        ["docker", "image", "inspect", image],
        text=True,
        capture_output=True,
        check=False,
    )
    if inspected.returncode:
        pytest.skip(f"release image is unavailable: {image}")


def test_wrapper_handles_unknown_uid_and_multihop_absolute_weight_link(tmp_path: Path):
    image = os.environ.get("SAM3_TEST_IMAGE", DEFAULT_IMAGE)
    _require_release_container(image)

    model_dir = tmp_path / "model"
    snapshot_dir = tmp_path / "snapshot"
    blob_dir = tmp_path / "blobs"
    onnx_dir = tmp_path / "onnx"
    for directory in (model_dir, snapshot_dir, blob_dir, onnx_dir):
        directory.mkdir()

    payload = b"sam3-wrapper-portability-fixture\n"
    blob = blob_dir / "weight-by-hash"
    blob.write_bytes(payload)
    snapshot_weight = snapshot_dir / "model.safetensors"
    snapshot_weight.symlink_to(blob)
    (model_dir / "model.safetensors").symlink_to(snapshot_weight)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_id = fake_bin / "id"
    fake_id.write_text(
        "#!/bin/sh\n"
        "case \"${1:-}\" in\n"
        "  -u|-g) printf '%s\\n' 424242 ;;\n"
        "  -un) printf '%s\\n' portable-user ;;\n"
        "  *) exec /usr/bin/id \"$@\" ;;\n"
        "esac\n"
    )
    fake_id.chmod(0o755)

    code = (
        "import getpass; from pathlib import Path; "
        "from transformers import AutoProcessor, Sam3VideoModel; "
        "assert getpass.getuser() == 'portable-user'; "
        "assert Path('/models/sam3/model.safetensors').read_bytes() == "
        "b'sam3-wrapper-portability-fixture\\n'"
    )
    env = os.environ | {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SAM3_DOCKER_IMAGE": image,
        "SAM3_MODEL_DIR": str(model_dir),
        "SAM3_ONNX_DIR": str(onnx_dir),
        "SAM3_DOCKER_STRICT": "1",
    }
    result = subprocess.run(
        [BASH, str(RUNNER), "python", "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_wrapper_rejects_broken_absolute_weight_link(tmp_path: Path):
    if not BASH:
        pytest.skip("bash is required")
    if not (Path("/dev/kfd").exists() and Path("/dev/dri").is_dir()):
        pytest.skip("the release wrapper requires ROCm device nodes")

    model_dir = tmp_path / "model"
    onnx_dir = tmp_path / "onnx"
    model_dir.mkdir()
    onnx_dir.mkdir()
    (model_dir / "model.safetensors").symlink_to(tmp_path / "missing" / "weights")
    env = os.environ | {
        "SAM3_MODEL_DIR": str(model_dir),
        "SAM3_ONNX_DIR": str(onnx_dir),
    }
    result = subprocess.run(
        [BASH, str(RUNNER), "true"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "absolute symlink does not resolve to a file" in result.stderr
