"""CPU-only guards for the backbone compiler's supported-runtime guidance."""

from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from export.backbone import compile_backbone_mxr as compiler


ROOT = Path(__file__).resolve().parents[1]


def _assert_current_guidance(message):
    for text in (
        "ROCm 7.14 / MIGraphX 2.17",
        "docker/rocm714/run.sh",
        "docker/rocm714/README.md",
        "SAM3_ONNX_DIR",
        "--onnx-dir",
        "baseline MXR files",
    ):
        assert text in message
    for removed in (
        "tools/install_migraphx_patched.sh",
        "PYTHONPATH=/opt/rocm-7.2",
        "2.15+patches",
    ):
        assert removed not in message
    assert (ROOT / "docker/rocm714/run.sh").is_file()
    assert (ROOT / "docker/rocm714/README.md").is_file()


def test_help_points_to_current_runtime_without_importing_migraphx(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["compile_backbone_mxr.py", "--help"])
    # A runtime import would fail: --help must not need MIGraphX or a GPU.
    monkeypatch.setitem(sys.modules, "migraphx", None)

    with pytest.raises(SystemExit) as exc:
        compiler.main()

    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    _assert_current_guidance(help_text)
    assert "SAM3_DOCKER_STRICT=0" in help_text
    assert "--imgsz 504 --backbone-source detector" in help_text
    assert "--onnx-dir /models/onnx_files_504 --gpu-io" in help_text


def test_invalid_host_output_layout_reports_runtime_and_artifact_checks(
    monkeypatch, tmp_path, capsys,
):
    artifact_dir = tmp_path / "backbone_tracker"
    artifact_dir.mkdir()
    source = artifact_dir / "single_simplified.onnx"
    source.touch()
    args = SimpleNamespace(
        onnx_dir=tmp_path, backbone_source="tracker", gpu_io=False,
        no_fp16=False, skip_verify=False, imgsz=2,
    )
    monkeypatch.setattr(compiler, "parse_args", lambda: args)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setenv("MIGRAPHX_SKIP_BENCHMARKING", "1")
    monkeypatch.delenv("MIGRAPHX_MLIR_USE_SPECIFIC_OPS", raising=False)

    output = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    assert not output.flags.c_contiguous
    program = Mock()
    program.run.return_value = [output]
    migraphx = SimpleNamespace(
        __file__="/mock/migraphx/__init__.py",
        parse_onnx=Mock(return_value=program),
        quantize_fp16=Mock(),
        get_target=Mock(return_value="mock-gpu"),
        save=Mock(side_effect=lambda prog, path: Path(path).touch()),
        argument=lambda array: array,
    )
    monkeypatch.setitem(sys.modules, "migraphx", migraphx)

    with pytest.raises(SystemExit) as exc:
        compiler.main()

    message = str(exc.value)
    assert "Outputs are NOT C-contiguous" in message
    assert f"do not use the saved cache {artifact_dir / 'tuned.mxr'}" in message
    assert "printed migraphx module path" in message
    _assert_current_guidance(message)
    assert "migraphx from: /mock/migraphx/__init__.py" in capsys.readouterr().out
    migraphx.parse_onnx.assert_called_once_with(str(source))
    program.compile.assert_called_once_with("mock-gpu", offload_copy=True)
