"""CLI contracts for the accelerated offline entry point and reference modes."""

from pathlib import Path

import pytest

from tools import text_baseline


def _parse(monkeypatch, *extra, video=True):
    monkeypatch.delenv("SAM3_DEFAULT_ONNX_DIR", raising=False)
    source = ["--video", "assets/blackswan.mp4"] if video else ["--image", "assets/truck.jpg"]
    return text_baseline.parse_args([
        "--checkpoint", "model/sam3", "--text", "swan", *source, *extra,
    ])


@pytest.mark.parametrize("video", [False, True])
def test_accelerated_defaults_select_the_supported_resolution(monkeypatch, video):
    args = _parse(monkeypatch, video=video)
    assert args.imgsz == 504
    assert args.dtype == "fp16"
    assert args.mig is True
    assert args.fixed_detr_decoder is True
    assert args.parallel_tail is True
    assert args.pipeline_backbone is video
    assert args.onnx_dir == Path("onnx_files_504_mgx217")
    assert args.text == ["swan"]
    assert args.max_objects == 5
    assert args.max_frames == 120


def test_artifact_root_uses_container_environment(monkeypatch):
    monkeypatch.setenv("SAM3_DEFAULT_ONNX_DIR", "/models/onnx_files_504")
    args = text_baseline.parse_args([
        "--checkpoint", "model/sam3", "--text", "swan",
        "--video", "assets/blackswan.mp4",
    ])
    assert args.onnx_dir == Path("/models/onnx_files_504")


def test_explicit_artifact_root_overrides_container_environment(monkeypatch):
    monkeypatch.setenv("SAM3_DEFAULT_ONNX_DIR", "/models/onnx_files_504")
    args = text_baseline.parse_args([
        "--checkpoint", "model/sam3", "--text", "swan",
        "--video", "assets/blackswan.mp4", "--onnx-dir", "/custom/artifacts",
    ])
    assert args.onnx_dir == Path("/custom/artifacts")


@pytest.mark.parametrize("video", [False, True])
def test_pytorch_reference_disables_gpu_overlap(monkeypatch, video):
    args = _parse(monkeypatch, "--no-mig", video=video)
    assert args.imgsz == 504
    assert args.mig is False
    assert args.fixed_detr_decoder is False
    assert args.parallel_tail is False
    assert args.pipeline_backbone is False


def test_native_resolution_pytorch_reference_remains_available(monkeypatch):
    args = _parse(monkeypatch, "--no-mig", "--imgsz", "1008")
    assert args.imgsz == 1008
    assert args.fixed_detr_decoder is False
    assert not args.mig and not args.parallel_tail and not args.pipeline_backbone


def test_serial_mig_disables_prefetch_automatically(monkeypatch):
    args = _parse(monkeypatch, "--no-parallel-tail")
    assert args.mig is True
    assert args.parallel_tail is False
    assert args.pipeline_backbone is False


def test_same_frame_overlap_can_run_without_prefetch(monkeypatch):
    args = _parse(monkeypatch, "--no-pipeline-backbone")
    assert args.mig is True
    assert args.parallel_tail is True
    assert args.pipeline_backbone is False


def test_existing_explicit_acceleration_flags_still_work(monkeypatch):
    args = _parse(monkeypatch, "--mig", "--parallel-tail", "--pipeline-backbone")
    assert args.mig and args.parallel_tail and args.pipeline_backbone


def test_native_decoder_comparison_preserves_other_acceleration(monkeypatch):
    args = _parse(monkeypatch, "--no-fixed-detr-decoder")
    assert args.fixed_detr_decoder is False
    assert args.mig and args.parallel_tail and args.pipeline_backbone


def test_fixed_decoder_can_be_requested_explicitly(monkeypatch):
    args = _parse(monkeypatch, "--fixed-detr-decoder")
    assert args.fixed_detr_decoder is True


@pytest.mark.parametrize(("flags", "message"), [
    (["--no-mig", "--parallel-tail"], "--parallel-tail requires --mig"),
    (["--no-mig", "--pipeline-backbone"], "--pipeline-backbone requires --parallel-tail"),
    (["--no-parallel-tail", "--pipeline-backbone"], "--pipeline-backbone requires --parallel-tail"),
    (["--no-mig", "--fixed-detr-decoder"], "--fixed-detr-decoder requires --mig"),
    (["--imgsz", "1008", "--fixed-detr-decoder"], "--fixed-detr-decoder requires --imgsz 504"),
])
def test_invalid_acceleration_combinations_are_rejected(monkeypatch, flags, message):
    with pytest.raises(SystemExit, match=message):
        _parse(monkeypatch, *flags)


def test_image_cannot_force_next_frame_prefetch(monkeypatch):
    with pytest.raises(SystemExit, match="--pipeline-backbone requires --video"):
        _parse(monkeypatch, "--pipeline-backbone", video=False)


@pytest.mark.parametrize("flags", [
    ["--mig", "--no-mig"],
    ["--parallel-tail", "--no-parallel-tail"],
    ["--pipeline-backbone", "--no-pipeline-backbone"],
    ["--fixed-detr-decoder", "--no-fixed-detr-decoder"],
    ["--imgsz", "512"],
])
def test_contradictory_flags_and_unsupported_resolution_are_rejected(monkeypatch, flags):
    with pytest.raises(SystemExit) as exc:
        _parse(monkeypatch, *flags)
    assert exc.value.code == 2


def test_1008_mig_requires_explicit_artifact_selection(monkeypatch):
    with pytest.raises(SystemExit, match="1008px MIG inference requires an explicit --onnx-dir"):
        _parse(monkeypatch, "--imgsz", "1008")
    args = _parse(monkeypatch, "--imgsz", "1008", "--onnx-dir", "/models/custom")
    assert args.imgsz == 1008
    assert args.onnx_dir == Path("/models/custom")
    assert args.fixed_detr_decoder is False


def _forbid_model_load(*args, **kwargs):
    raise AssertionError("Model loading must not start when runtime preflight fails")


def test_missing_rocm_gpu_fails_before_model_load(monkeypatch):
    args = _parse(monkeypatch)
    monkeypatch.setattr(text_baseline, "parse_args", lambda: args)
    monkeypatch.setattr(text_baseline.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(text_baseline.AutoProcessor, "from_pretrained", _forbid_model_load)
    with pytest.raises(SystemExit, match="MIG inference requires a ROCm GPU"):
        text_baseline.main()


@pytest.mark.parametrize("has_directory", [False, True])
def test_missing_artifacts_fail_before_model_load(monkeypatch, tmp_path, has_directory):
    artifact_dir = tmp_path if has_directory else tmp_path / "missing"
    args = _parse(monkeypatch, "--onnx-dir", str(artifact_dir))
    monkeypatch.setattr(text_baseline, "parse_args", lambda: args)
    monkeypatch.setattr(text_baseline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(text_baseline.torch.version, "hip", "test")
    monkeypatch.setattr(text_baseline.AutoProcessor, "from_pretrained", _forbid_model_load)
    message = "Backbone prefetch requires" if has_directory else "MIG artifact directory not found"
    with pytest.raises(SystemExit, match=message):
        text_baseline.main()


def test_missing_fixed_decoder_fails_before_model_load(monkeypatch, tmp_path):
    args = _parse(monkeypatch, "--onnx-dir", str(tmp_path), "--no-pipeline-backbone")
    monkeypatch.setattr(text_baseline, "parse_args", lambda: args)
    monkeypatch.setattr(text_baseline.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(text_baseline.torch.version, "hip", "test")
    monkeypatch.setattr(text_baseline.AutoProcessor, "from_pretrained", _forbid_model_load)
    with pytest.raises(SystemExit, match="Fixed DETR decoder artifact not found"):
        text_baseline.main()


@pytest.mark.parametrize(("prompts", "expected"), [
    (["people", "dog"], ["people", "dog"]),
    (["person on a bike"], ["person on a bike"]),
    ([" swan ", "water", "swan"], ["swan", "water"]),
])
def test_multiple_prompts_and_quoted_phrases(monkeypatch, prompts, expected):
    assert _parse(monkeypatch, "--text", *prompts).text == expected


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("2", 2), ("-1", 5)])
def test_object_limit_semantics(monkeypatch, value, expected):
    assert _parse(monkeypatch, "--max-objects", value).max_objects == expected


def test_zero_frame_limit_means_read_to_end(monkeypatch):
    assert _parse(monkeypatch, "--max-frames", "0").max_frames == 0


@pytest.mark.parametrize("flags", [
    ["--text", "  "],
    ["--max-objects", "-2"],
    ["--max-frames", "-1"],
    ["--min-score", "-0.1"],
    ["--min-score", "1.1"],
    ["--min-score", "nan"],
])
def test_invalid_output_options_are_rejected(monkeypatch, flags):
    with pytest.raises(SystemExit) as exc:
        _parse(monkeypatch, *flags)
    assert exc.value.code == 2
