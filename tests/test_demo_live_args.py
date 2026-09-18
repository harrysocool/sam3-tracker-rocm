from __future__ import annotations

import inspect
import sys

import pytest

import demo_live
from tracker.hybrid_inference import SAM3HybridLive
from tracker.live_inference import SAM3Live


def _parse(monkeypatch, *extra):
    monkeypatch.delenv("SAM3_DEFAULT_ONNX_DIR", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_live.py",
            "--checkpoint",
            "model/sam3",
            "--video",
            "assets/blackswan.mp4",
            "--text",
            "swan",
            *extra,
        ],
    )
    return demo_live.parse_args()


def test_live_defaults_to_mig_parallel_tail_and_fixed_auto(monkeypatch):
    args = _parse(monkeypatch)
    assert args.mig is True
    assert args.parallel_tail is True
    assert args.fixed_detr_decoder is None
    assert str(args.onnx_dir) == "onnx_files_504_mgx217"
    assert args.redetect_interval_ms == 0.0
    assert args.max_objects == 5
    assert args.max_frames == 0
    assert args.num_maskmem is None
    assert args.max_cond_frames is None


def test_parallel_tail_can_be_disabled_explicitly(monkeypatch):
    args = _parse(monkeypatch, "--no-parallel-tail")
    assert args.parallel_tail is False


def test_no_mig_explicitly_selects_pytorch_and_serial(monkeypatch):
    args = _parse(monkeypatch, "--no-mig")
    assert args.mig is False
    assert args.parallel_tail is False


def test_parallel_tail_still_requires_mig(monkeypatch):
    with pytest.raises(SystemExit, match="--parallel-tail requires --mig"):
        _parse(monkeypatch, "--no-mig", "--parallel-tail")


def test_fixed_decoder_can_be_disabled_explicitly(monkeypatch):
    args = _parse(monkeypatch, "--no-fixed-detr-decoder")
    assert args.fixed_detr_decoder is False


def test_fixed_decoder_force_requires_mig(monkeypatch):
    with pytest.raises(SystemExit, match="--fixed-detr-decoder requires --mig"):
        _parse(monkeypatch, "--no-mig", "--fixed-detr-decoder")


def test_bootstrap_requires_native_decoder_diagnostic(monkeypatch):
    with pytest.raises(SystemExit, match="incompatible with the fixed"):
        _parse(monkeypatch, "--bootstrap-frames", "2")
    args = _parse(
        monkeypatch,
        "--bootstrap-frames",
        "2",
        "--no-fixed-detr-decoder",
    )
    assert args.bootstrap_frames == 2
    assert args.fixed_detr_decoder is False

    args = _parse(
        monkeypatch,
        "--redetect-interval-ms",
        "1000",
        "--bootstrap-frames",
        "2",
        "--no-fixed-detr-decoder",
    )
    assert args.bootstrap_frames == 2


def test_live_api_defaults_parallel_tail_to_auto():
    assert inspect.signature(SAM3Live).parameters["mig"].default is True
    assert inspect.signature(SAM3Live).parameters["parallel_tail"].default is None
    assert inspect.signature(SAM3Live).parameters["num_maskmem"].default == 3
    assert inspect.signature(SAM3Live).parameters["max_cond_frame_num"].default == 1
    assert (
        inspect.signature(SAM3Live).parameters["fixed_detr_decoder"].default
        is None
    )


def test_live_api_rejects_bootstrap_with_default_fixed_decoder(tmp_path):
    with pytest.raises(ValueError, match="cannot be combined with bootstrap_frames"):
        SAM3Live(
            checkpoint="unused",
            prompts=["obstacle"],
            onnx_dir=tmp_path,
            mig=True,
            bootstrap_frames=1,
        )
    assert (
        inspect.signature(SAM3HybridLive).parameters["parallel_tail"].default
        is None
    )
    assert inspect.signature(SAM3HybridLive).parameters["mig"].default is True
    assert inspect.signature(SAM3HybridLive).parameters["num_maskmem"].default == 7
    assert inspect.signature(SAM3HybridLive).parameters["max_cond_frame_num"].default == 4
    assert (
        inspect.signature(SAM3HybridLive).parameters["fixed_detr_decoder"].default
        is None
    )


def test_filter_result_preserves_scheduler_metadata():
    result = {
        "object_ids": [1, 2],
        "scores": {1: 0.9, 2: 0.1},
        "masks": {1: "mask-1", 2: "mask-2"},
        "boxes": {1: "box-1", 2: "box-2"},
        "prompt_to_obj_ids": {"object": [1, 2]},
        "frame_idx": 7,
        "detected": False,
        "keyframe": False,
        "lost_object_ids": [3, 4],
        "redetect_reason": None,
        "negative_evidence_valid": False,
    }

    filtered = demo_live.filter_result(result, min_score=0.5)

    assert filtered == {
        "object_ids": [1],
        "scores": {1: 0.9},
        "masks": {1: "mask-1"},
        "boxes": {1: "box-1"},
        "prompt_to_obj_ids": {"object": [1]},
        "frame_idx": 7,
        "detected": False,
        "keyframe": False,
        "lost_object_ids": [3, 4],
        "redetect_reason": None,
        "negative_evidence_valid": False,
    }


def test_explicit_warmup_uses_propagation_after_first_hybrid_frame():
    assert [
        demo_live._warmup_uses_full_detection(index, hybrid=True)
        for index in range(4)
    ] == [True, False, False, False]


def test_explicit_warmup_keeps_full_detection_for_non_hybrid_mode():
    assert [
        demo_live._warmup_uses_full_detection(index, hybrid=False)
        for index in range(4)
    ] == [True, True, True, True]


@pytest.mark.parametrize(("prompts", "expected"), [
    (["people", "dog"], ["people", "dog"]),
    (["person on a bike"], ["person on a bike"]),
    ([" swan ", "water", "swan"], ["swan", "water"]),
])
def test_live_prompt_normalization(monkeypatch, prompts, expected):
    assert _parse(monkeypatch, "--text", *prompts).text == expected


@pytest.mark.parametrize(("value", "expected"), [("0", 0), ("2", 2), ("-1", 5)])
def test_live_object_limit_semantics(monkeypatch, value, expected):
    assert _parse(monkeypatch, "--max-objects", value).max_objects == expected


def test_live_memory_horizon_overrides(monkeypatch):
    args = _parse(
        monkeypatch,
        "--num-maskmem", "5",
        "--max-cond-frames", "2",
    )
    assert args.num_maskmem == 5
    assert args.max_cond_frames == 2


@pytest.mark.parametrize("flags", [
    ["--text", "  "],
    ["--max-objects", "-2"],
    ["--max-frames", "-1"],
    ["--min-score", "-0.1"],
    ["--min-score", "1.1"],
    ["--min-score", "nan"],
])
def test_live_invalid_output_options_are_rejected(monkeypatch, flags):
    with pytest.raises(SystemExit) as exc:
        _parse(monkeypatch, *flags)
    assert exc.value.code == 2
