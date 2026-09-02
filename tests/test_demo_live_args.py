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


def test_live_api_defaults_parallel_tail_to_auto():
    assert inspect.signature(SAM3Live).parameters["mig"].default is True
    assert inspect.signature(SAM3Live).parameters["parallel_tail"].default is None
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
    assert (
        inspect.signature(SAM3HybridLive).parameters["fixed_detr_decoder"].default
        is None
    )
