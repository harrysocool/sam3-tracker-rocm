from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import smoke_live_release as smoke


def _arguments(tmp_path, *extra):
    return ["--checkpoint", str(tmp_path / "model"), "--onnx-dir", str(tmp_path / "onnx"),
            "--output", str(tmp_path / "smoke.json"), *extra]


def _frame():
    return np.zeros((4, 5, 3), dtype=np.uint8)


def _output(frame=None, *, detected=True):
    frame = _frame() if frame is None else frame
    return {"object_ids": [0], "scores": {0: 0.9},
            "masks": {0: np.ones(frame.shape[:2], dtype=bool)},
            "boxes": {0: (0.0, 0.0, 4.0, 3.0)}, "prompt_to_obj_ids": {"swan": [0]},
            "frame_idx": 0, "detected": detected, "negative_evidence_valid": detected}


def test_cli_defaults_and_validation(tmp_path):
    args = smoke.parse_args(_arguments(tmp_path))
    assert args.frames == 12 and args.mode == "both" and args.text == "swan"
    assert args.video.name == "blackswan.mp4"
    for extra in (("--frames", "2"), ("--frames", "no"), ("--text", "  "), ("--mode", "bad")):
        with pytest.raises(SystemExit):
            smoke.parse_args(_arguments(tmp_path, *extra))


def test_import_and_help_work_without_gpu_or_video_modules():
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'onnxruntime', 'migraphx', 'cv2', 'tracker', 'numpy'}:
        raise AssertionError('eager dependency import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
namespace = runpy.run_path(sys.argv[1], run_name='cpu_import')
namespace['main'](['--help'])
"""
    result = subprocess.run([sys.executable, "-c", code, str(Path(smoke.__file__))],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout and "not a throughput" in result.stdout


@pytest.mark.parametrize("version,providers", [
    ("1.24.1", ["MIGraphXExecutionProvider"]), ("1.24.2", ["VitisAIExecutionProvider"]),
    ("1.24.2", ["CPUExecutionProvider"]),
])
def test_runtime_rejects_wrong_stack(version, providers):
    with pytest.raises(ValueError):
        smoke.validate_runtime(version, providers)


def test_runtime_accepts_production_provider():
    smoke.validate_runtime("1.24.2", ["MIGraphXExecutionProvider", "CPUExecutionProvider"])


def test_create_live_keeps_fixed_and_parallel_defaults(tmp_path):
    args = smoke.parse_args(_arguments(tmp_path))
    full, hybrid = [], []
    full_class = lambda **kwargs: full.append(kwargs)
    hybrid_class = lambda **kwargs: hybrid.append(kwargs)
    smoke.create_live(args, "full", full_class, hybrid_class)
    smoke.create_live(args, "hybrid", full_class, hybrid_class)
    for kwargs in (full[0], hybrid[0]):
        assert kwargs["imgsz"] == 504 and kwargs["bootstrap_frames"] == 0
        assert kwargs["mig"] is True and kwargs["periodic_rebootstrap_seconds"] == 0
        assert "parallel_tail" not in kwargs and "fixed_detr_decoder" not in kwargs
    assert hybrid[0]["redetect_interval_ms"] == 1000


def test_output_validation_accepts_current_schema():
    result = smoke.validate_output(_output(), _frame().shape, ["swan"])
    assert result["nonempty_masks"] == 1 and result["object_ids"] == [0]


@pytest.mark.parametrize("change,match", [
    (lambda out: out["scores"].update({0: float("nan")}), "score"),
    (lambda out: out["scores"].update({0: float("inf")}), "score"),
    (lambda out: out["masks"].update({0: np.ones((4, 5), dtype=float)}), "bool ndarray"),
    (lambda out: out["masks"].update({0: np.ones((5, 4), dtype=bool)}), "wrong shape"),
    (lambda out: out["masks"].clear(), "keys do not match"),
    (lambda out: out.update(object_ids=[0, 0]), "duplicate"),
    (lambda out: out.update(object_ids=[True]), "integer IDs"),
    (lambda out: out.update(prompt_to_obj_ids={"dog": [0]}), "Unexpected prompt"),
    (lambda out: out.update(prompt_to_obj_ids={"swan": []}), "ownership does not match"),
    (lambda out: out.update(negative_evidence_valid=False), "negative_evidence_valid"),
    (lambda out: out["boxes"].update({0: (0, 0, float("nan"), 1)}), "Invalid box"),
])
def test_output_validation_rejects_corruption(change, match):
    output = _output()
    change(output)
    with pytest.raises(ValueError, match=match):
        smoke.validate_output(output, _frame().shape, ["swan"])


def test_cross_prompt_id_reuse_is_rejected():
    output = _output()
    output["prompt_to_obj_ids"]["water"] = [0]
    with pytest.raises(ValueError, match="multiple prompts"):
        smoke.validate_output(output, _frame().shape, ["swan", "water"])


class FakeLive:
    def __init__(self, events, *, force_full=False, empty=False):
        self.events, self.force_full, self.empty = events, force_full, empty
        self.parallel_tail = True
        self._fixed_detr_decoder = object()
        self.calls = []

    def infer(self, frame, *, full_detection):
        self.calls.append(full_detection)
        output = _output(frame, detected=bool(full_detection or self.force_full))
        output["frame_idx"] = len(self.calls) - 1
        if self.empty:
            output["masks"][0].fill(False)
        return output

    def reset_tracking(self):
        self.events.append("reset")

    def close(self):
        self.events.append("live_close")


class FakePipeline:
    def __init__(self, live, *, copy_frames, fail=False, close_fail=False, stats_override=None):
        assert copy_frames is True
        self.live, self.fail, self.close_fail = live, fail, close_fail
        self.stats_override = stats_override or {}
        self.pending = None
        self.counters = {"submitted_frames": 0, "output_frames": 0, "dropped_frames": 0,
                         "failed_frames": 0, "aborted_frames": 0, "queued_frames": 0,
                         "drain_failed": False}

    def start(self):
        self.live.events.append("pipeline_start")

    def submit(self, frame, *, sequence, full_detection):
        assert self.pending is None, "Smoke must consume before submitting another frame"
        self.pending = (frame, sequence, full_detection)
        self.counters["submitted_frames"] += 1
        return True

    def infer_next(self):
        if self.pending is None:
            return None
        frame, sequence, request = self.pending
        self.pending = None
        if self.fail:
            self.counters["failed_frames"] += 1
            raise RuntimeError("synthetic inference failure")
        self.counters["output_frames"] += 1
        return SimpleNamespace(packet=SimpleNamespace(sequence=sequence),
                               output=self.live.infer(frame, full_detection=request))

    def finish_input(self):
        self.live.events.append("finish_input")

    def close(self):
        self.live.events.append("pipeline_close")
        if self.close_fail:
            raise RuntimeError("synthetic close failure")

    def stats(self):
        return {**self.counters, **self.stats_override}


def _runtime(*, force_full=False, empty=False, fail=False, close_fail=False, stats_override=None):
    events, lives = [], []

    def create_live(mode):
        live = FakeLive(events, force_full=force_full, empty=empty)
        lives.append(live)
        return live

    return SimpleNamespace(
        events=events, lives=lives, environment={"onnxruntime": "1.24.2"},
        create_live=create_live, synchronize=lambda: events.append("sync"),
        pipeline_class=lambda live, **kwargs: FakePipeline(live, **kwargs, fail=fail,
                                                           close_fail=close_fail, stats_override=stats_override),
    )


@pytest.mark.parametrize("mode,warmup", [("full", [True, True]), ("hybrid", [True, False])])
def test_serial_smoke_covers_warmup_reset_and_ordered_cleanup(mode, warmup):
    runtime = _runtime()
    result = smoke.run_mode(mode, [_frame() for _ in range(3)], "swan", runtime)
    assert result["passed"], result["errors"]
    assert runtime.lives[0].calls[:2] == warmup
    assert runtime.events.index("reset") < runtime.events.index("pipeline_start")
    assert runtime.events[-2:] == ["pipeline_close", "live_close"]
    assert result["frames_processed"] == 3 and result["nonempty_masks"] == 3
    assert result["lifecycle"]["close_order"] == ["pipeline", "live"]
    assert result["pipeline_stats"]["dropped_frames"] == 0
    assert not any(key in result for key in ("fps", "output_hz", "latency_ms"))
    assert result["propagated_frames"] == (1 if mode == "hybrid" else 0)


def test_hybrid_requires_real_propagation_warmup():
    runtime = _runtime(force_full=True)
    result = smoke.run_mode("hybrid", [_frame()] * 3, "swan", runtime)
    assert not result["passed"]
    assert "Warmup did not exercise" in result["errors"][0]["message"]
    assert "pipeline_start" not in runtime.events and runtime.events[-1] == "live_close"


def test_empty_masks_fail_known_positive_smoke():
    result = smoke.run_mode("full", [_frame()] * 3, "swan", _runtime(empty=True))
    assert not result["passed"] and "No non-empty masks" in result["errors"][0]["message"]


@pytest.mark.parametrize("options", [
    {"fail": True}, {"close_fail": True}, {"stats_override": {"drain_failed": True}},
    {"stats_override": {"aborted_frames": 1}}, {"stats_override": {"dropped_frames": 1}},
])
def test_failures_still_close_pipeline_then_live(options):
    runtime = _runtime(**options)
    result = smoke.run_mode("full", [_frame()] * 3, "swan", runtime)
    assert not result["passed"] and result["errors"]
    assert runtime.events[-2:] == ["pipeline_close", "live_close"]
    assert result["lifecycle"]["live_closed"] is True


def test_source_archive_reports_unknown_git_and_local_version(tmp_path):
    (tmp_path / "VERSION").write_text("0.2.0-rc1\n")
    identity = smoke.source_identity(tmp_path)
    assert identity["version"] == "0.2.0-rc1"
    assert identity["git_head"] is None and identity["git_diff_sha256"] is None


def test_artifact_identity_includes_installed_manifest(tmp_path):
    model, onnx = tmp_path / "model", tmp_path / "onnx"
    paths = [model / "model.safetensors", model / "config.json", tmp_path / "video.mp4",
             onnx / "backbone_detector/tuned_gpuio.mxr", onnx / "detr_decoder_fixed/direct_gpuio.mxr"]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    (onnx / "manifest.json").write_text(json.dumps({"format": "sam3-artifact-bundle", "release": "0.2.0-rc1"}))
    identity = smoke.artifact_identity(model, onnx, tmp_path / "video.mp4")
    assert identity["manifest_metadata"]["release"] == "0.2.0-rc1"
    assert len(identity["backbone"]["sha256"]) == 64


def test_main_writes_failure_json_before_loading_any_model(tmp_path, monkeypatch):
    def unavailable(_args):
        raise ValueError("MIGraphXExecutionProvider is required")

    monkeypatch.setattr(smoke, "source_identity", lambda: {"version": "0.2.0-rc1"})
    monkeypatch.setattr(smoke, "_load_runtime", unavailable)
    assert smoke.main(_arguments(tmp_path)) == 1
    result = json.loads((tmp_path / "smoke.json").read_text())
    assert result["kind"] == "installation_smoke" and result["passed"] is False
    assert result["modes"] == [] and "MIGraphX" in result["errors"][0]["message"]


def test_main_runs_both_modes_without_benchmark_metrics(tmp_path, monkeypatch):
    runtime = _runtime()
    monkeypatch.setattr(smoke, "source_identity", lambda: {"version": "0.2.0-rc1"})
    monkeypatch.setattr(smoke, "_load_runtime", lambda args: runtime)
    monkeypatch.setattr(smoke, "artifact_identity", lambda *args: {"fixture": True})
    monkeypatch.setattr(smoke, "read_frames", lambda video, count: [_frame()] * count)
    assert smoke.main(_arguments(tmp_path, "--frames", "3")) == 0
    result = json.loads((tmp_path / "smoke.json").read_text())
    assert result["passed"] and [row["mode"] for row in result["modes"]] == ["full", "hybrid"]
    assert result["kind"] == "installation_smoke" and len(runtime.lives) == 2
    assert "not a performance benchmark" in result["scope"]


def test_main_stops_after_failed_mode(tmp_path, monkeypatch):
    runtime = _runtime(fail=True)
    monkeypatch.setattr(smoke, "source_identity", lambda: {})
    monkeypatch.setattr(smoke, "_load_runtime", lambda args: runtime)
    monkeypatch.setattr(smoke, "artifact_identity", lambda *args: {})
    monkeypatch.setattr(smoke, "read_frames", lambda video, count: [_frame()] * count)
    assert smoke.main(_arguments(tmp_path, "--frames", "3")) == 1
    assert len(runtime.lives) == 1
