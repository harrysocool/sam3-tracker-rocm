from __future__ import annotations

import threading
import sys

import numpy as np
import pytest
import torch

import examples.ros_node_skeleton as ros_example


class _FakeNodeLive:
    instances = []

    def __init__(self, *args, **kwargs):
        self.device = torch.device("cpu")
        self.init_kwargs = dict(kwargs)
        self.frames = []
        self.reset_calls = []
        self.reset_tracking_calls = 0
        self.closed = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.instances.append(self)

    def infer(self, frame_bgr, *, full_detection=None):
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        self.frames.append((frame_bgr.copy(), full_detection))
        return {
            "object_ids": [],
            "scores": {},
            "masks": {},
            "boxes": {},
            "prompt_to_obj_ids": {},
            "frame_idx": len(self.frames) - 1,
            "detected": bool(full_detection),
        }

    def reset_prompts(self, prompts):
        self.reset_calls.append(list(prompts))

    def reset_tracking(self):
        self.reset_tracking_calls += 1

    def close(self):
        self.closed = True


class _RecordingNode(ros_example.SAM3Node):
    def __init__(self, *args, **kwargs):
        self.published = []
        super().__init__(*args, **kwargs)

    def _publish_masks(self, item, *, generation):
        self.published.append((generation, item))


def _frame(value: int) -> np.ndarray:
    return np.full((4, 5, 3), value, dtype=np.uint8)


def test_ros_callback_uses_owned_latest_frame_and_finishes(monkeypatch):
    _FakeNodeLive.instances.clear()
    monkeypatch.setattr(ros_example, "SAM3Live", _FakeNodeLive)
    node = _RecordingNode(
        checkpoint="unused",
        onnx_dir="unused",
        prompts=["obstacle"],
        policy=ros_example.AlwaysFull(),
        max_result_age_ms=None,
    )
    live = _FakeNodeLive.instances[-1]
    assert live.init_kwargs["parallel_tail"] is True
    assert live.init_kwargs["fixed_detr_decoder"] is None
    frame = _frame(1)
    assert node.on_image(frame, header_stamp_ns=123)
    assert live.entered.wait(timeout=1.0)
    frame.fill(9)
    assert node.on_image(_frame(2), header_stamp_ns=456)
    live.release.set()
    node.finish()
    stats = node.stats()
    assert stats["callbacks"] == 2
    assert stats["accepted_callbacks"] == 2
    assert stats["output_frames"] >= 1
    assert stats["completed_frames"] == stats["output_frames"]
    assert stats["completed_service_ms"] == stats["service_ms"]
    assert stats["completed_age_ms"] == stats["age_ms"]
    assert live.frames[0][0][0, 0, 0] == 1
    assert live.frames[-1][0][0, 0, 0] == 2
    assert live.frames[-1][1] is True
    assert node.published[-1][1].packet.sensor_timestamp == 456
    node.close()
    assert live.closed


def test_ros_non_mig_default_is_serial(monkeypatch):
    _FakeNodeLive.instances.clear()
    monkeypatch.setattr(ros_example, "SAM3Live", _FakeNodeLive)
    node = _RecordingNode(
        checkpoint="unused",
        onnx_dir="unused",
        prompts=["obstacle"],
        mig=False,
    )
    live = _FakeNodeLive.instances[-1]
    assert live.init_kwargs["parallel_tail"] is False
    node.close()


def test_prompt_reset_uses_new_pipeline_generation(monkeypatch):
    _FakeNodeLive.instances.clear()
    monkeypatch.setattr(ros_example, "SAM3Live", _FakeNodeLive)
    node = _RecordingNode(
        checkpoint="unused",
        onnx_dir="unused",
        prompts=["old"],
    )
    live = _FakeNodeLive.instances[-1]
    live.release.set()
    assert node.on_image(_frame(1))
    node.reset_prompts(["new"])
    assert live.reset_calls == [["new"]]
    assert node.on_image(_frame(2))
    node.finish()
    generations = node.stats()["pipeline_generations"]
    assert [entry["generation"] for entry in generations] == [0, 1]
    node.close()


def test_tracking_reset_uses_new_pipeline_generation(monkeypatch):
    _FakeNodeLive.instances.clear()
    monkeypatch.setattr(ros_example, "SAM3Live", _FakeNodeLive)
    node = _RecordingNode(
        checkpoint="unused",
        onnx_dir="unused",
        prompts=["obstacle"],
    )
    live = _FakeNodeLive.instances[-1]
    live.release.set()
    assert node.on_image(_frame(1))
    node.reset_tracking()
    assert live.reset_tracking_calls == 1
    assert node.on_image(_frame(2))
    node.finish()
    assert [entry["generation"] for entry in node.stats()["pipeline_generations"]] == [0, 1]
    node.close()


def test_stale_result_is_not_published(monkeypatch):
    _FakeNodeLive.instances.clear()
    monkeypatch.setattr(ros_example, "SAM3Live", _FakeNodeLive)
    node = _RecordingNode(
        checkpoint="unused",
        onnx_dir="unused",
        prompts=["obstacle"],
        max_result_age_ms=1.0,
    )
    live = _FakeNodeLive.instances[-1]
    assert node.on_image(_frame(1))
    assert live.entered.wait(timeout=1.0)
    threading.Event().wait(0.01)
    live.release.set()
    node.finish()
    stats = node.stats()
    assert stats["stale_results"] == 1
    assert stats["output_frames"] == 0
    assert stats["completed_frames"] == 1
    assert len(stats["completed_service_ms"]) == len(stats["completed_age_ms"]) == 1
    assert stats["completed_age_ms"][0] > 1.0
    assert stats["service_ms"] == stats["age_ms"] == []
    assert stats["superseded_results"] == 0
    assert node.published == []
    node.close()


def _args(*flags):
    return ["--checkpoint", "unused", "--video", "unused.mp4", "--text", "swan", *flags]


@pytest.mark.parametrize(("flags", "expected"), [
    ([], 5), (["--max-objects", "0"], 0),
    (["--max-objects", "1"], 1), (["--max-objects", "-1"], 5),
])
def test_ros_cli_object_cap_values(flags, expected):
    assert ros_example.parse_args(_args(*flags)).max_objects == expected


def test_ros_cli_rejects_invalid_object_cap(capsys):
    with pytest.raises(SystemExit) as exc:
        ros_example.parse_args(_args("--max-objects", "-2"))
    assert exc.value.code == 2
    assert "--max-objects must be non-negative" in capsys.readouterr().err


@pytest.mark.parametrize(("value", "expected"), [("0", None), ("1", 1), ("-1", 5)])
def test_ros_main_translates_cli_cap_before_model_loading(monkeypatch, value, expected):
    class StopBeforeLoading(Exception):
        pass

    captured = {}

    def node(**kwargs):
        captured.update(kwargs)
        raise StopBeforeLoading

    monkeypatch.setattr(sys, "argv", ["ros_node_skeleton.py", *_args("--max-objects", value)])
    monkeypatch.setattr(ros_example, "SAM3Node", node)
    with pytest.raises(StopBeforeLoading):
        ros_example.main()
    assert captured["max_objects_per_prompt"] == expected


@pytest.mark.parametrize("cap", [None, 0, 1])
def test_ros_python_api_keeps_its_existing_cap_semantics(monkeypatch, cap):
    monkeypatch.setattr(ros_example, "SAM3Live", _FakeNodeLive)
    node = _RecordingNode(checkpoint="unused", prompts=["swan"],
                          max_objects_per_prompt=cap)
    try:
        assert node.live.init_kwargs["max_objects_per_prompt"] is cap
    finally:
        node.close()


def test_superseded_completion_is_counted_but_not_published(monkeypatch):
    monkeypatch.setattr(ros_example, "SAM3Live", _FakeNodeLive)
    node = _RecordingNode(checkpoint="unused", prompts=["swan"])
    pipeline = None
    try:
        assert node.on_image(_frame(1))
        assert node.live.entered.wait(timeout=1.0)
        pipeline, consumer, generation = node._detach_generation()
        node.live.release.set()
        pipeline.close()
        node._join_consumer(consumer)
        node._record_pipeline_stats(generation, pipeline)
        stats = node.stats()
        assert stats["completed_frames"] == stats["superseded_results"] == 1
        assert stats["stale_results"] == stats["output_frames"] == 0
        assert len(stats["completed_service_ms"]) == 1
        assert stats["service_ms"] == []
        assert stats["pipeline_generations"][0]["output_frames"] == 1
        assert node.published == []
    finally:
        node.live.release.set()
        if pipeline is not None:
            pipeline.close()
        node.close()


@pytest.mark.parametrize("published", [False, True])
def test_summary_labels_completed_and_published_populations(capsys, published):
    stats = {
        "accepted_callbacks": 3, "rejected_callbacks": 1,
        "completed_frames": 3, "output_frames": int(published),
        "stale_results": 2 - int(published), "superseded_results": 1,
        "completed_service_ms": [10.0, 200.0, 300.0],
        "completed_age_ms": [12.0, 220.0, 330.0],
        "service_ms": [10.0] if published else [],
        "age_ms": [12.0] if published else [],
        "pipeline_generations": [],
    }
    ros_example._print_summary(stats, 4)
    output = capsys.readouterr().out
    assert "completed=3" in output
    assert f"published={int(published)}" in output
    assert f"age_rejected={2 - int(published)}" in output
    assert "superseded=1 callback_rejected=1" in output
    assert "completed service: mean=170.0 ms" in output
    assert "completed age:" in output
    assert ("published service:" in output) is published
    if published:
        assert "published service: mean=10.0 ms" in output
    assert "nan" not in output.lower()


def test_empty_summary_has_no_timing_statistics(capsys):
    stats = {key: 0 for key in ("accepted_callbacks", "rejected_callbacks",
             "completed_frames", "output_frames", "stale_results", "superseded_results")}
    stats.update({key: [] for key in ("completed_service_ms", "completed_age_ms",
                  "service_ms", "age_ms", "pipeline_generations")})
    ros_example._print_summary(stats, 0)
    output = capsys.readouterr().out
    assert "completed=0 published=0 age_rejected=0" in output
    assert "mean=" not in output
