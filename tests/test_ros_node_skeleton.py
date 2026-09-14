from __future__ import annotations

import threading

import numpy as np
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
    assert node.published == []
    node.close()
