"""Frame-source contracts for offline prefetch; no GPU/model load required."""
from types import SimpleNamespace

import torch

from tracker.backbone_pipeline import BackbonePrefetchPipeline


def test_future_frame_access_does_not_advance_streaming_session():
    frames = {i: torch.full((3, 2, 2), float(i)) for i in range(3)}
    session = SimpleNamespace(processed_frames={})
    calls = []

    def model(*, inference_session, frame_idx, frame, reverse):
        assert inference_session is session
        assert len(session.processed_frames) == frame_idx
        assert frame is frames[frame_idx]
        session.processed_frames[frame_idx] = frame
        calls.append(frame_idx)
        return frame_idx

    pipeline = BackbonePrefetchPipeline.__new__(BackbonePrefetchPipeline)
    pipeline.inference_session = session
    pipeline.frame_provider = frames.__getitem__
    pipeline.model = model
    assert pipeline._get_frame(1) is frames[1]
    assert session.processed_frames == {}
    assert pipeline._forward(0, False) == 0
    assert pipeline._get_frame(2) is frames[2]
    assert list(session.processed_frames) == [0]
    assert pipeline._forward(1, False) == 1
    assert calls == [0, 1]


def test_legacy_preloaded_forward_does_not_supply_streaming_frame():
    frames = {i: torch.full((3, 2, 2), float(i)) for i in range(2)}
    session = SimpleNamespace(get_frame=frames.__getitem__)
    pipeline = BackbonePrefetchPipeline.__new__(BackbonePrefetchPipeline)
    pipeline.inference_session = session
    pipeline.frame_provider = None
    calls = []

    def model(**kwargs):
        calls.append(kwargs)
        return "output"

    pipeline.model = model
    assert pipeline._get_frame(1) is frames[1]
    assert pipeline._forward(1, True) == "output"
    assert calls == [{"inference_session": session, "frame_idx": 1, "reverse": True}]
