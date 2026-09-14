from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tracker.hybrid_inference as hybrid_inference
from tracker.latest_frame import (
    LatestFramePipeline,
    LatestFramePipelineError,
    _LatestSlot,
)
from tracker.hybrid_inference import SAM3HybridLive
from tracker.live_inference import SAM3Live


class _FakeLive:
    def __init__(self, *, block_first: bool = False, fail: bool = False) -> None:
        self.device = torch.device("cpu")
        self.calls: list[tuple[np.ndarray, bool | None, int]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_first = block_first
        self.fail = fail

    def prepare_frame(self, *args, **kwargs):
        raise AssertionError("latest-frame scheduling must not prepare N+1")

    def _prefetch_prepared(self, *args, **kwargs):
        raise AssertionError("latest-frame scheduling must not prefetch N+1")

    def infer(self, frame_bgr: np.ndarray, *, full_detection=None) -> dict:
        call_index = len(self.calls)
        self.calls.append((frame_bgr.copy(), full_detection, threading.get_ident()))
        if call_index == 0:
            self.entered.set()
            if self.block_first:
                assert self.release.wait(timeout=2.0)
        if self.fail:
            raise RuntimeError("synthetic infer failure")
        return {"frame_idx": call_index, "value": int(frame_bgr[0, 0, 0])}


class _GuardedFakeLive(_FakeLive):
    def infer(self, frame_bgr: np.ndarray, *, full_detection=None) -> dict:
        active = getattr(self, "_latest_frame_pipeline_active", None)
        if active is not None and not active._is_inference_owner_thread():
            raise RuntimeError("external infer rejected")
        return super().infer(frame_bgr, full_detection=full_detection)


def _frame(value: int) -> np.ndarray:
    return np.full((4, 5, 3), value, dtype=np.uint8)


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition was not reached before timeout")
        time.sleep(0.001)


def test_latest_slot_replaces_only_unconsumed_item_and_reports_drop():
    dropped = []
    slot = _LatestSlot[int](on_drop=dropped.append)

    assert slot.put(1)
    assert slot.put(2)
    assert slot.dropped == 1
    assert dropped == [1]
    slot.close()
    assert slot.get() == 2
    assert slot.get() is None


def test_latest_slot_abort_discards_item_and_wakes_waiter():
    slot = _LatestSlot[int]()
    received = []

    waiter = threading.Thread(target=lambda: received.append(slot.get()))
    waiter.start()
    time.sleep(0.01)
    assert slot.close(discard=True) is None
    waiter.join(timeout=1.0)
    assert not waiter.is_alive()
    assert received == [None]

    occupied = _LatestSlot[int]()
    assert occupied.put(3)
    assert occupied.close(discard=True) == 3
    assert occupied.get() is None


def test_submit_owns_frame_and_latest_packet_metadata_stays_together():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()

    first = _frame(1)
    second = _frame(2)
    captured_at = time.perf_counter() - 0.01
    assert pipeline.submit(first, sequence=10, sensor_timestamp=1000)
    first.fill(99)
    assert pipeline.submit(
        second,
        sequence=11,
        captured_at=captured_at,
        sensor_timestamp=2000,
        full_detection=True,
        metadata={"pose": "pose-11"},
    )
    second.fill(88)
    pipeline.finish_input()

    result = pipeline.infer_next()
    assert result is not None
    assert result.packet.sequence == 11
    assert result.packet.sensor_timestamp == 2000
    assert result.packet.metadata == {"pose": "pose-11"}
    assert result.output == {"frame_idx": 0, "value": 2}
    assert result.queue_wait_ms >= 0.0
    assert result.service_ms >= 0.0
    assert result.age_ms == pytest.approx(
        result.queue_wait_ms + result.service_ms, abs=1e-6
    )
    assert pipeline.infer_next() is None
    assert pipeline.stats() == {
        "submitted_frames": 2,
        "output_frames": 1,
        "dropped_frames": 1,
        "failed_frames": 0,
        "aborted_frames": 0,
        "drain_failed": False,
        "queued_frames": 0,
    }
    pipeline.close()


def test_submit_copies_noncontiguous_input_to_owned_contiguous_storage():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    backing = np.full((4, 10, 3), 7, dtype=np.uint8)
    view = backing[:, ::2, :]
    assert not view.flags.c_contiguous
    assert pipeline.submit(view)
    backing.fill(99)
    pipeline.finish_input()
    assert pipeline.infer_next() is not None
    seen = live.calls[0][0]
    assert seen.flags.c_contiguous
    assert np.all(seen == 7)
    assert pipeline.infer_next() is None
    pipeline.close()


def test_copy_frames_false_requires_caller_owned_immutable_storage():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live, copy_frames=False)
    pipeline.start()
    frame = _frame(3)
    assert pipeline.submit(frame)
    frame.fill(4)
    pipeline.finish_input()
    assert pipeline.infer_next() is not None
    assert live.calls[0][0][0, 0, 0] == 4
    assert pipeline.infer_next() is None
    pipeline.close()


def test_no_nplus1_model_work_and_only_latest_runs_after_inflight_frame():
    live = _FakeLive(block_first=True)
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(0), sequence=0)

    results = []

    def consume_all():
        while True:
            result = pipeline.infer_next()
            if result is None:
                return
            results.append(result)

    consumer = threading.Thread(target=consume_all)
    consumer.start()
    assert live.entered.wait(timeout=1.0)

    assert pipeline.submit(_frame(1), sequence=1)
    assert pipeline.submit(_frame(2), sequence=2)
    assert pipeline.submit(_frame(3), sequence=3)
    # No preprocessing or model call for N+1 can happen while N is blocked.
    assert len(live.calls) == 1

    live.release.set()
    _wait_until(lambda: len(live.calls) == 2)
    pipeline.finish_input()
    consumer.join(timeout=2.0)
    assert not consumer.is_alive()

    assert [result.packet.sequence for result in results] == [0, 3]
    assert [call[0][0, 0, 0] for call in live.calls] == [0, 3]
    assert len({call[2] for call in live.calls}) == 1
    assert pipeline.stats() == {
        "submitted_frames": 4,
        "output_frames": 2,
        "dropped_frames": 2,
        "failed_frames": 0,
        "aborted_frames": 0,
        "drain_failed": False,
        "queued_frames": 0,
    }
    pipeline.close()


def test_finish_input_drains_last_frame_and_rejects_new_submit():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(7))
    pipeline.finish_input()
    assert not pipeline.submit(_frame(8))
    result = pipeline.infer_next()
    assert result is not None
    assert result.output["value"] == 7
    assert pipeline.infer_next() is None
    pipeline.close()
    pipeline.close()


@pytest.mark.parametrize("stop", ["finish", "abort", "close"])
def test_pipeline_stopped_before_start_cannot_start(stop):
    pipeline = LatestFramePipeline(_FakeLive())
    if stop == "finish":
        pipeline.finish_input()
    elif stop == "abort":
        pipeline.abort()
    else:
        pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        pipeline.start()


def test_abort_discards_latest_and_wakes_blocked_consumer():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    received = []
    consumer = threading.Thread(target=lambda: received.append(pipeline.infer_next()))
    consumer.start()
    time.sleep(0.01)
    pipeline.abort()
    consumer.join(timeout=1.0)
    assert not consumer.is_alive()
    assert received == [None]
    assert not pipeline.submit(_frame(1))
    pipeline.close()


def test_finish_input_wakes_blocked_consumer_without_aborting():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    received = []
    consumer = threading.Thread(target=lambda: received.append(pipeline.infer_next()))
    consumer.start()
    time.sleep(0.01)
    pipeline.finish_input()
    consumer.join(timeout=1.0)
    assert not consumer.is_alive()
    assert received == [None]
    assert pipeline.stats()["aborted_frames"] == 0
    pipeline.close()


def test_abort_after_dequeue_but_before_infer_discards_packet():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(5))
    pipeline._inference_lock.acquire()
    result = []
    consumer = threading.Thread(target=lambda: result.append(pipeline.infer_next()))
    try:
        consumer.start()
        _wait_until(lambda: pipeline.stats()["queued_frames"] == 0)
        pipeline.abort()
    finally:
        pipeline._inference_lock.release()
    consumer.join(timeout=1.0)
    assert not consumer.is_alive()
    assert result == [None]
    assert live.calls == []
    assert pipeline.stats()["aborted_frames"] == 1
    pipeline.close()


def test_abort_allows_inflight_result_but_never_starts_queued_frame():
    live = _FakeLive(block_first=True)
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(1), sequence=0)
    results = []
    consumer = threading.Thread(target=lambda: results.append(pipeline.infer_next()))
    consumer.start()
    assert live.entered.wait(timeout=1.0)
    try:
        assert pipeline.submit(_frame(2), sequence=1)
        pipeline.abort()
    finally:
        live.release.set()
    consumer.join(timeout=2.0)
    assert not consumer.is_alive()
    assert len(results) == 1
    assert results[0] is not None
    assert results[0].packet.sequence == 0
    assert len(live.calls) == 1
    stats = pipeline.stats()
    assert stats["output_frames"] == 1
    assert stats["aborted_frames"] == 1
    assert stats["queued_frames"] == 0
    pipeline.close()


def test_close_waits_for_inflight_inference_and_then_releases_live():
    live = _FakeLive(block_first=True)
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(4))

    consumer = threading.Thread(target=pipeline.infer_next)
    consumer.start()
    assert live.entered.wait(timeout=1.0)
    entered_close = [threading.Event(), threading.Event()]
    closed = [threading.Event(), threading.Event()]

    def close_pipeline(index):
        entered_close[index].set()
        pipeline.close()
        closed[index].set()

    closer = threading.Thread(target=close_pipeline, args=(0,))
    second_closer = threading.Thread(target=close_pipeline, args=(1,))
    closer.start()
    second_closer.start()
    assert entered_close[0].wait(timeout=1.0)
    assert entered_close[1].wait(timeout=1.0)
    time.sleep(0.02)
    assert not closed[0].is_set()
    assert not closed[1].is_set()
    live.release.set()
    consumer.join(timeout=2.0)
    closer.join(timeout=2.0)
    second_closer.join(timeout=2.0)
    assert not consumer.is_alive()
    assert not closer.is_alive()
    assert not second_closer.is_alive()
    assert closed[0].is_set()
    assert closed[1].is_set()
    assert not hasattr(live, "_latest_frame_pipeline_active")


def test_inference_failure_aborts_pipeline_and_is_sticky():
    live = _FakeLive(fail=True)
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(1))

    with pytest.raises(LatestFramePipelineError, match="latest-frame inference failed"):
        pipeline.infer_next()
    with pytest.raises(LatestFramePipelineError, match="synthetic infer failure"):
        pipeline.submit(_frame(2))
    with pytest.raises(LatestFramePipelineError, match="synthetic infer failure"):
        pipeline.infer_next()
    assert pipeline.stats()["failed_frames"] == 1
    pipeline.close()


def test_inference_failure_accounts_for_queued_latest_frame():
    live = _FakeLive(block_first=True, fail=True)
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(1), sequence=0)
    errors = []

    def consume():
        try:
            pipeline.infer_next()
        except BaseException as exc:
            errors.append(exc)

    consumer = threading.Thread(target=consume)
    consumer.start()
    assert live.entered.wait(timeout=1.0)
    try:
        assert pipeline.submit(_frame(2), sequence=1)
    finally:
        live.release.set()
    consumer.join(timeout=2.0)
    assert not consumer.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], LatestFramePipelineError)
    stats = pipeline.stats()
    assert stats["submitted_frames"] == 2
    assert stats["output_frames"] == 0
    assert stats["failed_frames"] == 1
    assert stats["aborted_frames"] == 1
    assert stats["queued_frames"] == 0
    pipeline.close()


def test_pipeline_constructed_before_poison_cannot_claim_session():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    live._latest_frame_pipeline_poisoned = True
    with pytest.raises(RuntimeError, match="failed latest-frame GPU drain"):
        pipeline.start()


def test_nested_poison_is_checked():
    inner = _FakeLive()
    outer = _FakeLive()
    outer.live = inner
    inner._latest_frame_pipeline_poisoned = True
    with pytest.raises(RuntimeError, match="failed latest-frame GPU drain"):
        LatestFramePipeline(outer)


def test_cuda_drain_failure_poisons_outer_and_nested_live(monkeypatch):
    inner = _FakeLive(fail=True)
    outer = _FakeLive(fail=True)
    outer.live = inner
    outer.device = torch.device("cuda")

    def fail_sync(*args, **kwargs):
        raise RuntimeError("synthetic drain failure")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    pipeline = LatestFramePipeline(outer)
    pipeline.start()
    assert pipeline.submit(_frame(1))
    with pytest.raises(LatestFramePipelineError):
        pipeline.infer_next()
    assert pipeline.stats()["drain_failed"] is True
    assert outer._latest_frame_pipeline_poisoned is True
    assert inner._latest_frame_pipeline_poisoned is True
    pipeline.close()
    with pytest.raises(RuntimeError, match="failed latest-frame GPU drain"):
        LatestFramePipeline(inner)


def test_source_sequence_must_be_monotonic_and_input_is_validated():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(1), sequence=4)
    with pytest.raises(ValueError, match="monotonic"):
        pipeline.submit(_frame(2), sequence=4)
    with pytest.raises(ValueError, match="uint8"):
        pipeline.submit(_frame(2).astype(np.float32), sequence=5)
    with pytest.raises(ValueError, match="finite"):
        pipeline.submit(_frame(2), sequence=5, captured_at=float("nan"))
    with pytest.raises(ValueError, match="past host-monotonic"):
        pipeline.submit(_frame(2), sequence=5, captured_at=time.perf_counter() + 1.0)
    with pytest.raises(TypeError, match="sequence"):
        pipeline.submit(_frame(2), sequence=5.0)
    with pytest.raises(TypeError, match="full_detection"):
        pipeline.submit(_frame(2), sequence=5, full_detection=1)
    pipeline.close()


def test_forced_detection_request_survives_latest_frame_replacement():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(1), sequence=0, full_detection=True)
    assert pipeline.submit(_frame(2), sequence=1, full_detection=False)
    pipeline.finish_input()
    result = pipeline.infer_next()
    assert result is not None
    assert result.packet.sequence == 1
    assert live.calls[0][1] is True
    assert pipeline.infer_next() is None
    pipeline.close()


def test_forced_detection_latch_is_cleared_after_one_inference():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(1), full_detection=True)
    assert pipeline.infer_next() is not None
    assert pipeline.submit(_frame(2), full_detection=False)
    pipeline.finish_input()
    assert pipeline.infer_next() is not None
    assert [call[1] for call in live.calls] == [True, False]
    assert pipeline.infer_next() is None
    pipeline.close()


def test_infer_next_is_bound_to_one_consumer_thread():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(1))
    assert pipeline.infer_next() is not None
    assert pipeline.submit(_frame(2))

    errors = []

    def consume_from_second_thread():
        try:
            pipeline.infer_next()
        except BaseException as exc:
            errors.append(exc)

    second = threading.Thread(target=consume_from_second_thread)
    second.start()
    second.join(timeout=1.0)
    assert not second.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "one consumer thread" in str(errors[0])
    pipeline.abort()
    pipeline.close()


def test_first_consumer_claim_is_atomic():
    live = _FakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    assert pipeline.submit(_frame(1))
    barrier = threading.Barrier(3)
    results = []
    errors = []

    def compete():
        barrier.wait()
        try:
            results.append(pipeline.infer_next())
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=compete)
    second = threading.Thread(target=compete)
    first.start()
    second.start()
    barrier.wait()
    first.join(timeout=1.0)
    second.join(timeout=1.0)
    if first.is_alive() or second.is_alive():
        pipeline.finish_input()
        first.join(timeout=1.0)
        second.join(timeout=1.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert len(results) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "one consumer thread" in str(errors[0])
    pipeline.close()


def test_one_live_instance_allows_only_one_pipeline():
    live = _FakeLive()
    first = LatestFramePipeline(live)
    second = LatestFramePipeline(live)
    first.start()
    with pytest.raises(RuntimeError, match="already active"):
        second.start()
    first.close()
    second.start()
    second.close()


def test_concurrent_start_claims_live_once():
    live = _FakeLive()
    pipelines = [LatestFramePipeline(live), LatestFramePipeline(live)]
    barrier = threading.Barrier(3)
    started = []
    errors = []

    def start_pipeline(pipeline):
        barrier.wait()
        try:
            pipeline.start()
            started.append(pipeline)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=start_pipeline, args=(p,)) for p in pipelines]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=1.0)
        assert not thread.is_alive()
    assert len(started) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    started[0].close()


def test_active_latest_frame_pipeline_blocks_external_session_mutation():
    live = SAM3Live.__new__(SAM3Live)
    live._latest_frame_pipeline_active = SimpleNamespace()

    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.set_prompts(["new"])
    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.reset_prompts(["new"])
    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.reset_tracking()
    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.close()
    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.infer(_frame(1))


def test_active_latest_frame_pipeline_blocks_hybrid_session_mutation():
    live = SAM3HybridLive.__new__(SAM3HybridLive)
    live._latest_frame_pipeline_active = SimpleNamespace()

    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.reset_prompts(["new"])
    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.reset_tracking()
    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.close()
    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.infer(_frame(1))


def test_direct_sam3live_negative_evidence_matches_detector_execution():
    class _Model:
        _skip_detection = False

        def __call__(self, *, inference_session, frame, frame_idx):
            return SimpleNamespace(
                frame_idx=frame_idx,
                obj_id_to_tracker_score={},
                obj_id_to_mask={},
                obj_id_to_score={},
                suppressed_obj_ids=set(),
            )

    live = SAM3Live.__new__(SAM3Live)
    live.processor = SimpleNamespace(
        video_processor=lambda **_kwargs: SimpleNamespace(
            pixel_values_videos=torch.zeros((1, 1, 3, 4, 5)),
        ),
        postprocess_outputs=lambda **_kwargs: {
            "object_ids": torch.empty(0, dtype=torch.int64),
            "scores": torch.empty(0),
            "prompt_to_obj_ids": {},
        },
    )
    live.device = torch.device("cpu")
    live.dtype = torch.float32
    live.model = _Model()
    live.session = object()
    live._force_detect_next = True
    live._infer_calls = 0
    live.redetect_every = 1
    live._next_frame_idx = 0
    live.bootstrap_frames = 0
    live.max_objects_per_prompt = None
    live.keep_recent_frames = 0
    live._drift_enabled = False

    forced_detection = live.infer(_frame(1), full_detection=False)
    propagation = live.infer(_frame(2), full_detection=False)

    assert forced_detection["detected"] is True
    assert forced_detection["negative_evidence_valid"] is True
    assert propagation["detected"] is False
    assert propagation["negative_evidence_valid"] is False


class _FakeUnifiedLive:
    def __init__(self, outputs):
        self.device = torch.device("cpu")
        self.outputs = list(outputs)
        self.infer_calls = []
        self.reset_prompts_calls = []
        self.reset_tracking_calls = 0
        self.fresh_session_calls = 0
        self.close_calls = 0
        self._infer_calls = 0

    def infer(self, _frame, *, full_detection):
        self.infer_calls.append(full_detection)
        self._infer_calls += 1
        next_output = self.outputs.pop(0)
        if isinstance(next_output, dict):
            result = dict(next_output)
            result["object_ids"] = list(next_output["object_ids"])
            result["scores"] = dict(next_output.get("scores", {}))
            result["masks"] = dict(next_output.get("masks", {}))
            result["boxes"] = dict(next_output.get("boxes", {}))
            result["prompt_to_obj_ids"] = {
                prompt: list(object_ids)
                for prompt, object_ids in next_output.get(
                    "prompt_to_obj_ids", {}
                ).items()
            }
            result["detected"] = bool(next_output.get("detected", full_detection))
            result["negative_evidence_valid"] = bool(
                next_output.get("negative_evidence_valid", result["detected"])
            )
            return result
        object_ids = list(next_output)
        return {
            "object_ids": object_ids,
            "scores": {object_id: 0.9 for object_id in object_ids},
            "masks": {},
            "boxes": {},
            "prompt_to_obj_ids": {"object": object_ids},
            "frame_idx": len(self.infer_calls) - 1,
            "detected": full_detection,
            "negative_evidence_valid": full_detection,
        }

    def reset_prompts(self, prompts):
        self.reset_prompts_calls.append(list(prompts))

    def reset_tracking(self):
        self.reset_tracking_calls += 1
        self._infer_calls = 0

    def _replace_tracking_session_preserving_prompts(self):
        self.fresh_session_calls += 1
        self._infer_calls = 0

    def close(self):
        self.close_calls += 1


def _unified_hybrid(outputs):
    hybrid = SAM3HybridLive.__new__(SAM3HybridLive)
    hybrid.live = _FakeUnifiedLive(outputs)
    hybrid.device = hybrid.live.device
    hybrid.keyframe_interval_s = 1.0
    hybrid.iou_thresh = 0.3
    hybrid.max_per_prompt = 5
    hybrid._call_count = 0
    hybrid._last_keyframe_time = 0.0
    hybrid._last_was_keyframe = False
    hybrid._force_keyframe_next = True
    hybrid._pending_redetect_reason = "first_frame"
    hybrid._previous_output_object_ids = set()
    hybrid._previous_public_masks = {}
    hybrid._previous_public_prompts = {}
    hybrid._inner_to_public = {}
    hybrid._next_public_object_id = 0
    hybrid._inner_session_fresh = True
    return hybrid


def test_hybrid_constructs_only_one_sam3live_backend(monkeypatch):
    captured = {}

    class FakeConstructedLive:
        device = torch.device("cpu")

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(hybrid_inference, "SAM3Live", FakeConstructedLive)

    live = SAM3HybridLive(
        checkpoint="checkpoint",
        prompts=["floor", "wall"],
        onnx_dir="onnx",
        iou_assoc_threshold=0.42,
        bootstrap_frames=3,
    )

    assert type(live.live) is FakeConstructedLive
    assert not hasattr(live, "shared")
    assert not hasattr(live, "trackers")
    assert live.iou_thresh == pytest.approx(0.42)
    assert captured["prompts"] == ["floor", "wall"]
    assert captured["bootstrap_frames"] == 3


def test_hybrid_first_frame_and_wall_clock_schedule(monkeypatch):
    live = _unified_hybrid([[1], [1], [1]])
    clock = iter((10.0, 10.9, 11.01))
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: next(clock))

    first = live.infer(_frame(1), full_detection=False)
    before_interval = live.infer(_frame(2))
    at_interval = live.infer(_frame(3))

    assert live.live.infer_calls == [True, False, True]
    assert first["keyframe"] is True
    assert first["redetect_reason"] == "first_frame"
    assert first["negative_evidence_valid"] is True
    assert before_interval["keyframe"] is False
    assert before_interval["redetect_reason"] is None
    assert before_interval["negative_evidence_valid"] is False
    assert at_interval["keyframe"] is True
    assert at_interval["redetect_reason"] == "interval"
    assert at_interval["negative_evidence_valid"] is True
    assert live._last_keyframe_time == 11.01


def test_hybrid_explicit_detection_override(monkeypatch):
    live = _unified_hybrid([[1], [1]])
    live._force_keyframe_next = False
    live._pending_redetect_reason = None
    live._last_keyframe_time = 10.0
    clock = iter((10.2, 12.0))
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: next(clock))

    forced = live.infer(_frame(1), full_detection=True)
    skipped = live.infer(_frame(2), full_detection=False)

    assert live.live.infer_calls == [True, False]
    assert forced["redetect_reason"] == "caller_override"
    assert skipped["redetect_reason"] is None


def test_hybrid_uses_inner_actual_detected_state(monkeypatch):
    live = _unified_hybrid([
        {
            "object_ids": [2],
            "detected": True,
            "negative_evidence_valid": False,
        },
    ])
    live._force_keyframe_next = False
    live._pending_redetect_reason = None
    live._last_keyframe_time = 9.0
    live._previous_output_object_ids = {1}
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: 10.0)

    result = live.infer(_frame(1), full_detection=False)

    assert live.live.infer_calls == [False]
    assert result["detected"] is True
    assert result["keyframe"] is True
    assert result["redetect_reason"] == "inner_forced"
    assert result["negative_evidence_valid"] is True
    assert result["lost_object_ids"] == []
    assert live._last_keyframe_time == 10.0
    assert live._force_keyframe_next is False


def test_hybrid_requested_detection_retries_when_inner_did_not_detect(monkeypatch):
    live = _unified_hybrid([
        {"object_ids": [], "detected": False},
        {"object_ids": [4], "detected": True},
    ])
    clock = iter((10.0, 10.1))
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: next(clock))

    missed = live.infer(_frame(1))
    retry = live.infer(_frame(2), full_detection=False)

    assert missed["detected"] is False
    assert missed["keyframe"] is False
    assert missed["redetect_reason"] is None
    assert missed["negative_evidence_valid"] is False
    assert retry["detected"] is True
    assert retry["keyframe"] is True
    assert retry["redetect_reason"] == "detection_retry"
    assert retry["negative_evidence_valid"] is True
    assert live.live.infer_calls == [True, True]


def test_hybrid_object_loss_sticky_forces_next_detection(monkeypatch):
    live = _unified_hybrid([[2, 9, 5], [5], [5, 11]])
    clock = iter((10.0, 10.1, 10.2))
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: next(clock))

    first = live.infer(_frame(1))
    lost = live.infer(_frame(2), full_detection=False)
    recovery = live.infer(_frame(3), full_detection=False)

    assert first["lost_object_ids"] == []
    assert lost["keyframe"] is False
    assert lost["lost_object_ids"] == [0, 1]
    assert lost["redetect_reason"] is None
    assert recovery["keyframe"] is True
    assert recovery["redetect_reason"] == "object_loss"
    assert live.live.infer_calls == [True, False, True]


def _hybrid_output(object_ids, prompt_ids, masks, *, detected):
    return {
        "object_ids": list(object_ids),
        "scores": {object_id: 0.9 for object_id in object_ids},
        "masks": dict(masks),
        "boxes": {object_id: (0.0, 0.0, 1.0, 1.0) for object_id in object_ids},
        "prompt_to_obj_ids": {
            prompt: list(ids) for prompt, ids in prompt_ids.items()
        },
        "frame_idx": 0,
        "detected": detected,
    }


def test_hybrid_clean_keyframes_preserve_public_id_by_prompt_iou(monkeypatch):
    mask = np.array([[True, True], [False, False]])
    live = _unified_hybrid([
        _hybrid_output([10], {"object": [10]}, {10: mask}, detected=True),
        _hybrid_output([0], {"object": [0]}, {0: mask.copy()}, detected=True),
    ])
    clock = iter((10.0, 11.1))
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: next(clock))

    first = live.infer(_frame(1))
    second = live.infer(_frame(2))

    assert first["object_ids"] == [0]
    assert second["object_ids"] == [0]
    assert second["prompt_to_obj_ids"] == {"object": [0]}
    assert live.live.fresh_session_calls == 1
    assert live.live._infer_calls == 2
    assert live._next_public_object_id == 1


def test_hybrid_clean_keyframe_retires_scene_cut_object(monkeypatch):
    old_mask = np.array([[True, False], [False, False]])
    new_mask = np.array([[False, False], [False, True]])
    live = _unified_hybrid([
        _hybrid_output([7], {"object": [7]}, {7: old_mask}, detected=True),
        _hybrid_output([0], {"object": [0]}, {0: new_mask}, detected=True),
    ])
    clock = iter((10.0, 11.1))
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: next(clock))

    first = live.infer(_frame(1))
    second = live.infer(_frame(2))

    assert first["object_ids"] == [0]
    assert second["object_ids"] == [1]
    assert 0 not in second["scores"]
    assert 0 not in live._previous_public_masks


def test_hybrid_keyframe_matching_never_crosses_prompts(monkeypatch):
    left = np.array([[True, False], [False, False]])
    right = np.array([[False, False], [False, True]])
    live = _unified_hybrid([
        _hybrid_output(
            [10, 11],
            {"floor": [10], "wall": [11]},
            {10: left, 11: right},
            detected=True,
        ),
        _hybrid_output(
            [0, 1],
            {"floor": [0], "wall": [1]},
            {0: right.copy(), 1: left.copy()},
            detected=True,
        ),
    ])
    clock = iter((10.0, 11.1))
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: next(clock))

    first = live.infer(_frame(1))
    second = live.infer(_frame(2))

    assert first["prompt_to_obj_ids"] == {"floor": [0], "wall": [1]}
    assert second["prompt_to_obj_ids"] == {"floor": [2], "wall": [3]}


def test_hybrid_propagation_reuses_inner_to_public_mapping(monkeypatch):
    mask = np.ones((2, 2), dtype=bool)
    live = _unified_hybrid([
        _hybrid_output([12], {"object": [12]}, {12: mask}, detected=True),
        _hybrid_output([12], {"object": [12]}, {12: mask}, detected=False),
    ])
    clock = iter((10.0, 10.1))
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: next(clock))

    keyframe = live.infer(_frame(1))
    propagation = live.infer(_frame(2), full_detection=False)

    assert keyframe["object_ids"] == [0]
    assert propagation["object_ids"] == [0]
    assert propagation["prompt_to_obj_ids"] == {"object": [0]}
    assert propagation["lost_object_ids"] == []
    assert live.live.fresh_session_calls == 0


@pytest.mark.parametrize(
    ("method_name", "reason", "attribute"),
    [
        ("reset_prompts", "reset_prompts", "reset_prompts_calls"),
        ("reset_tracking", "reset_tracking", "reset_tracking_calls"),
    ],
)
def test_hybrid_reset_delegates_and_forces_fresh_detection(
    monkeypatch,
    method_name,
    reason,
    attribute,
):
    live = _unified_hybrid([[1], [2]])
    clock = iter((10.0, 10.1))
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: next(clock))
    live.infer(_frame(1))

    if method_name == "reset_prompts":
        live.reset_prompts(["new"])
        assert getattr(live.live, attribute) == [["new"]]
    else:
        live.reset_tracking()
        assert getattr(live.live, attribute) == 1

    result = live.infer(_frame(2), full_detection=False)
    assert result["keyframe"] is True
    assert result["redetect_reason"] == reason
    assert result["lost_object_ids"] == []


def test_hybrid_close_delegates():
    live = _unified_hybrid([])
    live.close()
    assert live.live.close_calls == 1


def test_hybrid_pipeline_owner_may_infer(monkeypatch):
    live = _unified_hybrid([[1]])
    pipeline = SimpleNamespace(_is_inference_owner_thread=lambda: True)
    live._latest_frame_pipeline_active = pipeline
    live.live._latest_frame_pipeline_active = pipeline
    monkeypatch.setattr(hybrid_inference.time, "perf_counter", lambda: 10.0)

    result = live.infer(_frame(1))

    assert result["keyframe"] is True
    assert live.live.infer_calls == [True]


def test_hybrid_rejects_inner_only_foreign_pipeline_before_session_reset():
    live = _unified_hybrid([[1]])
    live._force_keyframe_next = True
    live._inner_session_fresh = False
    live.live._latest_frame_pipeline_active = SimpleNamespace(
        _is_inference_owner_thread=lambda: False
    )

    with pytest.raises(RuntimeError, match="LatestFramePipeline"):
        live.infer(_frame(1))

    assert live.live.fresh_session_calls == 0
    assert live.live.infer_calls == []


def test_nested_live_session_is_claimed_and_released():
    inner = _FakeLive()
    outer = _FakeLive()
    outer.live = inner
    pipeline = LatestFramePipeline(outer)
    pipeline.start()
    assert outer._latest_frame_pipeline_active is pipeline
    assert inner._latest_frame_pipeline_active is pipeline
    pipeline.close()
    assert not hasattr(outer, "_latest_frame_pipeline_active")
    assert not hasattr(inner, "_latest_frame_pipeline_active")


def test_pipeline_owner_guard_allows_only_pipeline_inference():
    live = _GuardedFakeLive()
    pipeline = LatestFramePipeline(live)
    pipeline.start()
    with pytest.raises(RuntimeError, match="external infer rejected"):
        live.infer(_frame(0))
    assert pipeline.submit(_frame(1))
    pipeline.finish_input()
    result = pipeline.infer_next()
    assert result is not None
    assert result.output["value"] == 1
    assert pipeline.infer_next() is None
    pipeline.close()


def test_context_manager_exception_releases_live_claim():
    live = _FakeLive()
    with pytest.raises(RuntimeError, match="body failed"):
        with LatestFramePipeline(live):
            assert hasattr(live, "_latest_frame_pipeline_active")
            raise RuntimeError("body failed")
    assert not hasattr(live, "_latest_frame_pipeline_active")
