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


def test_hybrid_honors_full_detection_override(monkeypatch):
    live = SAM3HybridLive.__new__(SAM3HybridLive)
    live.imgsz = 4
    live.keyframe_interval_s = 1000.0
    live._last_keyframe_time = time.perf_counter()
    live._force_keyframe_next = False
    live._last_was_keyframe = False
    live._call_count = 0
    calls = []
    monkeypatch.setattr(
        hybrid_inference,
        "preprocess_image",
        lambda frame, imgsz: np.empty((3, imgsz, imgsz), dtype=np.float32),
    )
    live._keyframe_infer = lambda frame, image, height, width: calls.append(
        "keyframe"
    ) or {}
    live._propagation_infer = lambda image, height, width: calls.append(
        "propagation"
    ) or {}

    assert live.infer(_frame(1), full_detection=True)["keyframe"] is True
    live._last_keyframe_time = 0.0
    assert live.infer(_frame(2), full_detection=False)["keyframe"] is False
    assert calls == ["keyframe", "propagation"]


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
