"""Bounded latest-frame scheduling for live SAM3 inference.

The capture producer may run continuously, but the pipeline retains only the
newest not-yet-consumed frame.  Preprocessing, the vision backbone, and the
stateful detector/tracker path all begin only after the preceding inference
has completed.  In particular, this module never reads or computes frame N+1
on the GPU while frame N is being inferred.

The topology is deliberately small::

    capture -> owned raw latest(1) -> ordered SAM3Live.infer

This trades output cadence for lower source-to-result age.  ``SAM3Live`` has a
single inference owner, and stale frames are discarded before any model or
session state is mutated.
"""
from __future__ import annotations

import math
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

import numpy as np
import torch

from .live_inference import SAM3Live


T = TypeVar("T")
_LIVE_CLAIM_LOCK = threading.Lock()


class LatestFramePipelineError(RuntimeError):
    """A latest-frame inference failed and the pipeline was aborted."""


class _LatestSlot(Generic[T]):
    """Capacity-one slot where a newer item replaces an unconsumed item."""

    def __init__(self, on_drop: Callable[[T], None] | None = None) -> None:
        self._condition = threading.Condition()
        self._item: T | None = None
        self._closed = False
        self._on_drop = on_drop
        self._dropped = 0

    @property
    def dropped(self) -> int:
        with self._condition:
            return self._dropped

    @property
    def occupied(self) -> bool:
        with self._condition:
            return self._item is not None

    def put(self, item: T) -> bool:
        dropped_item = None
        with self._condition:
            if self._closed:
                return False
            if self._item is not None:
                self._dropped += 1
                dropped_item = self._item
            self._item = item
            self._condition.notify_all()
        if dropped_item is not None and self._on_drop is not None:
            self._on_drop(dropped_item)
        return True

    def get(self) -> T | None:
        with self._condition:
            while self._item is None and not self._closed:
                self._condition.wait()
            if self._item is None:
                return None
            item = self._item
            self._item = None
            return item

    def close(self, *, discard: bool = False) -> T | None:
        """Close the slot, optionally removing and returning its item."""
        with self._condition:
            self._closed = True
            item = self._item if discard else None
            if discard:
                self._item = None
            self._condition.notify_all()
            return item


@dataclass(slots=True)
class LiveFramePacket:
    """One owned source frame and its capture-time identity.

    ``captured_at`` must use the same monotonic clock as ``time.perf_counter``
    because it is used for age accounting.  A camera/ROS/PTP timestamp belongs
    in ``sensor_timestamp`` and must be associated with the robot pose at
    exposure time by the downstream map updater.
    """

    sequence: int
    captured_at: float
    frame_bgr: np.ndarray
    sensor_timestamp: int | float | None = None
    full_detection: bool | None = None
    metadata: Any = None


@dataclass(slots=True)
class LatestFrameResult:
    """One ordered result with source identity and latency measurements."""

    packet: LiveFramePacket
    output: dict
    inference_started_at: float
    completed_at: float

    @property
    def queue_wait_ms(self) -> float:
        return (self.inference_started_at - self.packet.captured_at) * 1000.0

    @property
    def age_ms(self) -> float:
        return (self.completed_at - self.packet.captured_at) * 1000.0

    @property
    def service_ms(self) -> float:
        return (self.completed_at - self.inference_started_at) * 1000.0


@dataclass(slots=True)
class _Failure:
    stage: str
    exception: BaseException
    traceback: str


class LatestFramePipeline:
    """Latest-frame front end for one caller-owned ``SAM3Live`` session.

    ``submit`` never waits for inference; it replaces any unconsumed frame in
    the one-slot queue.  By default it copies the ndarray so camera, ROS, or
    GStreamer buffer reuse cannot corrupt queued input.  Set ``copy_frames``
    to ``False`` only when the caller transfers immutable ownership until the
    frame is consumed or dropped.

    ``infer_next`` must always be called from one consumer thread.  It performs
    preprocessing and the complete model call synchronously, so there is no
    N+1 GPU lookahead.  Close the pipeline before changing prompts or resetting
    the wrapped session.  ``metadata`` is retained by reference and must be
    treated as immutable by the submitter.
    """

    def __init__(self, live: SAM3Live, *, copy_frames: bool = True) -> None:
        self.live = live
        self.copy_frames = copy_frames
        self._raw = _LatestSlot[LiveFramePacket]()
        self._state_lock = threading.Lock()
        self._failure_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._close_complete = threading.Event()
        self._failure: _Failure | None = None
        self._drain_failed = False
        self._started = False
        self._input_closed = False
        self._aborted = False
        self._closed = False
        self._drained = False
        self._consumer_thread_id: int | None = None
        self._next_sequence = 0
        self._force_detection_pending = False
        self._submitted = 0
        self._outputs = 0
        self._failed_items = 0
        self._aborted_items = 0
        self._inference_owner_thread_id: int | None = None
        self._guard_targets = [live]
        nested_live = getattr(live, "live", None)
        if nested_live is not None and nested_live is not live:
            self._guard_targets.append(nested_live)
        if any(
            getattr(target, "_latest_frame_pipeline_poisoned", False)
            for target in self._guard_targets
        ):
            raise RuntimeError(
                "live session cannot be reused after a failed latest-frame GPU drain"
            )

    def __enter__(self) -> "LatestFramePipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, tb) -> None:
        self.close()

    def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("latest-frame pipeline is closed")
            if self._started:
                return
            if self._input_closed or self._aborted:
                raise RuntimeError("latest-frame pipeline input is already closed")
            with _LIVE_CLAIM_LOCK:
                if any(
                    getattr(target, "_latest_frame_pipeline_poisoned", False)
                    for target in self._guard_targets
                ):
                    raise RuntimeError(
                        "live session cannot be reused after a failed "
                        "latest-frame GPU drain"
                    )
                if any(
                    getattr(target, "_latest_frame_pipeline_active", None)
                    is not None
                    for target in self._guard_targets
                ):
                    raise RuntimeError(
                        "a latest-frame pipeline is already active on this live session"
                    )
                for target in self._guard_targets:
                    target._latest_frame_pipeline_active = self
            self._started = True

    def submit(
        self,
        frame_bgr: np.ndarray,
        *,
        sequence: int | None = None,
        captured_at: float | None = None,
        sensor_timestamp: int | float | None = None,
        full_detection: bool | None = None,
        metadata: Any = None,
    ) -> bool:
        """Publish the newest source frame without waiting for inference.

        ``captured_at`` is host-monotonic arrival time.  ``sensor_timestamp``
        is preserved but never mixed into host age calculations.
        """
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError("frame_bgr must be a numpy.ndarray")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR, got shape {frame_bgr.shape}")
        if frame_bgr.dtype != np.uint8:
            raise ValueError(f"expected uint8 BGR, got dtype {frame_bgr.dtype}")
        if sequence is not None and (
            isinstance(sequence, bool) or not isinstance(sequence, int)
        ):
            raise TypeError("sequence must be an int or None")
        if full_detection is not None and not isinstance(full_detection, bool):
            raise TypeError("full_detection must be bool or None")

        now = time.perf_counter()
        host_capture_time = now if captured_at is None else float(captured_at)
        if not math.isfinite(host_capture_time):
            raise ValueError("captured_at must be finite")
        if host_capture_time > now:
            raise ValueError(
                "captured_at must be a past host-monotonic timestamp; put sensor "
                "or wall-clock time in sensor_timestamp"
            )

        self._raise_if_failed()
        with self._state_lock:
            if not self._started:
                raise RuntimeError("start the latest-frame pipeline before submit")
            if self._input_closed or self._closed:
                return False

        # Copy before publishing so a producer can immediately recycle its
        # camera/loaned-message buffer after submit returns.
        owned_frame = (
            np.array(frame_bgr, copy=True, order="C", subok=False)
            if self.copy_frames
            else frame_bgr
        )

        self._raise_if_failed()
        with self._state_lock:
            if self._input_closed or self._closed:
                return False
            if sequence is None:
                sequence = self._next_sequence
            elif sequence < self._next_sequence:
                raise ValueError(
                    f"source sequence must be monotonic: got {sequence}, "
                    f"expected at least {self._next_sequence}"
                )
            self._next_sequence = sequence + 1
            if full_detection is True:
                # Do not lose an externally requested detection merely because
                # its frame is replaced before the consumer becomes available.
                self._force_detection_pending = True
            packet = LiveFramePacket(
                sequence=sequence,
                captured_at=host_capture_time,
                frame_bgr=owned_frame,
                sensor_timestamp=sensor_timestamp,
                full_detection=full_detection,
                metadata=metadata,
            )
            accepted = self._raw.put(packet)
            if accepted:
                self._submitted += 1
            return accepted

    def finish_input(self) -> None:
        """Stop accepting input and drain the newest already accepted frame."""
        with self._state_lock:
            if self._input_closed:
                return
            self._input_closed = True
            self._raw.close()

    def infer_next(self) -> LatestFrameResult | None:
        """Infer the freshest queued frame on the sole consumer thread."""
        with self._state_lock:
            if not self._started:
                raise RuntimeError(
                    "start the latest-frame pipeline before infer_next"
                )
        self._bind_consumer_thread()
        self._raise_if_failed()
        packet = self._raw.get()
        if packet is None:
            with self._state_lock:
                self._drained = True
            self._raise_if_failed()
            return None

        with self._inference_lock:
            with self._state_lock:
                if self._aborted or self._closed:
                    self._aborted_items += 1
                    return None
                self._inference_owner_thread_id = threading.get_ident()
                full_detection = packet.full_detection
                if self._force_detection_pending:
                    full_detection = True
                    self._force_detection_pending = False
            started_at = time.perf_counter()
            try:
                output = self.live.infer(
                    packet.frame_bgr,
                    full_detection=full_detection,
                )
            except BaseException as exc:
                with self._state_lock:
                    self._failed_items += 1
                self._fail("infer", exc)
                self._drain_inference_device()
                raise LatestFramePipelineError(
                    "latest-frame inference failed"
                ) from exc
            finally:
                with self._state_lock:
                    self._inference_owner_thread_id = None
            completed_at = time.perf_counter()
            with self._state_lock:
                self._outputs += 1

        self._raise_if_failed()
        return LatestFrameResult(
            packet=packet,
            output=output,
            inference_started_at=started_at,
            completed_at=completed_at,
        )

    def abort(self) -> None:
        """Stop accepting input and discard the queued, not-yet-used frame."""
        with self._state_lock:
            self._input_closed = True
            self._aborted = True
            discarded = self._raw.close(discard=True)
            if discarded is not None:
                self._aborted_items += 1

    def close(self) -> None:
        """Idempotently close input and wait for any active inference call."""
        with self._state_lock:
            if self._closed:
                wait_for_close = True
            else:
                wait_for_close = False
                if self._inference_owner_thread_id == threading.get_ident():
                    raise RuntimeError(
                        "cannot close LatestFramePipeline from inside live.infer"
                    )
                self._closed = True
                self._input_closed = True
                self._aborted = True
                discarded = self._raw.close(discard=True)
                if discarded is not None:
                    self._aborted_items += 1
        if wait_for_close:
            self._close_complete.wait()
            return
        try:
            # An in-flight synchronous infer cannot be cancelled safely.
            # Waiting prevents session teardown from racing that call.
            with self._inference_lock:
                pass
            with _LIVE_CLAIM_LOCK:
                for target in self._guard_targets:
                    if getattr(target, "_latest_frame_pipeline_active", None) is self:
                        delattr(target, "_latest_frame_pipeline_active")
        finally:
            self._close_complete.set()

    def stats(self) -> dict[str, int]:
        """Return a consistent snapshot of bounded-pipeline counters."""
        with self._state_lock:
            return {
                "submitted_frames": self._submitted,
                "output_frames": self._outputs,
                "dropped_frames": self._raw.dropped,
                "failed_frames": self._failed_items,
                "aborted_frames": self._aborted_items,
                "drain_failed": self._drain_failed,
                "queued_frames": int(self._raw.occupied),
            }

    def _bind_consumer_thread(self) -> None:
        ident = threading.get_ident()
        with self._state_lock:
            if self._consumer_thread_id is None:
                self._consumer_thread_id = ident
            elif self._consumer_thread_id != ident:
                raise RuntimeError("infer_next must be called from one consumer thread")

    def _is_inference_owner_thread(self) -> bool:
        """Return whether the caller is the pipeline's current model owner."""
        return self._inference_owner_thread_id == threading.get_ident()

    def _fail(self, stage: str, exc: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = _Failure(stage, exc, traceback.format_exc())
        self.abort()

    def _raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise LatestFramePipelineError(
                f"latest-frame pipeline stage {failure.stage!r} failed: "
                f"{failure.exception}"
            ) from failure.exception

    def _drain_inference_device(self) -> None:
        device = getattr(self.live, "device", None)
        if device is None:
            device = getattr(getattr(self.live, "live", None), "device", None)
        if device is None:
            return
        if device.type != "cuda":
            return
        try:
            torch.cuda.synchronize(device=device)
        except BaseException as exc:
            self._drain_failed = True
            with _LIVE_CLAIM_LOCK:
                for target in self._guard_targets:
                    target._latest_frame_pipeline_poisoned = True
            with self._failure_lock:
                if self._failure is None:
                    self._failure = _Failure(
                        "inference_drain", exc, traceback.format_exc()
                    )


__all__ = [
    "LatestFramePipeline",
    "LatestFramePipelineError",
    "LatestFrameResult",
    "LiveFramePacket",
]
