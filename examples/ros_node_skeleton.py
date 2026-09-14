#!/usr/bin/env python3
"""Freshness-first ROS 2 integration skeleton for SAM3Live.

This file deliberately does not import rclpy. It is runnable against a video
file while showing the ownership and threading rules a real ROS 2 node must
preserve:

* the image subscription callback only timestamps and submits an image;
* LatestFramePipeline makes an owning image copy and keeps one queued frame;
* exactly one consumer thread performs inference and publication; and
* prompt changes use close-old, reset, new-generation ordering.

Use sensor-data QoS with KEEP_LAST(depth=1) upstream. The application queue
cannot remove frames already buffered by DDS, a camera driver, or a decoder.

Standalone test (the file source is paced at its declared FPS):

    python examples/ros_node_skeleton.py \
        --checkpoint model/sam3 --onnx-dir onnx_files_504 \
        --video assets/blackswan.mp4 --text swan water
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


# Make the example runnable from outside the repository root.
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from tracker.rocm_env import apply as _apply_rocm_env

_apply_rocm_env()

import cv2
import numpy as np

from tracker.latest_frame import LatestFramePipeline
from tracker.live_inference import SAM3Live


# ======================================================================
# Detection policies
# ======================================================================

class Policy(Protocol):
    """A callable deciding whether an arrival requests full detection."""

    def __call__(self, ctx: dict) -> bool: ...


class AlwaysFull:
    """Default: detect on every frame that survives latest-frame dropping."""

    def __call__(self, ctx: dict) -> bool:
        return True


class TimeBasedRedetect:
    """Request detection after the configured host-monotonic interval."""

    def __init__(self, interval_ms: float):
        self.interval_ms = float(interval_ms)
        self._last_request_ms = -float("inf")
        self._lock = threading.Lock()

    def __call__(self, ctx: dict) -> bool:
        with self._lock:
            now_ms = ctx["wall_ms"]
            if now_ms - self._last_request_ms >= self.interval_ms:
                self._last_request_ms = now_ms
                return True
            return False


class PeriodicRedetect:
    """Request detection every Nth source arrival."""

    def __init__(self, period: int):
        self.period = max(1, int(period))
        self._count = 0
        self._lock = threading.Lock()

    def __call__(self, ctx: dict) -> bool:
        with self._lock:
            requested = self._count % self.period == 0
            self._count += 1
            return requested


class OnDemandTrigger:
    """Request one full detection after each external trigger."""

    def __init__(self, start_armed: bool = True):
        self._armed = bool(start_armed)
        self._lock = threading.Lock()

    def trigger(self) -> None:
        with self._lock:
            self._armed = True

    def __call__(self, ctx: dict) -> bool:
        with self._lock:
            if not self._armed:
                return False
            self._armed = False
            return True


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Immutable application metadata carried with one source frame."""

    generation: int
    callback_index: int
    source_sequence: int
    detection_request_epoch: int


# ======================================================================
# ROS-shaped node wrapper
# ======================================================================

class SAM3Node:
    """Own one model plus a succession of latest-frame generations.

    The subscription-facing on_image method never runs inference. The private
    consumer thread is the only thread allowed to call the model or publish its
    outputs. Runtime prompt replacement detaches and closes the old generation,
    joins its consumer, resets model state, then starts a new generation.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        onnx_dir: str | Path | None = None,
        prompts: list[str],
        policy: Policy | None = None,
        imgsz: int = 504,
        mig: bool = True,
        parallel_tail: bool | None = None,
        fixed_detr_decoder: bool | None = None,
        max_result_age_ms: float | None = 200.0,
        max_objects_per_prompt: int | None = 5,
    ):
        # In a lifecycle node, construct this object in on_configure. Model load
        # and MIG warmup are one-time costs.
        if parallel_tail is None:
            parallel_tail = mig
        self.live = SAM3Live(
            checkpoint=checkpoint,
            prompts=prompts,
            onnx_dir=onnx_dir,
            imgsz=imgsz,
            mig=mig,
            parallel_tail=parallel_tail,
            fixed_detr_decoder=fixed_detr_decoder,
            max_objects_per_prompt=max_objects_per_prompt,
            redetect_every=1,
        )
        self.policy = policy or AlwaysFull()
        self.max_result_age_ms = (
            None
            if max_result_age_ms is None or max_result_age_ms <= 0
            else float(max_result_age_ms)
        )

        # _transition_lock serializes reset/finish/close. _state_lock protects
        # callback-visible references and counters, but is never held while
        # waiting for inference or joining a thread.
        self._transition_lock = threading.Lock()
        self._callback_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._pipeline: LatestFramePipeline | None = None
        self._consumer_thread: threading.Thread | None = None
        self._consumer_errors: dict[int, BaseException] = {}
        self._pipeline_history: list[dict] = []
        self._recorded_generations: set[int] = set()
        self._generation = -1
        self._accepting = False
        self._closed = False
        self._live_closed = False

        self._callback_count = 0
        self._source_sequence = 0
        self._accepted_callbacks = 0
        self._rejected_callbacks = 0
        self._output_count = 0
        self._stale_result_count = 0
        self._service_ms: list[float] = []
        self._age_ms: list[float] = []

        # Detection requests are sticky across latest-slot replacement. This is
        # essential for on-demand policies: dropping one camera frame must not
        # silently drop the control request attached to it.
        self._detection_pending = True
        self._detection_request_epoch = 0
        self._t0 = time.perf_counter()
        self._start_generation()

    # ------------------------------------------------------------------
    # Image subscription callback
    # ------------------------------------------------------------------
    def on_image(
        self,
        frame_bgr: np.ndarray,
        *,
        header_stamp_ns: int | None = None,
    ) -> bool:
        """Serialize callback submission so source sequences cannot reorder."""
        with self._callback_lock:
            return self._on_image_serialized(
                frame_bgr,
                header_stamp_ns=header_stamp_ns,
            )

    def _on_image_serialized(
        self,
        frame_bgr: np.ndarray,
        *,
        header_stamp_ns: int | None = None,
    ) -> bool:
        """Timestamp and submit one image without running inference.

        In ROS 2, bind this method to a sensor-data subscription with
        KEEP_LAST(depth=1). Convert with cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        and pass msg.header.stamp as nanoseconds.

        LatestFramePipeline copies frame_bgr before returning, so a loaned ROS
        or camera buffer may be recycled immediately. The sensor timestamp is
        preserved for TF lookup; latency uses a separate host-monotonic value.
        """
        host_arrived_at = time.perf_counter()
        with self._state_lock:
            callback_index = self._callback_count
            self._callback_count += 1
            source_sequence = self._source_sequence
            self._source_sequence += 1
            pipeline = self._pipeline if self._accepting else None
            generation = self._generation

            error = self._consumer_errors.get(generation)
            if error is not None:
                raise RuntimeError(
                    f"latest-frame consumer generation {generation} failed"
                ) from error
            if pipeline is None:
                self._rejected_callbacks += 1
                return False

            context = {
                "callback_idx": callback_index,
                "source_sequence": source_sequence,
                "generation": generation,
                "wall_ms": (host_arrived_at - self._t0) * 1000.0,
                "stamp_ns": header_stamp_ns,
            }
            if bool(self.policy(context)):
                self._detection_pending = True
                self._detection_request_epoch += 1
            request_epoch = self._detection_request_epoch
            full_detection = self._detection_pending

        accepted = pipeline.submit(
            frame_bgr,
            sequence=source_sequence,
            captured_at=host_arrived_at,
            sensor_timestamp=header_stamp_ns,
            full_detection=full_detection,
            metadata=FrameContext(
                generation=generation,
                callback_index=callback_index,
                source_sequence=source_sequence,
                detection_request_epoch=request_epoch,
            ),
        )
        with self._state_lock:
            if accepted:
                self._accepted_callbacks += 1
            else:
                # A close/reset can detach the generation after the callback
                # snapshots its pipeline but before submit acquires its lock.
                self._rejected_callbacks += 1
        return accepted

    # ------------------------------------------------------------------
    # Output publication -- consumer thread only
    # ------------------------------------------------------------------
    def _publish_masks(self, item, *, generation: int) -> None:
        """Publish one result with source timestamp and generation identity.

        Replace this print with ROS publishers. Messages used for occupancy
        updates must carry item.packet.sensor_timestamp, and TF must be queried
        at that exposure time. Publish empty results too if downstream clearing
        or decay logic depends on observations with no detected object.
        """
        result = item.output
        groups = "  ".join(
            f"{prompt}:{len(ids)}"
            for prompt, ids in result["prompt_to_obj_ids"].items()
        ) or "none"
        print(
            f"[gen={generation} src={item.packet.sequence:4d}]  "
            f"sensor_ns={item.packet.sensor_timestamp!s:<12}  "
            f"det={str(result['detected']):<5}  "
            f"age={item.age_ms:6.1f} ms  "
            f"objs={len(result['object_ids']):2d}  "
            f"{groups}"
        )

    def _consumer_loop(
        self,
        pipeline: LatestFramePipeline,
        generation: int,
    ) -> None:
        try:
            while True:
                item = pipeline.infer_next()
                if item is None:
                    return
                context = item.packet.metadata
                with self._state_lock:
                    # A reset or abort detaches first. Suppress any old result
                    # that completed while the lifecycle caller waited.
                    current = (
                        self._pipeline is pipeline
                        and self._generation == generation
                    )
                    stale = (
                        current
                        and self.max_result_age_ms is not None
                        and item.age_ms > self.max_result_age_ms
                    )
                    if stale:
                        self._stale_result_count += 1
                if stale:
                    continue
                if current:
                    self._publish_masks(item, generation=generation)
                    with self._state_lock:
                        self._output_count += 1
                        self._service_ms.append(item.service_ms)
                        self._age_ms.append(item.age_ms)
                        if (
                            item.output.get("detected", False)
                            and isinstance(context, FrameContext)
                            and context.detection_request_epoch
                            == self._detection_request_epoch
                        ):
                            self._detection_pending = False
        except BaseException as exc:
            # infer_next already aborts on model failures; abort also handles a
            # publisher exception and wakes any future wait.
            pipeline.abort()
            with self._state_lock:
                self._consumer_errors[generation] = exc
                if self._pipeline is pipeline:
                    self._accepting = False

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------
    def reset_prompts(self, new_prompts: list[str]) -> None:
        """Close the old generation, reset prompts, and start a new one.

        Wire this method to a ROS service or lifecycle transition. Source
        callbacks arriving during the transition return False instead of
        entering either prompt generation.
        """
        prompts = [prompt for prompt in new_prompts if prompt]
        if not prompts:
            raise ValueError("new_prompts must contain at least one prompt")

        with self._transition_lock:
            self._require_open()
            pipeline, consumer, generation = self._detach_generation()
            if pipeline is not None:
                pipeline.close()
                self._join_consumer(consumer)
                self._record_pipeline_stats(generation, pipeline)
                self._raise_consumer_error(generation)

            # The old pipeline has released its active marker and its sole
            # consumer has exited, so direct model mutation is now safe.
            self.live.reset_prompts(prompts)
            with self._state_lock:
                self._detection_pending = True
                self._detection_request_epoch += 1
            self._start_generation()

    def trigger_redetect(self) -> None:
        """Arm one sticky full-detection request when the policy supports it."""
        if hasattr(self.policy, "trigger"):
            self.policy.trigger()

    def reset_tracking(self) -> None:
        """Reset tracker history through a generation-safe lifecycle barrier."""
        with self._transition_lock:
            self._require_open()
            pipeline, consumer, generation = self._detach_generation()
            if pipeline is not None:
                pipeline.close()
                self._join_consumer(consumer)
                self._record_pipeline_stats(generation, pipeline)
                self._raise_consumer_error(generation)

            self.live.reset_tracking()
            with self._state_lock:
                self._detection_pending = True
                self._detection_request_epoch += 1
            self._start_generation()

    def finish(self) -> None:
        """Stop accepting images and drain the final queued frame."""
        with self._transition_lock:
            self._require_not_consumer_thread()
            with self._state_lock:
                pipeline = self._pipeline
                consumer = self._consumer_thread
                generation = self._generation
                self._accepting = False
            if pipeline is None:
                return

            pipeline.finish_input()
            self._join_consumer(consumer)
            pipeline.close()
            with self._state_lock:
                if self._pipeline is pipeline:
                    self._pipeline = None
                    self._consumer_thread = None
            self._record_pipeline_stats(generation, pipeline)
            self._raise_consumer_error(generation)

    def close(self) -> None:
        """Abort queued work, join the consumer, then release the model."""
        with self._transition_lock:
            self._require_not_consumer_thread()
            with self._state_lock:
                if self._live_closed:
                    return
                self._closed = True
            pipeline, consumer, generation = self._detach_generation()
            if pipeline is not None:
                pipeline.close()
                self._join_consumer(consumer)
                self._record_pipeline_stats(generation, pipeline)
            try:
                self.live.close()
            finally:
                with self._state_lock:
                    self._live_closed = True

    def stats(self) -> dict:
        """Return node counters and completed generation statistics."""
        with self._state_lock:
            return {
                "callbacks": self._callback_count,
                "accepted_callbacks": self._accepted_callbacks,
                "rejected_callbacks": self._rejected_callbacks,
                "output_frames": self._output_count,
                "stale_results": self._stale_result_count,
                "service_ms": list(self._service_ms),
                "age_ms": list(self._age_ms),
                "pipeline_generations": list(self._pipeline_history),
            }

    # ------------------------------------------------------------------
    # Lifecycle internals
    # ------------------------------------------------------------------
    def _start_generation(self) -> None:
        pipeline = LatestFramePipeline(self.live, copy_frames=True)
        pipeline.start()
        with self._state_lock:
            if self._closed:
                pipeline.close()
                raise RuntimeError("SAM3Node is closed")
            self._generation += 1
            generation = self._generation
            consumer = threading.Thread(
                target=self._consumer_loop,
                args=(pipeline, generation),
                name=f"sam3-latest-consumer-{generation}",
            )
            self._pipeline = pipeline
            self._consumer_thread = consumer
            self._accepting = True
        try:
            consumer.start()
        except BaseException:
            with self._state_lock:
                if self._pipeline is pipeline:
                    self._pipeline = None
                    self._consumer_thread = None
                    self._accepting = False
            pipeline.close()
            raise

    def _detach_generation(
        self,
    ) -> tuple[LatestFramePipeline | None, threading.Thread | None, int]:
        self._require_not_consumer_thread()
        with self._state_lock:
            pipeline = self._pipeline
            consumer = self._consumer_thread
            generation = self._generation
            self._accepting = False
            self._pipeline = None
            self._consumer_thread = None
            return pipeline, consumer, generation

    @staticmethod
    def _join_consumer(consumer: threading.Thread | None) -> None:
        if consumer is None:
            return
        if consumer is threading.current_thread():
            raise RuntimeError("cannot join the latest-frame consumer from itself")
        consumer.join()

    def _require_not_consumer_thread(self) -> None:
        with self._state_lock:
            consumer = self._consumer_thread
        if consumer is threading.current_thread():
            raise RuntimeError(
                "lifecycle transitions must not run on the inference thread"
            )

    def _require_open(self) -> None:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("SAM3Node is closed")

    def _record_pipeline_stats(
        self,
        generation: int,
        pipeline: LatestFramePipeline,
    ) -> None:
        with self._state_lock:
            if generation in self._recorded_generations:
                return
            self._recorded_generations.add(generation)
            self._pipeline_history.append(
                {"generation": generation, **pipeline.stats()}
            )

    def _raise_consumer_error(self, generation: int) -> None:
        with self._state_lock:
            error = self._consumer_errors.get(generation)
        if error is not None:
            raise RuntimeError(
                f"latest-frame consumer generation {generation} failed"
            ) from error


# ======================================================================
# Standalone source-paced video harness
# ======================================================================

def _build_policy(name: str, args) -> Policy:
    if name == "always_full":
        return AlwaysFull()
    if name == "time_based":
        return TimeBasedRedetect(interval_ms=args.redetect_interval_ms)
    if name == "periodic":
        return PeriodicRedetect(period=args.redetect_period)
    if name == "on_demand":
        return OnDemandTrigger()
    raise ValueError(f"unknown policy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=Path(
            os.environ.get("SAM3_DEFAULT_ONNX_DIR", "onnx_files_504_mgx217")
        ),
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--text", type=str, nargs="+", required=True)
    parser.add_argument("--imgsz", type=int, default=504, choices=(504, 1008))
    parser.add_argument("--no-mig", action="store_true")
    parallel = parser.add_mutually_exclusive_group()
    parallel.add_argument(
        "--parallel-tail",
        dest="parallel_tail",
        action="store_true",
        help="Enable detector/tracker overlap (default when MIG is enabled).",
    )
    parallel.add_argument(
        "--no-parallel-tail",
        dest="parallel_tail",
        action="store_false",
        help="Disable detector/tracker overlap for diagnosis.",
    )
    parser.set_defaults(parallel_tail=None)
    fixed_decoder = parser.add_mutually_exclusive_group()
    fixed_decoder.add_argument(
        "--fixed-detr-decoder",
        dest="fixed_detr_decoder",
        action="store_true",
        help="Require the fixed 504px direct-MXR DETR decoder.",
    )
    fixed_decoder.add_argument(
        "--no-fixed-detr-decoder",
        dest="fixed_detr_decoder",
        action="store_false",
        help="Use the native DETR decoder for diagnosis.",
    )
    parser.set_defaults(fixed_detr_decoder=None)
    parser.add_argument("--max-objects", type=int, default=5)
    parser.add_argument(
        "--max-result-age-ms",
        type=float,
        default=200.0,
        help="Drop completed results older than this host-arrival age; 0 disables.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Cap source arrivals (0 = full video); outputs may be fewer.",
    )
    parser.add_argument(
        "--policy",
        choices=["always_full", "time_based", "periodic", "on_demand"],
        default="always_full",
    )
    parser.add_argument("--redetect-interval-ms", type=float, default=300.0)
    parser.add_argument("--redetect-period", type=int, default=5)
    parser.add_argument("--trigger-at", type=int, nargs="*", default=[])
    parser.add_argument(
        "--switch-at",
        type=str,
        nargs="*",
        default=[],
        help="Source-index prompt swaps, for example 30:car,sidewalk.",
    )
    args = parser.parse_args()
    if args.parallel_tail is None:
        args.parallel_tail = not args.no_mig
    if args.parallel_tail and args.no_mig:
        parser.error("--parallel-tail requires MIG (remove --no-mig)")
    if args.fixed_detr_decoder is True and args.no_mig:
        parser.error("--fixed-detr-decoder requires MIG")

    switch_at: dict[int, list[str]] = {}
    for spec in args.switch_at:
        index_text, prompts_text = spec.split(":", 1)
        switch_at[int(index_text)] = [
            prompt.strip()
            for prompt in prompts_text.split(",")
            if prompt.strip()
        ]
    trigger_at = set(args.trigger_at)

    node = SAM3Node(
        checkpoint=args.checkpoint,
        onnx_dir=args.onnx_dir,
        imgsz=args.imgsz,
        prompts=args.text,
        policy=_build_policy(args.policy, args),
        mig=not args.no_mig,
        parallel_tail=args.parallel_tail,
        fixed_detr_decoder=args.fixed_detr_decoder,
        max_result_age_ms=args.max_result_age_ms,
        max_objects_per_prompt=args.max_objects,
    )
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        node.close()
        raise SystemExit(f"Cannot open {args.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = 24.0
    period = 1.0 / source_fps
    next_arrival = time.perf_counter()
    source_count = 0
    print(
        f"[node] source={source_fps:.3f} FPS policy={args.policy} "
        f"prompts={args.text} max_objects={args.max_objects}"
    )

    try:
        while args.max_frames <= 0 or source_count < args.max_frames:
            delay = next_arrival - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            ok, frame_bgr = capture.read()
            if not ok:
                break

            if source_count in trigger_at:
                print(f"[node] trigger at source={source_count}")
                node.trigger_redetect()
            if source_count in switch_at:
                prompts = switch_at[source_count]
                print(f"[node] reset_prompts({prompts}) at source={source_count}")
                node.reset_prompts(prompts)

            node.on_image(
                frame_bgr,
                # A real adapter passes msg.header.stamp here. The standalone
                # file has no sensor clock, so use no synthetic sensor stamp.
                header_stamp_ns=None,
            )
            source_count += 1

            # Never burst through several file frames after a slow reset. A
            # real camera keeps its own cadence and the DDS queue drops stale
            # arrivals while this process is busy.
            next_arrival = max(next_arrival + period, time.perf_counter())

        node.finish()
        stats = node.stats()
    finally:
        capture.release()
        node.close()

    service = np.asarray(stats["service_ms"], dtype=np.float64)
    age = np.asarray(stats["age_ms"], dtype=np.float64)
    print(
        f"\n[node] done. source={source_count} "
        f"accepted={stats['accepted_callbacks']} "
        f"outputs={stats['output_frames']} "
        f"rejected={stats['rejected_callbacks']}"
    )
    if service.size:
        print(
            f"  service: mean={service.mean():.1f} ms "
            f"p50={np.median(service):.1f} "
            f"p95={np.percentile(service, 95):.1f} ms"
        )
        print(
            f"  age:     mean={age.mean():.1f} ms "
            f"p50={np.median(age):.1f} "
            f"p95={np.percentile(age, 95):.1f} ms"
        )
    for generation in stats["pipeline_generations"]:
        print(f"  generation: {generation}")


if __name__ == "__main__":
    main()
