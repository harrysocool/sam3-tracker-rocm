#!/usr/bin/env python3
"""ROS 2 integration skeleton for SAM3Live — drop-in starting point.

This file does not import ``rclpy``, so it stays readable (and runnable
against a video file) without a ROS 2 install. It is structured as a
``rclpy.node.Node``-style class: construct the model once at startup,
feed frames in from image-topic callbacks, decide per-callback whether
to spend compute on full SAM3 detection or just tracker propagation.

To turn this into a real ROS 2 node, replace four marked sections
(search for ``REPLACE``):

  1. ``_subscribe_*``      → ``self.create_subscription(Image, topic,
                              self.on_image, qos_profile_sensor_data)``
                              with ``cv_bridge.imgmsg_to_cv2(msg, "bgr8")``
  2. ``_publish_*``        → ``self.create_publisher(...)`` for your
                              output topics (typically
                              ``vision_msgs/Detection2DArray`` for
                              boxes+scores and ``sensor_msgs/Image``
                              for an overlay; consider per-prompt mask
                              topics for Nav2 costmap layer inputs)
  3. ``reset_prompts``     → ``self.create_service(...)`` for runtime
                              prompt swap (e.g. context switch on
                              lifecycle transition or external trigger)
  4. ``trigger_redetect``  → service / parameter / GUI source for the
                              ``OnDemandTrigger`` policy

If you wrap this in a lifecycle node, do the ``SAM3Live(...)``
construction in ``on_configure`` (model load + MIG compile is ~5 s) and
start the image subscription in ``on_activate``.

Policies provided (write your own — protocol is just
``__call__(ctx) -> bool``):

  - ``AlwaysFull``        : full SAM3 every callback (baseline)
  - ``TimeBasedRedetect`` : full SAM3 if >=X ms since the last
                            (recommended for steady-rate sensors;
                            self-heals on dropped frames)
  - ``PeriodicRedetect``  : full SAM3 every Nth callback
  - ``OnDemandTrigger``   : propagation only; external ``trigger()``
                            arms one full detection

Standalone test (no ROS 2 install needed — feeds a video file in place
of the image topic):

    python examples/ros_node_skeleton.py \\
        --checkpoint model/sam3 --onnx-dir onnx_files_504 \\
        --video assets/blackswan.mp4 --text swan water \\
        --policy time_based --redetect-interval-ms 500
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Protocol

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")

# Add project root to path so this example works when run from anywhere
# (``python examples/ros_node_skeleton.py`` or from the project root).
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

import cv2
import numpy as np

from tracker.live_inference import SAM3Live


# ======================================================================
# Policies — pluggable rules deciding "should this callback run detection?"
# ======================================================================

class Policy(Protocol):
    """A policy is anything callable as ``policy(ctx) -> bool``.

    ``ctx`` is a small dict the node fills in each callback (see SAM3Node).
    Add fields freely if your policy needs them (e.g. queue_depth,
    last_publish_age, robot speed, etc.).
    """
    def __call__(self, ctx: dict) -> bool: ...


class AlwaysFull:
    """Default — full SAM3 every callback. Matches the 5 Hz baseline."""
    def __call__(self, ctx: dict) -> bool:
        return True


class TimeBasedRedetect:
    """Detect if at least ``interval_ms`` has passed since the last detect.

    Good default for steady-rate sensors. If the sensor occasionally drops
    a frame or comes in late, the schedule self-heals on wall clock.
    """
    def __init__(self, interval_ms: float):
        self.interval_ms = float(interval_ms)
        self._last_detect_ms: float = -float("inf")

    def __call__(self, ctx: dict) -> bool:
        now_ms = ctx["wall_ms"]
        if (now_ms - self._last_detect_ms) >= self.interval_ms:
            self._last_detect_ms = now_ms
            return True
        return False


class PeriodicRedetect:
    """Detect every Nth callback. Use when sensor rate is reliable."""
    def __init__(self, period: int):
        self.period = max(1, int(period))
        self._count = 0

    def __call__(self, ctx: dict) -> bool:
        do = (self._count % self.period) == 0
        self._count += 1
        return do


class OnDemandTrigger:
    """Tracker propagation only, except when externally armed.

    Wire ``trigger()`` to a ROS service, UI button, or other event source
    that says "look for new objects now". The next callback after each
    trigger runs full detection, then reverts to propagation.
    """
    def __init__(self, start_armed: bool = True):
        self._armed = bool(start_armed)

    def trigger(self) -> None:
        self._armed = True

    def __call__(self, ctx: dict) -> bool:
        if self._armed:
            self._armed = False
            return True
        return False


# ======================================================================
# The node
# ======================================================================

class SAM3Node:
    """Callback-driven wrapper around SAM3Live, shaped like a rclpy.Node.

    Construct once at node startup (or in ``on_configure`` for a
    lifecycle node). Bind ``on_image`` to your image subscription and
    replace ``_publish_*`` with your real publishers.
    """

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        onnx_dir: str | Path,
        prompts: list[str],
        policy: Policy,
        imgsz: int = 504,
        mig: bool = True,
        max_objects_per_prompt: int | None = 5,
    ):
        # Model load + MIG warmup is slow (~5 s) — do it once at startup.
        # In a lifecycle node, put this in on_configure; only activate
        # the image subscription in on_activate.
        self.live = SAM3Live(
            checkpoint=checkpoint,
            prompts=prompts,
            onnx_dir=onnx_dir,
            imgsz=imgsz,
            mig=mig,
            max_objects_per_prompt=max_objects_per_prompt,
            redetect_every=1,  # the policy drives detection via full_detection
        )
        self.policy = policy
        self._callback_count = 0
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Image callback — REPLACE: self.create_subscription(Image, topic,
    # self.on_image, qos_profile_sensor_data); convert msg with
    # cv_bridge.imgmsg_to_cv2(msg, "bgr8") and pass header_stamp_ns from
    # msg.header.stamp (sec*1e9 + nanosec).
    # ------------------------------------------------------------------
    def on_image(
        self,
        frame_bgr: np.ndarray,
        *,
        header_stamp_ns: int | None = None,
    ) -> dict:
        """Image topic callback. Call once per incoming frame.

        Args:
            frame_bgr: HxWx3 uint8 BGR. Convert from sensor_msgs/Image with
                ``cv_bridge.imgmsg_to_cv2(msg, "bgr8")``.
            header_stamp_ns: sensor timestamp from ``msg.header.stamp``
                (use ``stamp.sec * 1_000_000_000 + stamp.nanosec``). Pass
                through to your output messages so downstream consumers
                (e.g. Nav2 tf2 lookups) see correct timing.

        Returns:
            The SAM3Live result dict (object_ids, masks, boxes, scores,
            prompt_to_obj_ids, detected, frame_idx). Use it to populate
            your output messages.
        """
        self._callback_count += 1
        wall_ms = (time.perf_counter() - self._t0) * 1000.0

        # Context handed to the policy. Add fields as you need them
        # (queue depth from the subscriber, current robot velocity, etc).
        ctx = {
            "callback_idx": self._callback_count,
            "wall_ms": wall_ms,
            "stamp_ns": header_stamp_ns,
        }
        do_full = bool(self.policy(ctx))

        out = self.live.infer(frame_bgr, full_detection=do_full)
        self._publish_masks(out, wall_ms=wall_ms)
        return out

    # ------------------------------------------------------------------
    # Output publishing — REPLACE with self.create_publisher(...) calls.
    # Typical Nav2-adjacent shape:
    #   - vision_msgs/Detection2DArray for boxes + per-detection score/class
    #   - sensor_msgs/Image (mono8 or rgba8) per prompt for mask overlays
    #     usable as a Nav2 costmap filter / spatio-temporal voxel input
    #   - a small msg with the prompt_to_obj_ids mapping if downstream
    #     consumers need the class grouping
    # ------------------------------------------------------------------
    def _publish_masks(self, result: dict, *, wall_ms: float) -> None:
        """Push masks/boxes/scores to your output topic(s).

        Here we just print one line per callback for debugging.
        """
        if not result["object_ids"]:
            return
        groups = "  ".join(
            f"{p}:{len(ids)}" for p, ids in result["prompt_to_obj_ids"].items()
        )
        print(
            f"[{wall_ms:7.1f} ms]  "
            f"cb={self._callback_count:4d}  "
            f"det={str(result['detected']):<5}  "
            f"objs={len(result['object_ids']):2d}  "
            f"{groups}"
        )

    # ------------------------------------------------------------------
    # Runtime control — REPLACE with self.create_service(...) calls or
    # parameter callbacks. In a lifecycle node these can also be wired to
    # transition callbacks (e.g. swap prompts on on_activate).
    # ------------------------------------------------------------------
    def reset_prompts(self, new_prompts: list[str]) -> None:
        """ROS 2 service callback for swapping the prompt list at runtime.

        Setup cost on next callback (~ms for tokenizer; CLIP encode folded
        into the next forward) — fine for context switches.
        """
        self.live.reset_prompts(new_prompts)

    def trigger_redetect(self) -> None:
        """Arm a one-shot detection on the next callback.

        Only meaningful when the policy supports it (e.g. OnDemandTrigger).
        Silently no-op otherwise. Wire to a std_srvs/Trigger service or
        a parameter change for external control.
        """
        if hasattr(self.policy, "trigger"):
            self.policy.trigger()


# ======================================================================
# Standalone test harness — feeds a video file in place of the image topic
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


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--onnx-dir", type=Path, default=Path("onnx_files_504"))
    p.add_argument("--video", type=Path, required=True,
                   help="Stand-in for the image topic — fed one frame at a time.")
    p.add_argument("--text", type=str, nargs="+", required=True,
                   help="Initial prompts (multi-class supported).")
    p.add_argument("--imgsz", type=int, default=504, choices=(504, 1008))
    p.add_argument("--no-mig", action="store_true",
                   help="Disable MIGraphX (slower; for debugging the wrapper itself).")
    p.add_argument("--max-objects", type=int, default=5,
                   help="Per-prompt cap. Required for multi-instance prompts.")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Cap frames processed (0 = full video).")
    p.add_argument("--policy", choices=["always_full", "time_based", "periodic", "on_demand"],
                   default="time_based")
    p.add_argument("--redetect-interval-ms", type=float, default=300.0,
                   help="time_based: ms since last detect to trigger another.")
    p.add_argument("--redetect-period", type=int, default=5,
                   help="periodic: detect every Nth callback.")
    p.add_argument("--trigger-at", type=int, nargs="*", default=[],
                   help="on_demand: callback indices at which to arm detection "
                        "(simulates ROS service calls). Example: --trigger-at 0 30 60")
    p.add_argument("--switch-at", type=str, nargs="*", default=[],
                   help="Simulate runtime prompt swap. Format: 'callback_idx:p1,p2'. "
                        "Example: --switch-at 30:car,sidewalk 60:person")
    args = p.parse_args()

    policy = _build_policy(args.policy, args)
    node = SAM3Node(
        checkpoint=args.checkpoint,
        onnx_dir=args.onnx_dir,
        imgsz=args.imgsz,
        prompts=args.text,
        policy=policy,
        mig=not args.no_mig,
        max_objects_per_prompt=args.max_objects,
    )

    # Parse switch schedule: {callback_idx -> [prompts]}
    switch_at = {}
    for spec in args.switch_at:
        idx_s, prompts_s = spec.split(":", 1)
        switch_at[int(idx_s)] = [p.strip() for p in prompts_s.split(",") if p.strip()]
    trigger_set = set(args.trigger_at)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")

    n = 0
    latencies = []
    print(f"[node] starting policy={args.policy}  prompts={args.text}  "
          f"max_objects={args.max_objects}")
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Simulate external events that a real ROS node would receive on
        # other topics/services.
        if n in trigger_set:
            print(f"[node] trigger() at cb={n}")
            node.trigger_redetect()
        if n in switch_at:
            print(f"[node] reset_prompts({switch_at[n]}) at cb={n}")
            node.reset_prompts(switch_at[n])

        t0 = time.perf_counter()
        node.on_image(frame_bgr)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    cap.release()

    if latencies:
        steady = np.asarray(latencies[1:] if len(latencies) > 1 else latencies)
        print(
            f"\n[node] done. callbacks={n}  policy={args.policy}\n"
            f"  steady-state: mean={steady.mean():.0f} ms  "
            f"p50={np.median(steady):.0f}  p95={np.percentile(steady, 95):.0f}  "
            f"max={steady.max():.0f}\n"
            f"  effective rate: {1000.0 / steady.mean():.2f} Hz"
        )


if __name__ == "__main__":
    main()
