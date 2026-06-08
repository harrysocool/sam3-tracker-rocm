# SAM3Live — practical guide

A focused reference for integrating `tracker.live_inference.SAM3Live` into a
live sensor pipeline (robotics, RTSP, webcam). For the model architecture see
the top-level README; this file documents only the knobs that matter at
deployment time, with measured numbers and recommended starting points.

All numbers below are on the AMD Strix Halo dev box (`gfx1151`), 504 px,
MIGraphX path enabled. Use them as relative baselines; absolute latency on
your hardware will differ.

---

## 1. Minimum usable code

```python
from tracker.live_inference import SAM3Live

live = SAM3Live(
    checkpoint="model/sam3",
    prompts=["person", "car", "sidewalk"],
    onnx_dir="onnx_files_504",
    imgsz=504,
    mig=True,
)

for frame_bgr in your_camera_stream():       # HxWx3 uint8, OpenCV BGR
    out = live.infer(frame_bgr)
    for prompt, obj_ids in out["prompt_to_obj_ids"].items():
        for oid in obj_ids:
            mask  = out["masks"][oid]         # HxW bool, original resolution
            score = out["scores"][oid]
            box   = out["boxes"][oid]         # (x1, y1, x2, y2)
            # publish / visualize / fuse with your pipeline
```

The constructor does the slow stuff once (model load + MIG compile, ~5 s);
each `infer()` is the per-frame call you put in your callback.

The constructor default is `imgsz=504` (the primary supported resolution).
`mig=False` by default so the class is usable without MIG artifacts; for
deployment **pass `mig=True` as shown above** to get the ~5 FPS numbers in
this guide. 1008px is supported as an advanced option but isn't the
recommended path — most numbers in this guide assume 504 + MIG.

---

## 2. The four knobs that decide perf

| Knob | Default | What it does | When to change |
|---|---|---|---|
| **`max_objects_per_prompt`** | `5` | Cap on simultaneously tracked objects per prompt. Excess (lowest score) are evicted via `session.remove_object` so the tracker stops propagating them. | Lower (1-3) if you know each prompt has at most a handful of real instances. Set higher only if you genuinely expect more (crowds). **Do not set to `None` unless you've verified your scene's true object count — without a cap the session silently accumulates ghost detections that bloat per-frame cost 5-20×.** |
| **`redetect_every`** | `1` | Run the full SAM3 detector every Nth frame; intermediate frames run tracker propagation only. | Bump to 3-5 once tracking is stable to recover FPS. New objects entering the scene are only discovered at detect frames. |
| **`full_detection=` (per-call)** | `None` | Override the redetect schedule for a single `infer()` call. `True` forces detection, `False` forces propagate-only, `None` uses the schedule. | Use in ROS callbacks to drive detection by your own policy (wall-clock interval, external trigger, queue depth) instead of a fixed frame counter. |
| **`min_score`** (in your post-filter) | n/a | Drop detections below this confidence from the **output** you publish. Higher values reduce visible noise; they do *not* prevent the detector from spawning the candidate inside the session. | Pair with `max_objects_per_prompt` for both clean output and bounded compute. 0.4-0.6 is a sensible band for outdoor robotics. |

A 5th knob exists but you can usually ignore it: `keep_recent_frames` (default
`0` = don't prune raw frame tensors). Pruning is unsafe in isolation because
the tracker keeps per-frame state in `output_dict_per_obj` that must stay in
sync. For long-running streams, periodically call `live.reset_tracking()`
instead.

---

## 3. Recommended starting configs

| Use case | prompts | max_objects | redetect_every | min_score | Measured FPS |
|---|---|---|---|---|---|
| Single class, single target (e.g. tracking one swan) | 1 | 5 (default) | 1 (default) | 0.5 | ~4-5 |
| Two classes, bounded instances (e.g. swan + water) | 2 | 5 (default) | 5 | 0.4 | ~4.4 |
| Three classes, semi-dense (e.g. person + car + sidewalk) | 3 | 5 (default) | 5 | 0.4 | ~3.2 |
| Tight 5 Hz budget, accept tracking-only most frames | 3 | 5 (default) | 10 | 0.4 | extrapolated ~4-5 (not directly measured at N=10) |

These come from the perf sweep in `results/perf_rerun_*.log` and
`results/cap_default_perf_*.log` — see `git log` for context.

---

## 4. Resetting state

Two escape hatches, least to most destructive. The model itself stays
loaded across both — only session state is cleared.

```python
live.reset_tracking()     # drop tracked-object history; keep prompts + cache
live.reset_prompts([...]) # drop prompts + objects; install a new prompt list
                          # (1 ms; CLIP encode folded into next frame)
```

Frame 0 and the first frame after any reset always run full detection
regardless of `redetect_every`, so you never end up with an empty session
trying to propagate nothing.

---

## 5. ROS 2 integration (Nav2-friendly)

The end-to-end pattern (image subscription → infer → mask/detection
publication, with a pluggable "when do I run full detection" policy) is
in **[`ros_node_skeleton.py`](ros_node_skeleton.py)**. The file does not
import `rclpy`, so it is readable and runnable as a plain Python script
against a video file — but it's structured as a `rclpy.node.Node`-style
class with four marked `# REPLACE` sections you swap for real ROS 2
plumbing:

1. `self.create_subscription(Image, "/camera/image_raw", self.on_image, qos)`
   with `rclpy.qos.qos_profile_sensor_data` (BestEffort, depth 1) for
   live sensor streams.
2. `self.create_publisher(...)` for your output topics — typically
   `vision_msgs/Detection2DArray` for boxes+scores, `sensor_msgs/Image`
   for an overlay, and/or a per-prompt mask topic for downstream
   costmap layers (e.g. Nav2 costmap filter inputs, spatio-temporal
   voxel layer occupancy).
3. `self.create_service(SetPrompts, ...)` to wire `reset_prompts(...)`
   to a runtime prompt swap (e.g. switching from "indoor objects" to
   "outdoor objects" when crossing a doorway).
4. Trigger source (service, parameter callback, GUI button) for the
   `OnDemandTrigger` policy if you go that route.

Four reference policies, all subclasses of the same `(ctx) -> bool`
protocol — write your own if you need queue-depth-aware or
velocity-aware scheduling:

- `AlwaysFull` — full SAM3 every callback (Nav2 baseline)
- `TimeBasedRedetect` — full SAM3 only if N ms have elapsed since the
  last (recommended for steady-rate sensors; self-heals on dropped frames)
- `PeriodicRedetect` — full SAM3 every Nth callback
- `OnDemandTrigger` — propagation only, external `trigger()` arms one
  detection (use with a service / lifecycle transition)

If you wrap this as a lifecycle node, do the `SAM3Live(...)` construction
in `on_configure` (model load + MIG compile is ~5 s) and start the
image subscription in `on_activate`.

Run it standalone against a video file to verify the pattern works on
your hardware before wiring it up:

```bash
python examples/ros_node_skeleton.py \
    --checkpoint model/sam3 --onnx-dir onnx_files_504 \
    --video assets/blackswan.mp4 --text swan water \
    --policy time_based --redetect-interval-ms 300
```

---

## 6. Common gotchas

- **`--max-objects 0` is "explicitly unlimited", not "use default"** in the
  demo CLI. Pass `-1` for the SAM3Live default (5). Pass `0` only when you
  understand the ghost-accumulation hazard.

- **`init_video_session(video=...)` is for batch processing, not streaming.**
  It preprocesses every frame up front (~2 ms × N frames). SAM3Live uses
  the streaming entry point under the hood (empty session + `add_new_frame`
  per call) — don't bypass it.

- **Don't share one `SAM3Live` across threads.** One model, one session, one
  thread. If you need parallelism, run multiple instances on separate model
  loads (memory cost is non-trivial — only do this if you've measured the
  need).

- **Long-running streams**: raw frame tensors and tracker memory grow
  monotonically. Call `reset_tracking()` every few minutes (your call —
  depends on memory budget and how much continuity you want across the
  reset). The model itself is unaffected.

- **Multi-instance prompts ("tree", "person") in dense scenes** generate
  many low-score candidates per frame. `max_objects_per_prompt` keeps them
  bounded, but if you also need them out of the *output*, raise `min_score`
  in your post-filter — the cap controls compute, the score threshold
  controls what reaches the user.

---

## 7. What `infer()` returns

```python
{
    "object_ids":         [3, 7, 12],                # tracked obj IDs this frame
    "scores":             {3: 0.91, 7: 0.83, ...},   # detection score per ID
    "masks":              {3: <HxW bool>, ...},      # mask at original resolution
    "boxes":              {3: (x1,y1,x2,y2), ...},   # XYXY format
    "prompt_to_obj_ids":  {"person": [3, 7], "car": [12]},
    "frame_idx":          42,                        # session-internal counter
    "detected":           True,                      # did the detector run this frame?
}
```

`detected` reflects what actually happened (after forced-detect and override
precedence). Useful for logging and for your policy to know when it last got
new candidates.
