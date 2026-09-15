# SAM3Live freshness-first integration guide

This directory provides [ros_node_skeleton.py](ros_node_skeleton.py), a
standalone video example and integration skeleton for camera / ROS 2 input.
It runs without ROS and prints per-result logs and final statistics; it does
not subscribe to ROS topics, publish ROS messages, or save a rendered video.

For real ROS integration, your application must provide the ROS 2 environment
and dependencies (including `rclpy`, `sensor_msgs`, and `cv_bridge`), subscriptions,
and publishers. The supported SAM3 runtime image does not install ROS, and this
repository does not provide a ROS package or launch file.

Complete the [Quick start](../README.md#quick-start) first. The standalone
commands below use the supported container with `SAM3_MODEL_DIR` and
`SAM3_ONNX_DIR` exported to the model and MIGraphX 2.17 artifact directories.

## Standalone video check

Run this from the repository root. The bundled `blackswan.mp4` contains
50 frames at 24 FPS; the example paces arrivals at that rate:

```bash
./docker/rocm714/run.sh python examples/ros_node_skeleton.py \
  --checkpoint /models/sam3 \
  --onnx-dir /models/onnx_files_504 \
  --video assets/blackswan.mp4 \
  --text swan \
  --policy always_full \
  --max-frames 50
```

Published results produce `[gen=... src=...]` log lines, followed by a final
`[node] done.` summary. There may be fewer results than source frames because
waiting frames are replaced, and completed results older than the age budget
are not published. A short clip can finish during cold-start inference;
with the default 200 ms age gate, all its results may be rejected. Inspect
`completed` and `age_rejected` before treating zero publications as a model
failure. For a functional check only, add `--max-result-age-ms 0` to inspect
results regardless of age; restore an appropriate age budget for deployment.
Use [the video demos](../docs/usage.md) when you want an output MP4.

### Parameters and statistics

| Parameter | Default / behavior |
|---|---|
| `--text` | Required; one or more prompts |
| `--max-frames` | 0: read to EOF; positive values cap source arrivals, not output count |
| `--max-objects` | 5 per prompt; 0 is unlimited; -1 selects the default |
| `--max-result-age-ms` | 200: reject older results before publication; 0 disables this gate |
| `--policy` | `always_full`; alternatives request detection through the direct `SAM3Live` backend |
| `--redetect-interval-ms` | 300; used with `--policy time_based`, not the demo's clean-keyframe hybrid |
| `--redetect-period` | 5 source arrivals; used with `--policy periodic` |

The CLI converts `--max-objects 0` to the Python API's unlimited value, `None`.
For direct `SAM3Node` / `SAM3Live` calls, use `max_objects_per_prompt=None` to
remove the cap; API value 0 retains zero objects. The skeleton does not apply
the demos' `--min-score` filter. If needed, apply the
[shared output filter](../docs/usage.md#shared-output-behavior) in your publisher.

MIG and same-frame parallel tails are enabled by default. At 504px, the default
fixed decoder requires `detr_decoder_fixed/direct_gpuio.mxr`; a missing artifact
is an error. Use `--no-parallel-tail` or `--no-fixed-detr-decoder` for diagnosis.

The final summary distinguishes these populations:

| Terminal label | Meaning |
|---|---|
| `accepted` / `callback_rejected` | Source callbacks accepted into the pipeline or rejected during lifecycle transitions |
| `completed` | All inference results returned to the consumer, before age and generation checks |
| `published` | Results that passed those checks and whose publication callback completed |
| `age_rejected` | Results rejected because their host-arrival age exceeded the budget |
| `superseded` | Completed results suppressed because their generation was detached |

Service and age statistics are printed separately for `completed` and
`published` results. The latter exclude rejected results and must not be used
as the latency distribution of all inference work. These timings exclude the
publication callback and downstream ROS transport. Failed inference and frames
dropped or aborted before a result are not timing samples. The separate
`generation` counters report queue drops and aborts. An inference failure ends
the CLI with an error rather than a normal summary.

In `SAM3Node.stats()`, `completed_frames`, `completed_service_ms`, and
`completed_age_ms` describe returned inference results. The existing
`output_frames`, `service_ms`, and `age_ms` fields remain publication-only;
`stale_results` and `superseded_results` record the two rejection reasons.

### Exercise prompt changes

```bash
./docker/rocm714/run.sh python examples/ros_node_skeleton.py \
  --checkpoint /models/sam3 \
  --onnx-dir /models/onnx_files_504 \
  --video assets/blackswan.mp4 \
  --text swan \
  --policy always_full \
  --max-frames 50 \
  --switch-at 15:swan,water 30:swan
```

Expect both reset messages:

```text
[node] reset_prompts(['swan', 'water']) at source=15
[node] reset_prompts(['swan']) at source=30
```

The indices are zero-based source arrivals, not model frame indices or output
counts. Check the final generation statistics as well; reset messages show
that requests occurred, while the statistics report work in each generation.
A reset can cancel a queued frame, so nonzero dropped or aborted counts during
transitions are expected. Inspect inference failures and drain failures separately.

## Runtime structure

The design optimizes observation freshness for occupancy-grid updates:

    camera callback
        |
        | owned frame copy + host monotonic arrival + sensor timestamp
        v
    LatestFramePipeline raw latest[1]
        |
        | one consumer thread
        v
    preprocess -> backbone -> detector/tracker -> publish

There is no live N+1 preprocessing or backbone lookahead. While inference is
busy, newer camera arrivals replace the one waiting frame before any model
state is touched.

The default policy runs full text detection on every consumed frame using
`SAM3Live`. The separate demo's optional `SAM3HybridLive` mode is described in
[Usage](../docs/usage.md#optional-hybrid-detection); it is not this skeleton's
backend.

## 1. Required threading and ownership rules

1. The subscription callback must not call SAM3Live.infer. It should call
   SAM3Node.on_image and return.
2. LatestFramePipeline uses copy_frames=True in the example. A ROS loaned
   message, cv_bridge shared view, V4L2 buffer, or GStreamer buffer may therefore
   be recycled as soon as on_image returns.
3. Exactly one consumer thread calls infer_next and publishes its result.
   Do not share the wrapped SAM3Live with another inference thread.
4. Configure the upstream transport for latest semantics too. For ROS 2 images,
   use sensor-data QoS, best effort when appropriate, KEEP_LAST, depth 1.
5. Do not pass a ROS, PTP, or camera timestamp as captured_at. The pipeline age
   clock is host-monotonic. The sensor exposure timestamp is carried separately
   as sensor_timestamp for TF and pose lookup.

The callback-facing API is intentionally small:

    accepted = node.on_image(
        frame_bgr,
        header_stamp_ns=msg.header.stamp.sec * 1_000_000_000
                        + msg.header.stamp.nanosec,
    )

accepted=False means a lifecycle transition or shutdown rejected that arrival.
It is not a request to retry an old image.

## 2. ROS 2 wiring

The skeleton does not import rclpy, but maps directly to a node:

1. Construct SAM3Node in on_configure. Model loading and MIG warmup happen once.
2. Start a sensor_msgs/Image subscription in on_activate:

       qos = QoSProfile(
           history=HistoryPolicy.KEEP_LAST,
           depth=1,
           reliability=ReliabilityPolicy.BEST_EFFORT,
       )
       self.create_subscription(Image, topic, self.on_ros_image, qos)

3. The ROS callback converts and submits only:

       def on_ros_image(self, msg):
           frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
           stamp_ns = msg.header.stamp.sec * 1_000_000_000
           stamp_ns += msg.header.stamp.nanosec
           self.sam3.on_image(frame, header_stamp_ns=stamp_ns)

4. Replace SAM3Node._publish_masks with publishers for masks, detections, or a
   costmap input. This function already runs on the sole inference consumer
   thread. If ROS publisher ownership requires a different executor thread,
   enqueue immutable result messages after all tensor-to-CPU conversion.
5. Call node.finish for an orderly finite-source drain. Call node.close from
   on_cleanup/on_shutdown to abort queued work, join the consumer, and close the
   model.

The pipeline queue does not compensate for a deep DDS, camera-driver, RTSP, or
decoder queue. Those queues must also be bounded and configured for low latency.

## 3. Occupancy-grid timestamps and age

Each result retains:

- packet.sequence: host source ordering;
- packet.captured_at: host monotonic arrival used only for latency;
- packet.sensor_timestamp: camera exposure time in the sensor/ROS clock;
- packet.metadata.generation: prompt generation;
- age_ms and service_ms.

Update the map with the transform at sensor_timestamp:

    T_map_camera(sensor_timestamp)

Do not use the robot pose at inference completion. Apply a maximum result-age
gate based on allowable spatial error and relative velocity. The skeleton
defaults `max_result_age_ms` to 200 ms and drops older results before publish;
set a stricter value for faster motion. A stale positive
observation can create obstacle ghosts; stale negative/free-space evidence can
incorrectly clear a current obstacle. Publish and handle empty observations
explicitly rather than silently omitting them.

Tracker-only output must also be treated asymmetrically even when it is fresh:

- `detected=False`: masks that are present may add positive occupied evidence;
  an absent/lost mask is unknown and must not clear free space.
- `detected=True`: the detector ran on this frame. This alone does not prove
  that an area is free of obstacles, even after timestamp and age checks.

`negative_evidence_valid` is a detection-policy gate, not a free-space
guarantee. Clearing also requires application-specific geometric evidence,
visibility and sensor coverage checks, and a policy for the selected semantic
classes. Missing masks can result from missed detections, score filtering,
object caps, or suppression; they must not by themselves clear a costmap.

Applications using `SAM3HybridLive` separately also receive lost IDs in
`lost_object_ids`. Its recovery detection request remains sticky until the
next consumed frame. Latest-frame dropping
means that recovery frame need not be the next source sequence. Use
`redetect_reason == "object_loss"` to identify that recovery keyframe, and use
`negative_evidence_valid` as the detection-policy gate alongside the
application's geometric and observation checks.
Other `redetect_reason` values are `first_frame`, `interval`,
`caller_override`, `inner_forced`, `reset_prompts`, `reset_tracking`,
`detection_retry`, and `None`.

Recorded full-detection and hybrid rates have different workloads and warmup
windows; see [performance records](../docs/performance.md). Treat them as
references, not deadline guarantees. Re-measure p50, p95, and p99 age with the
complete robot stack sharing the GPU.

## 4. Detection policy

The example defaults to AlwaysFull. Every frame that survives latest-frame
selection requests the text detector:

    node = SAM3Node(
        checkpoint="/models/sam3",
        onnx_dir="/models/onnx_files_504",
        prompts=["person", "vehicle", "obstacle"],
        policy=AlwaysFull(),
        imgsz=504,
        mig=True,
    )

Optional policies remain available:

- TimeBasedRedetect: request detection on a host-monotonic interval;
- PeriodicRedetect: request detection every N source arrivals;
- OnDemandTrigger: an external service arms one detection.

These policies operate on the skeleton's direct `SAM3Live` backend. They
schedule detection requests; they do not install `SAM3HybridLive` or its
clean-session replacement and public-ID association. Selecting `time_based`
here is not equivalent to the demo's clean-keyframe hybrid mode and does not
reproduce its reported performance.

Detection requests are sticky across latest-slot replacement, so dropping the
particular camera frame that carried a trigger does not lose the trigger.
AlwaysFull is the recommended default for the full-text occupancy workload.
Tracker-heavy policies need their own map-quality regression because source
frames can be skipped.

The [hybrid measurements and resource checks](../docs/performance.md#optional-hybrid-live-reference)
record single/multi-object scaling, the real-scene quality comparison, and the
fresh-session soak. Those checks do not make tracker-only output valid
negative evidence or replace a long-stream reset policy.

## 5. Prompt reset and generations

Never call live.reset_prompts while a pipeline is active. The skeleton exposes
SAM3Node.reset_prompts, which performs:

    stop accepting into generation G
        -> detach G so late G results are suppressed
        -> close G and wait for its consumer
        -> live.reset_prompts(...)
        -> create and start generation G+1

Callbacks arriving during the transition return False. A queued old-prompt
frame is discarded, and no old-generation result is published after the reset
returns. Include the generation in downstream messages if publication is
followed by another asynchronous queue; consumers can then reject late messages
from an older generation.

The reset service must run outside the inference/publish consumer thread.
`SAM3Node.reset_tracking()` uses the same generation barrier and should be
called periodically according to the robot's memory budget; do not reset the
underlying `SAM3Live` directly.

## 6. Finish and shutdown

Use the two shutdown paths deliberately:

- finish(): stop new submissions, process the final queued latest frame, join
  the consumer, and leave the loaded model available;
- close(): reject new submissions, discard queued work, wait for in-flight
  inference, join the consumer, and close model resources.

The standalone harness calls finish at video EOF and close in a finally block.
A real camera adapter must also provide a way to cancel a blocking camera read;
closing LatestFramePipeline cannot unblock a driver call that it does not own.

## 7. Model output

The result carried by `LatestFrameResult.output` has the normal `SAM3Live`
schema. For this single-object example, `mask` is an HxW NumPy boolean array
at the original image resolution, and box coordinates are source-image pixels:

```python
{
    "object_ids": [3],
    "scores": {3: 0.91},
    "masks": {3: mask},
    "boxes": {3: (40.0, 20.0, 140.0, 180.0)},
    "prompt_to_obj_ids": {"person": [3]},
    "frame_idx": 42,
    "detected": True,
    "negative_evidence_valid": True,
}
```

Every returned object ID has a score, mask, box, and prompt assignment. An
empty result has no object IDs and empty object maps; prompt groups may remain
with empty lists.

Only `SAM3HybridLive` adds these fields; the current `SAM3Node` does not:

| Field | Meaning |
|---|---|
| `keyframe` | Whether this is a clean full-detection keyframe |
| `lost_object_ids` | Public object IDs lost during propagation |
| `redetect_reason` | Why a detection step was requested |

Both direct `SAM3Live` and hybrid callers receive `detected` and
`negative_evidence_valid`; the latter is true exactly when the detector ran.
It is not a clearing guarantee; apply the observation checks in
[Occupancy-grid timestamps and age](#3-occupancy-grid-timestamps-and-age).

For long-running streams, model/session history still needs a bounded reset
policy. Perform any tracking reset with the same stop/join/reset/new-generation
discipline used for prompt changes; never mutate the active SAM3Live directly.
