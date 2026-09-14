# SAM3Live freshness-first integration guide

This directory shows the deployment shape for a camera, ROS 2 image topic, or
other real-time source. The primary example is
[ros_node_skeleton.py](ros_node_skeleton.py).

Complete the [Quick start](../README.md#quick-start) first. The standalone
commands below use the supported container with `SAM3_MODEL_DIR` and
`SAM3_ONNX_DIR` exported to the model and MIGraphX 2.17 artifact directories.

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

The default robotics policy still runs full text detection on every consumed
frame. `demo_live.py --redetect-interval-ms N`, with `N > 0`, is an opt-in
unified hybrid: one `SAM3Live` model performs full detection on
wall-clock keyframes and native detector-skip tracking between them. Before a
keyframe it replaces the inner session with a fresh one, reuses encoded prompt
tensors, and maps the new detections to stable public IDs by same-prompt mask
IoU. There is only one active session at a time. It does not create a second
tracker backbone or load legacy MIGraphX 2.16 tracker artifacts.

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
- `detected=True`: negative/free-space clearing is permitted after the normal
  timestamp and result-age checks.

The unified hybrid reports lost IDs in `lost_object_ids` and makes the recovery
detection request sticky until the next consumed frame. Latest-frame dropping
means that recovery frame need not be the next source sequence. Use
`redetect_reason == "object_loss"` to identify that recovery keyframe, and use
`negative_evidence_valid` as the direct gate for clearing operations.
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

## 7. Standalone source-paced check

The example reads at the video file's declared FPS instead of processing the
file as fast as possible:

    ./docker/rocm714/run.sh python examples/ros_node_skeleton.py \
        --checkpoint /models/sam3 \
        --onnx-dir /models/onnx_files_504 \
        --video assets/blackswan.mp4 \
        --text swan \
        --policy always_full \
        --max-frames 200

The default policy is always_full, and MIG enables parallel detector/tracker
tails and the fixed 504px DETR decoder automatically when its artifact is
present. Use `--no-parallel-tail` or `--no-fixed-detr-decoder` only for
diagnosis. The report separates source callbacks,
accepted submissions, emitted outputs, rejected transition frames, inference
service time, frame age, and per-generation drop counters.

Prompt generation changes can be exercised with:

    --switch-at 60:person,vehicle 120:floor,wall

The indices are source arrivals, not model frame indices or output counts.

## 8. Model output

The result carried by LatestFrameResult.output has the normal SAM3Live schema:

    {
        "object_ids":         [3, 7, 12],
        "scores":             {3: 0.91, 7: 0.83},
        "masks":              {3: HxW_bool_array},
        "boxes":              {3: (x1, y1, x2, y2)},
        "prompt_to_obj_ids":  {"person": [3, 7], "car": [12]},
        "frame_idx":          42,
        "detected":           True,
        "keyframe":           True,
        "lost_object_ids":    [],
        "redetect_reason":    "interval",
        "negative_evidence_valid": True,
    }

`keyframe`, `lost_object_ids`, and `redetect_reason` are supplied by
`SAM3HybridLive`. Both direct `SAM3Live` and hybrid callers always receive
`detected` and `negative_evidence_valid`; the latter is true exactly when the
detector actually ran. Costmap integrations should read
`negative_evidence_valid` directly, not infer clearing permission from an
empty mask.

For long-running streams, model/session history still needs a bounded reset
policy. Perform any tracking reset with the same stop/join/reset/new-generation
discipline used for prompt changes; never mutate the active SAM3Live directly.
