# Usage and examples

Start with the [Quick start](../README.md#quick-start). Run commands below from
the repository root, with `SAM3_MODEL_DIR` and `SAM3_ONNX_DIR` exported to the
model directory and locally built MIGraphX 2.17 artifact directory.
All current GPU examples use [the supported container](../docker/rocm714/README.md).

## Live video

`demo_live.py` simulates a live source by pacing a video file at its declared
frame rate. `SAM3Live` is the underlying streaming API; camera / ROS callbacks
should use the [integration skeleton](../examples/README.md), not the file demo.

```bash
# Default: full detection on every consumed frame
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan --max-frames 60

# Multiple prompts, same defaults
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan water --max-frames 60
```

Output defaults to `results/<video-stem>_live_<timestamp>.mp4`; override it with
`--output`. Container-relative output paths under `/workspace` refer to the
checkout on the host.

### Scheduling and ownership

```text
continuous capture -> owned latest[1] -> one ordered inference owner -> result
```

While inference is busy, a newer arrival replaces the one waiting frame.
Preprocessing and all GPU work for that waiting frame start only after the
current inference completes. There is no live N+1 preprocessing or backbone
lookahead. Same-frame detector/tracker parallel tails are independent of this
policy and are enabled by default for MIG.

`--max-frames` counts source arrivals. Fewer output frames are expected, and
the emitted-only MP4 is time-compressed when frames drop. It is not a
wall-clock recording or proof of the model's throughput.

### Optional hybrid detection

```bash
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan \
  --redetect-interval-ms 1000
```

A positive interval opts into `SAM3HybridLive`: full-detection keyframes plus
native detector-skip tracking between them, using one `SAM3Live` model.
Each clean keyframe replaces the inner session while reusing encoded prompt
tensors. Same-prompt mask-IoU matching associates fresh detections with stable
public IDs; unmatched detections can change IDs. Only one inner session is
active at a time. No separate legacy box-tracker backbone is loaded.

If a tracker-only result loses an object, `lost_object_ids` records the loss
and a sticky request forces detection on the next consumed frame. That may
not be the next source frame because the latest slot can drop arrivals.
The native no-object gate is preserved.

**Mapping safety:** tracker-only masks may add positive occupied evidence;
an absent mask must not clear free space. Use `negative_evidence_valid` as the
detection gate, then apply exposure-time pose alignment and a result-age budget.
Full output fields, recovery reasons, and reset rules are documented in the
[integration guide](../examples/README.md).

### Live parameters

| Parameter | Default / behavior |
|---|---|
| `--text` | One or more prompts; quote a multiword prompt, e.g. `--text "person on a bike"` |
| `--imgsz` | 504; the default fixed decoder supports 504px only |
| `--redetect-interval-ms` | 0: full detection on every consumed frame; positive: opt-in hybrid |
| `--min-score` | 0.5: output confidence filter |
| `--max-objects` | -1 selects the API default of 5 per prompt; positive values set a per-prompt cap; 0 explicitly removes the cap |
| `--max-frames` | 0: whole input; otherwise cap source frames, not outputs |
| `--warmup-frames` | 0: no explicit prewarm; positive values pre-run file frames, reset tracking, and seek back |
| `--output` | Override the emitted-frame MP4 path |

Unlimited live objects are not recommended: accumulated detections can make
tracker work grow substantially. The offline object's cap has different
semantics; do not assume these defaults apply to `text_baseline.py`.

MIG, same-frame parallel tails, and the available 504px fixed decoder are
the optimized defaults. `--no-mig`, `--no-parallel-tail`, and
`--no-fixed-detr-decoder` are diagnostic switches, not alternate supported
deployment stacks. Never load MIGraphX 2.17 MXR files with host 2.16 libraries.

For all advanced flags:

```bash
./docker/rocm714/run.sh python demo_live.py --help
```

Do not change prompts or reset tracking on an active inference owner. Close
the pipeline, reset the session, and start a new generation. The integration
skeleton provides this lifecycle; the demo does not switch prompts mid-stream.

## Offline text reference

`tools/text_baseline.py` uses HF `Sam3VideoModel` with a preloaded video session.
It processes the selected video frames in order rather than dropping stale
waiting frames. Both this path and default live run detection on every
processed frame; their scheduling and output-rate measurements differ.

Unlike live, this tool defaults to **pure PyTorch and 1008px**. Set `--imgsz 504`
and `--onnx-dir` explicitly when using the artifacts built by Quick start:

```bash
# Offline MIG at 504px
./docker/rocm714/run.sh python tools/text_baseline.py \
  --checkpoint /models/sam3 --onnx-dir /models/onnx_files_504 \
  --video assets/blackswan.mp4 --text swan --imgsz 504 --mig --max-frames 60

# Pure-PyTorch video reference
./docker/rocm714/run.sh python tools/text_baseline.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan --imgsz 504 --max-frames 60

# Pure-PyTorch single-image reference
./docker/rocm714/run.sh python tools/text_baseline.py \
  --checkpoint /models/sam3 \
  --image assets/truck.jpg --text truck --imgsz 504
```

Output defaults to `demo_out/text/<input-stem>_text.{jpg,mp4}`; override with
`--output`. Use short noun phrases such as `"swan"`, `"yellow taxi"`, or
`"a person on a bike"`.

- `--min-score` defaults to 0.5.
- `--max-objects` defaults to 0 (all qualifying objects); a positive cap selects
  objects by frame-0 detection score, rather than live's per-prompt cap.
- `--parallel-tail` is opt-in here and requires `--mig`.
- `--pipeline-backbone` additionally requires video input and `--parallel-tail`.
  It overlaps frame N+1's stateless backbone with frame N's tail, adds pipeline
  fill, and retains an extra frame. This offline-only optimization is not a
  live latency improvement.

The current offline tool uses the native PyTorch DETR decoder rather than
live's auto-loaded fixed decoder. Likewise, a pure-PyTorch reference is not
the same graph configuration as default live. See [measurement scope](evaluation.md).

## Box-prompt reference

`demo_box.py` uses `SAM3OnnxTracker` for single-object tracking from a supplied
frame-0 box. It skips text detection. The box is `x1,y1,x2,y2` in source pixels.

This path needs a separate tracker backbone and tracker decoder artifacts;
**the text/live Quick start does not build that artifact set**. Its historical
12.21 FPS and DAVIS J result are not current full-text live measurements.
See [archived box usage and evaluation](historical/legacy-runtime.md#box-prompt-usage)
for the old commands and artifact context. Do not mix that runtime's MXR files
into the supported MIGraphX 2.17 live root.

## Visual examples

These are qualitative examples from the reference tools. They are not
recordings of the current live throughput or frame-age measurements.

### Offline text detection and tracking

| `"swan"` | `"camel"` | `"pig"` (3 objects) |
|:---:|:---:|:---:|
| <img src="images/demo_swan_text_mig.gif" width="260" alt="swan text-prompt segmentation"> | <img src="images/demo_camel_text_mig.gif" width="260" alt="camel text-prompt segmentation"> | <img src="images/demo_pigs_multi_object.gif" width="260" alt="three pigs tracked with a text prompt"> |

### Box-prompt reference

| truck — single image | dog-agility — video |
|:---:|:---:|
| <img src="images/demo_tracked.jpg" width="400" alt="truck box-prompt segmentation"> | <img src="images/demo_dog_agility_box.gif" width="400" alt="dog-agility box-prompt tracking"> |

## Source map

| Area | Entry points |
|---|---|
| Runtime / model build | [setup.sh](../setup.sh), [container wrapper](../docker/rocm714/run.sh), [text exporter](../export/build_text_prompt_mig.py) |
| Streaming / hybrid API | [live_inference.py](../tracker/live_inference.py), [hybrid_inference.py](../tracker/hybrid_inference.py) |
| Latest-frame scheduling | [latest_frame.py](../tracker/latest_frame.py) |
| Same-frame overlap / offline lookahead | [parallel_video.py](../tracker/parallel_video.py), [backbone_pipeline.py](../tracker/backbone_pipeline.py) |
| Fixed decoder | [mig_detr_decoder.py](../tracker/mig_detr_decoder.py) |
| Benchmarks and regressions | [evaluation guide](evaluation.md), [eval/](../eval/) |
