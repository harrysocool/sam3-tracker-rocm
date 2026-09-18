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
  --video assets/blackswan.mp4 --text swan --max-frames 50

# Multiple prompts, same defaults
./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan water --max-frames 50
```

Output defaults to `results/<video-stem>_live_<timestamp>.mp4`; override it with
`--output`. Container-relative output paths under `/workspace` refer to the
checkout on the host.

### Use your own video

The wrapper mounts the repository at `/workspace`. Copy your input into the
ignored `results/inputs/` directory and use its container path:

```bash
mkdir -p results/inputs
cp /absolute/path/to/my-video.mp4 results/inputs/my-video.mp4

./docker/rocm714/run.sh python demo_live.py \
  --checkpoint /models/sam3 \
  --video /workspace/results/inputs/my-video.mp4 \
  --text person \
  --output /workspace/results/my-video-segmented.mp4
```

Replace the input path and prompt with your own. The output is saved as
`results/my-video-segmented.mp4` in the host checkout.

An arbitrary host path such as `/home/user/videos/my-video.mp4` is not visible
inside the container unless it lies in one of the configured mounts. A video
symlink pointing outside those mounts is insufficient. See the
[mount reference](../docker/rocm714/README.md#directories-and-mounts) for the
host-to-container mapping.

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
  --video assets/blackswan.mp4 --text swan --max-frames 50 \
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
| `--max-objects` | 5 per prompt; positive values set a persistent per-prompt cap; 0 explicitly removes the cap |
| `--num-maskmem` | Full mode defaults to 3 tracker-memory frames; hybrid retains 7; pass 7 to restore the previous full-mode horizon |
| `--max-cond-frames` | Full mode defaults to the nearest conditioning frame only; hybrid retains its previous four-conditioning-frame horizon |
| `--max-frames` | 0: whole input; otherwise cap source frames, not outputs |
| `--warmup-frames` | 0: no explicit prewarm; positive values pre-run file frames, reset tracking, and seek back |
| `--output` | Override the emitted-frame MP4 path |

Accumulated detections can make tracker work grow substantially. Both live and
offline default to five tracked objects per prompt; use an unlimited setting
only when the workload calls for it.

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

## Offline text inference

`tools/text_baseline.py` uses HF `Sam3VideoModel`. It preloads input frames and
advances the tracking session one frame at a time, processing all selected
frames in order rather than dropping stale waiting frames. Both this path and
default live run detection on every processed frame; their scheduling and
output-rate measurements differ.

The tool defaults to **504px MIGraphX inference with the fixed DETR decoder**,
same-frame parallel tails, and next-frame backbone prefetch for videos. It uses
the artifact mount configured by Quick start through `SAM3_DEFAULT_ONNX_DIR`.
Single-image runs do not prefetch another frame.

Like live, `--text` accepts multiple prompts and quoted multiword phrases.
The default `--max-frames 120` limits the number of input frames preloaded; use
`--max-frames 0` to read the entire video. Full-video loading requires memory
for the selected input frames and inference session, while live reads frames
as they arrive and defaults to no source-frame limit.

```bash
# Accelerated offline video, using the defaults
./docker/rocm714/run.sh python tools/text_baseline.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan --max-frames 50

# Multiple prompts, at most two tracked objects per prompt
./docker/rocm714/run.sh python tools/text_baseline.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan water \
  --max-objects 2 --max-frames 50

# Accelerated single image, using the defaults
./docker/rocm714/run.sh python tools/text_baseline.py \
  --checkpoint /models/sam3 \
  --image assets/truck.jpg --text truck

# Explicit PyTorch video reference at the same resolution
./docker/rocm714/run.sh python tools/text_baseline.py \
  --checkpoint /models/sam3 \
  --video assets/blackswan.mp4 --text swan --no-mig --max-frames 50
```

For either images or videos, add `--no-mig` to select the PyTorch reference.
It disables the fixed decoder, parallel tails, and backbone prefetch.
Add `--imgsz 1008` for the original-resolution PyTorch reference. MIG at 1008px
requires an explicit `--onnx-dir` containing separately built 1008px artifacts;
the 504px Quick start artifacts cannot be used at that resolution.

Output defaults to `demo_out/text/<input-stem>_text.{jpg,mp4}`; override with
`--output`. Use short noun phrases such as `"swan"`, `"yellow taxi"`, or
`"a person on a bike"`.

- `--min-score` defaults to 0.5 and filters output on every frame.
- `--max-objects` defaults to 5 per prompt. After each inference, excess objects
  are removed from the session using their current tracker scores, matching
  live's policy. Use 0 to leave the session uncapped.
- Same-frame parallel tails are enabled by default with MIG. Use
  `--no-parallel-tail` for serial MIG comparison; it also disables automatic
  backbone prefetch.
- Backbone prefetch is enabled by default for MIG videos with parallel tails.
  Use `--no-pipeline-backbone` to retain same-frame overlap without lookahead.
  Prefetch overlaps frame N+1's stateless backbone with frame N's tail, adds
  pipeline fill, and retains an extra frame. It improves offline throughput;
  live scheduling remains unchanged.
- The fixed DETR decoder is enabled by default for 504px MIG and requires
  `detr_decoder_fixed/direct_gpuio.mxr`. Use `--no-fixed-detr-decoder` for a
  native-decoder comparison while keeping the other MIG optimizations.
  The 1008px path uses the native decoder.

The offline and live 504px paths use the same fixed decoder implementation.
Their scheduling and timing windows differ: offline prefetch optimizes file
throughput, while live consumes the newest available frame. See
[measurement scope](evaluation.md) before comparing their rates.

### Shared output behavior

Both CLI tools use the same object-limit and postprocessing helpers. They
resize mask logits to the original image size before binarization, apply the
processor's suppression and per-prompt overlap rules, and return the same
mask, box, score, object-ID, and prompt-ownership fields before rendering.
Both apply `--min-score` (default 0.5) to every output frame.

The Python API `SAM3Live.infer()` returns standard postprocessed outputs
without applying the CLI's score threshold. Direct API callers can apply
`tracker.output_processing.filter_result(result, min_score=0.5)` when needed.
This output filter does not remove objects from the tracking session; the
per-prompt object cap is enforced separately.

Offline keeps preloaded inputs separate from tracking history and passes each
consumed frame through the same streaming model entry point as live. Future
backbone prefetch does not advance the tracking session. This also aligns
the upstream model's session-length-dependent tracker encoding and hotstart
rules; preloading a video is an input-storage choice, not a different tracking
policy.

An empty result is valid. Offline video processing continues after an empty
first frame so targets can be detected later, and frames whose objects fall
below the score threshold are still written to the output video.

Different frame-selection policies can still produce different tracking
histories: comparisons with live must replay the same selected input frames.

## Historical box-prompt reference

`demo_box.py` and `SAM3OnnxTracker` are retained for historical box-prompt and
DAVIS regression work. They are outside the current supported text/live build
and release smoke. The tracker uses a supplied frame-0 box, expressed as
`x1,y1,x2,y2` in source pixels, and skips text detection.

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

### Historical box-prompt examples

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
| Object limits and output processing | [output_processing.py](../tracker/output_processing.py) |
| Benchmarks and regressions | [evaluation guide](evaluation.md), [eval/](../eval/) |
