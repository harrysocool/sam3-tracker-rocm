#!/usr/bin/env python3
"""Offline text-prompt inference for SAM3 images and videos.

Defaults to 504px MIGraphX inference with the fixed DETR decoder and same-frame
detector/tracker parallel tails. Preloaded videos use backbone prefetch for throughput;
all selected frames are processed in order, with full text detection per frame.
Use --no-mig for a PyTorch reference or --no-parallel-tail for serial MIG.

Use --no-fixed-detr-decoder for native-decoder comparisons with MIG enabled.
Camera and ROS consumers use demo_live.py / SAM3Live for latest-frame scheduling.

Usage:
  # Single image, using the supported container and configured artifact mount
  ./docker/rocm714/run.sh python tools/text_baseline.py \\
      --checkpoint /models/sam3 \\
      --image assets/truck.jpg \\
      --text "truck"

  # Video with MIGraphX and backbone prefetch enabled by default
  ./docker/rocm714/run.sh python tools/text_baseline.py \\
      --checkpoint /models/sam3 \\
      --video assets/blackswan.mp4 \\
      --text "swan" \\
      --max-frames 50

Output goes to demo_out/text/<input-stem>_text.{jpg,mp4} unless --output is given.
"""
from __future__ import annotations
import os

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# This script lives under tools/; add the repo root to sys.path so the
# top-level `tracker` package resolves regardless of the invocation cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracker.rocm_env import apply as _apply_rocm_env; _apply_rocm_env()

from transformers import Sam3VideoModel, AutoProcessor
import tracker  # noqa: F401,E402  -- applies ROCm patches (scipy fill_holes + PyTorch NMS)
from tracker.output_processing import (
    DEFAULT_MAX_OBJECTS_PER_PROMPT,
    enforce_per_prompt_cap,
    filter_result,
    postprocess_frame_output,
)



# ─── Args ──────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to model/sam3 (containing model.safetensors)")
    p.add_argument("--text", type=str, nargs="+", required=True,
                   help='One or more prompts; quote a multiword prompt, e.g. "person on a bicycle"')
    p.add_argument("--image", type=Path, default=None,
                   help="Single image input (mutually exclusive with --video)")
    p.add_argument("--video", type=Path, default=None,
                   help="Video input — any mp4 readable by OpenCV")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path. Default: demo_out/text/<input-stem>_text.{jpg,mp4}")
    p.add_argument("--max-frames", type=int, default=120,
                   help="Cap video frames loaded into the session (default 120; 0 = entire video)")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--min-score", type=float, default=0.5,
                   help="Minimum output score on every frame (default 0.5).")
    p.add_argument("--max-objects", type=int, default=DEFAULT_MAX_OBJECTS_PER_PROMPT,
                   help="Maximum tracked objects per prompt (default 5; 0 = unlimited). "
                        "The legacy -1 value also selects the default.")
    p.add_argument("--imgsz", type=int, default=504, choices=(504, 1008),
                   help="Input resolution (default: 504). MIG requires matching "
                        "artifacts; supply --onnx-dir explicitly for 1008px.")
    mig = p.add_mutually_exclusive_group()
    mig.add_argument("--mig", dest="mig", action="store_true",
                     help="Use MIGraphX acceleration (default).")
    mig.add_argument("--no-mig", dest="mig", action="store_false",
                     help="Use the PyTorch reference path.")
    p.set_defaults(mig=True)
    decoder = p.add_mutually_exclusive_group()
    decoder.add_argument(
        "--fixed-detr-decoder",
        dest="fixed_detr_decoder",
        action="store_true",
        help="Use the direct-MXR fixed DETR decoder (default for 504px MIG).",
    )
    decoder.add_argument(
        "--no-fixed-detr-decoder",
        dest="fixed_detr_decoder",
        action="store_false",
        help="Use the native PyTorch DETR decoder while retaining other MIG optimizations.",
    )
    p.set_defaults(fixed_detr_decoder=None)
    parallel = p.add_mutually_exclusive_group()
    parallel.add_argument(
        "--parallel-tail",
        dest="parallel_tail",
        action="store_true",
        help="Overlap detector and tracker tails (default with MIG).",
    )
    parallel.add_argument(
        "--no-parallel-tail",
        dest="parallel_tail",
        action="store_false",
        help="Run serially for comparison; disables automatic backbone prefetch.",
    )
    p.set_defaults(parallel_tail=None)
    prefetch = p.add_mutually_exclusive_group()
    prefetch.add_argument(
        "--pipeline-backbone",
        dest="pipeline_backbone",
        action="store_true",
        help="Prefetch the next frame's backbone (default for MIG video with "
             "parallel tails). Requires a GPU-I/O backbone artifact.",
    )
    prefetch.add_argument(
        "--no-pipeline-backbone",
        dest="pipeline_backbone",
        action="store_false",
        help="Disable next-frame prefetch while retaining same-frame overlap.",
    )
    p.set_defaults(pipeline_backbone=None)
    p.add_argument("--onnx-dir", type=Path, default=None,
                   help="MIG artifact root. Defaults to SAM3_DEFAULT_ONNX_DIR "
                        "from the container wrapper, or onnx_files_504_mgx217.")
    args = p.parse_args(argv)
    if (args.image is None) == (args.video is None):
        sys.exit("Pass exactly one of --image or --video")
    args.text = list(dict.fromkeys(prompt.strip() for prompt in args.text))
    if not all(args.text):
        p.error("--text prompts must not be empty")
    if args.max_objects == -1:
        args.max_objects = DEFAULT_MAX_OBJECTS_PER_PROMPT
    if args.max_objects < 0:
        p.error("--max-objects must be non-negative (or -1 for the default)")
    if args.max_frames < 0:
        p.error("--max-frames must be non-negative; 0 reads to the end")
    if not 0.0 <= args.min_score <= 1.0:
        p.error("--min-score must be between 0 and 1")
    if args.fixed_detr_decoder is None:
        args.fixed_detr_decoder = args.mig and args.imgsz == 504
    if args.fixed_detr_decoder and not args.mig:
        sys.exit("--fixed-detr-decoder requires --mig")
    if args.fixed_detr_decoder and args.imgsz != 504:
        sys.exit("--fixed-detr-decoder requires --imgsz 504")
    if args.parallel_tail is None:
        args.parallel_tail = args.mig
    if args.parallel_tail and not args.mig:
        sys.exit("--parallel-tail requires --mig")
    if args.pipeline_backbone is None:
        args.pipeline_backbone = args.mig and args.parallel_tail and args.video is not None
    if args.pipeline_backbone and not args.parallel_tail:
        sys.exit("--pipeline-backbone requires --parallel-tail")
    if args.pipeline_backbone and args.video is None:
        sys.exit("--pipeline-backbone requires --video")
    if args.onnx_dir is None:
        if args.mig and args.imgsz != 504:
            sys.exit("1008px MIG inference requires an explicit --onnx-dir with matching artifacts")
        args.onnx_dir = Path(
            os.environ.get("SAM3_DEFAULT_ONNX_DIR", "onnx_files_504_mgx217")
        )
    return args


# ─── Helpers ───────────────────────────────────────────────────────────────
def load_video_frames(path: Path, max_n: int):
    if max_n < 0:
        raise ValueError("max_n must be non-negative; 0 reads to the end")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        sys.exit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames_pil, frames_bgr = [], []
    try:
        while max_n == 0 or len(frames_pil) < max_n:
            ret, frame = cap.read()
            if not ret:
                break
            frames_bgr.append(frame)
            frames_pil.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    return frames_pil, frames_bgr, fps


def _process_frame_output(args, processor, session, raw_output, original_size):
    cap = None if args.max_objects == 0 else args.max_objects
    evicted = enforce_per_prompt_cap(session, raw_output.obj_id_to_tracker_score, cap)
    result = postprocess_frame_output(processor, session, raw_output, original_size, evicted)
    result["detected"] = True
    result["negative_evidence_valid"] = True
    return filter_result(result, args.min_score)


# Fixed palette for up to 8 objects (BGR).
_OBJ_COLORS = [
    (0, 200, 80),    # green
    (255, 80, 0),    # blue
    (0, 80, 255),    # red-orange
    (0, 220, 220),   # yellow
    (200, 0, 200),   # magenta
    (0, 200, 255),   # gold
    (180, 255, 100), # lime
    (255, 100, 180), # pink
]


def overlay(bgr: np.ndarray,
            result: dict,
            prompts: list[str],
            frame_idx: int | None = None) -> np.ndarray:
    """Render already-postprocessed masks, colored and labeled by prompt."""
    vis = bgr.copy()
    prompt_colors = {prompt: _OBJ_COLORS[index % len(_OBJ_COLORS)]
                     for index, prompt in enumerate(prompts)}
    object_prompts = {oid: prompt for prompt, ids in result["prompt_to_obj_ids"].items()
                      for oid in ids}
    for obj_id in result["object_ids"]:
        prompt = object_prompts.get(obj_id, "?")
        color = prompt_colors.get(prompt, (255, 255, 255))
        score = result["scores"][obj_id]
        m = result["masks"][obj_id]
        if not m.any():
            continue
        color_arr = np.asarray(color, dtype=np.float32)
        vis[m] = (vis[m] * 0.55 + color_arr * 0.45).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        # Label near top of the largest contour bounding box
        if contours:
            x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
            label_pt = (max(x, 4), max(y - 6, 16))
        else:
            label_pt = (10, 30 + obj_id * 24)
        cv2.putText(vis, f"{prompt} #{obj_id} {score:.2f}", label_pt,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Header: prompt + frame index
    header = ", ".join(prompts)
    if frame_idx is not None:
        header += f"  f={frame_idx}"
    cv2.putText(vis, header, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return vis


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    if args.mig:
        if device.type != "cuda" or torch.version.hip is None:
            sys.exit("MIG inference requires a ROCm GPU; use docker/rocm714/run.sh "
                     "or select --no-mig for the PyTorch reference")
        if not args.onnx_dir.is_dir():
            sys.exit(f"MIG artifact directory not found: {args.onnx_dir}. "
                     "Run setup.sh --models and select the built directory with --onnx-dir.")
        if args.pipeline_backbone:
            gpu_io = args.onnx_dir / "backbone_detector" / "tuned_gpuio.mxr"
            if not gpu_io.is_file():
                sys.exit(f"Backbone prefetch requires {gpu_io}. Build the GPU-I/O "
                         "artifact with setup.sh --models or use --no-pipeline-backbone.")
        if args.fixed_detr_decoder:
            decoder_mxr = args.onnx_dir / "detr_decoder_fixed" / "direct_gpuio.mxr"
            if not decoder_mxr.is_file():
                sys.exit(f"Fixed DETR decoder artifact not found: {decoder_mxr}. "
                         "Run setup.sh --models or use --no-fixed-detr-decoder for diagnosis.")
    print(f"Device: {device}  dtype: {dtype}")
    print(f"Mode: {'MIGraphX' if args.mig else 'PyTorch'}  imgsz={args.imgsz}  "
          f"parallel_tail={args.parallel_tail}  pipeline_backbone={args.pipeline_backbone}  "
          f"fixed_detr_decoder={args.fixed_detr_decoder}")

    print(f"Loading Sam3VideoModel from {args.checkpoint} ...")
    t = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))

    # Build config first so we can rewrite image_size BEFORE module __init__
    # bakes derived sizes (backbone_feature_sizes, RoPE, low_res_mask_size).
    # The config's image_size setter cascades to detector + tracker sub-configs;
    # low_res_mask_size has no setter so we patch it manually.
    from transformers import Sam3VideoConfig
    config = Sam3VideoConfig.from_pretrained(str(args.checkpoint))
    if args.imgsz != 1008:
        config.image_size = args.imgsz
        config.low_res_mask_size = 4 * args.imgsz // 14
        # Processor side: image/video processors carry their own size + mask_size
        # which drive pixel_values shape and output mask shape respectively.
        new_size = {"height": args.imgsz, "width": args.imgsz}
        new_mask = {"height": 4 * args.imgsz // 14, "width": 4 * args.imgsz // 14}
        for sub in (getattr(processor, "image_processor", None),
                    getattr(processor, "video_processor", None)):
            if sub is not None:
                if hasattr(sub, "size"):
                    sub.size = new_size
                if hasattr(sub, "mask_size"):
                    sub.mask_size = new_mask
        if hasattr(processor, "target_size"):
            processor.target_size = args.imgsz
        print(f"  config rewritten: image_size={args.imgsz}, "
              f"low_res_mask_size={config.low_res_mask_size}")

    model = (Sam3VideoModel.from_pretrained(str(args.checkpoint), config=config)
             .to(device).to(dtype).eval())
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  loaded in {time.perf_counter() - t:.1f}s")

    if args.mig:
        print(f"Patching detector_model.vision_encoder with MIGraphX backbone ...")
        t = time.perf_counter()
        from tracker.migraphx_runtime import MIGraphXBackbone
        from tracker.mig_vision_encoder import patch_sam3_video_model_with_mig
        det_dir = args.onnx_dir / "backbone_detector"
        mxr = MIGraphXBackbone(
            onnx_path=det_dir / "single_simplified.onnx",
            cache_path=det_dir / "tuned.mxr",
            gpu_io_cache_path=det_dir / "tuned_gpuio.mxr",
        )
        mxr.warmup(n=2)
        patch_sam3_video_model_with_mig(model, mxr)
        print(f"  MIG backbone ready in {time.perf_counter() - t:.1f}s")

        # Optional: also MIG-ize the DETR encoder (~245 ms PT → ~80 ms MIG per frame).
        detr_onnx = args.onnx_dir / "detector_modules" / "detr_encoder_simplified.onnx"
        if detr_onnx.exists():
            print(f"Patching detector_model.detr_encoder with MIGraphX shim ...")
            from tracker.mig_detr_encoder import patch_sam3_video_model_detr_encoder
            patch_sam3_video_model_detr_encoder(model, detr_onnx)
            print(f"  MIG detr_encoder ready")

        if args.fixed_detr_decoder:
            from tracker.mig_detr_decoder import MIGFixedDetrDecoder

            original_decoder = model.detector_model.detr_decoder
            model.detector_model.detr_decoder = MIGFixedDetrDecoder(
                args.onnx_dir / "detr_decoder_fixed" / "direct_gpuio.mxr",
                original_decoder,
            )
            print("  Fixed DETR decoder direct-MXR enabled")

        # Optional: also MIG-ize memory_attention (steady-state padding).
        # K (pointer-token cap) is resolution-dependent because the MIGraphX
        # MLIR attention kernel has a shape-dependent perf cliff:
        #   504px:  no cliff up to K=64 (deploy K=64 → 16 obj capacity)
        #   1008px: cliff between K=48 and K=56 (deploy K=48 → 12 obj capacity)
        _K_PER_IMGSZ = {504: 64, 1008: 48}
        _k = _K_PER_IMGSZ.get(args.imgsz, 32)
        mem_attn_onnx = args.onnx_dir / "tracker_modules" / f"memory_attention_fixed_S7_P{_k}.onnx"
        if not mem_attn_onnx.exists():
            # Fall back to whatever P-variant exists for this resolution
            for _alt in (32, 48, 64, 16, 4):
                _alt_path = args.onnx_dir / "tracker_modules" / f"memory_attention_fixed_S7_P{_alt}.onnx"
                if _alt_path.exists():
                    mem_attn_onnx = _alt_path
                    break
        if mem_attn_onnx.exists():
            print(f"Patching tracker_model.memory_attention with MIGraphX shim ...")
            from tracker.mig_memory_attention import patch_sam3_video_model_memory_attention
            patch_sam3_video_model_memory_attention(model, mem_attn_onnx)
            print(f"  MIG memory_attention ready (PT fallback for non-steady-state shapes)")
        else:
            print("  (memory_attention artifact absent; retaining PyTorch attention)")

        # Batched mask_decoder: ~2× speedup on multi-object propagation
        # (single-object falls through to original per-obj path with no overhead).
        from tracker.batched_mask_decoder import patch_batched_mask_decoder
        patch_batched_mask_decoder(model)
        print(f"  Batched mask_decoder patch applied (active for N>1 obj)")

        if args.parallel_tail:
            from tracker.parallel_video import patch_parallel_video_tail
            patch_parallel_video_tail(model)
            print("  Parallel detector/tracker tail enabled")
            if args.pipeline_backbone:
                print("  Cross-frame backbone prefetch enabled")

    try:
        return _run_input(args, processor, model, device, dtype)
    finally:
        if args.parallel_tail:
            from tracker.parallel_video import close_parallel_video_tail
            close_parallel_video_tail(model)


def _run_input(args, processor, model, device, dtype):
    # Collect frames
    if args.image is not None:
        bgr0 = cv2.imread(str(args.image))
        if bgr0 is None:
            sys.exit(f"Cannot read image: {args.image}")
        frames_bgr = [bgr0]
        frames_pil = [Image.fromarray(cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB))]
        fps = 1.0
    else:
        print(f"Loading video frames (max {args.max_frames or 'all'}) ...")
        frames_pil, frames_bgr, fps = load_video_frames(args.video, args.max_frames)
        if not frames_pil:
            sys.exit(f"No frames decoded from {args.video}")
        print(f"  loaded {len(frames_pil)} frames @ ~{fps:.1f} fps")

    # Init session + prompt
    print("Initialising session ...")
    t = time.perf_counter()
    session = processor.init_video_session(
        video=frames_pil, inference_device=device, dtype=dtype,
    )
    # Keep offline input storage separate from the ordered tracking history.
    # HF uses session.num_frames for tracker temporal encoding, and switches
    # hotstart rules based on whether forward receives a frame. Let the model
    # insert only consumed frames, exactly as SAM3Live.infer does; future
    # backbone prefetch must not advance the tracking session.
    preloaded_frames = session.processed_frames
    session.processed_frames = {}
    processor.add_text_prompt(session, args.text)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  ready in {(time.perf_counter() - t) * 1000:.0f} ms")

    # Frame 0 — detection + tracker init
    print(f"Detecting {args.text} on frame 0 ...")
    t = time.perf_counter()
    with torch.inference_mode():
        out0 = model(inference_session=session, frame=preloaded_frames[0], frame_idx=0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    init_ms = (time.perf_counter() - t) * 1000
    result0 = _process_frame_output(args, processor, session, out0, frames_bgr[0].shape[:2])
    print(f"  init: {init_ms:.0f} ms  →  {len(result0['object_ids'])} output object(s)")
    for obj_id in result0["object_ids"]:
        score = result0["scores"][obj_id]
        print(f"  object #{obj_id}  score={score:.2f}")
    if not result0["object_ids"]:
        print("  No objects above the output threshold on frame 0.")

    # Output path
    if args.output is not None:
        out_path = args.output
    elif args.image is not None:
        out_path = Path("demo_out/text") / f"{args.image.stem}_text.jpg"
    else:
        out_path = Path("demo_out/text") / f"{args.video.stem}_text.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Single image — done
    if args.image is not None:
        vis = overlay(frames_bgr[0], result0, args.text)
        cv2.imwrite(str(out_path), vis)
        print(f"Saved: {out_path}")
        return 0

    # Video — propagate frame 1..N-1
    H, W = frames_bgr[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))
    n_total = len(frames_pil)
    print(f"Propagating through frames 1..{n_total - 1} ...")
    t_prop = time.perf_counter()

    def serial_outputs():
        for frame_idx in range(1, n_total):
            with torch.inference_mode():
                output = model(inference_session=session, frame=preloaded_frames[frame_idx],
                               frame_idx=frame_idx)
            yield frame_idx, output

    def consume_outputs(outputs):
        for frame_idx, output in outputs:
            result = _process_frame_output(
                args, processor, session, output, frames_bgr[frame_idx].shape[:2],
            )
            writer.write(
                overlay(frames_bgr[frame_idx], result, args.text, frame_idx=frame_idx)
            )
            if frame_idx % 20 == 0:
                elapsed = time.perf_counter() - t_prop
                print(
                    f"  frame {frame_idx}/{n_total - 1}  "
                    f"({frame_idx / elapsed:.1f} prop FPS)"
                )

    try:
        if not writer.isOpened():
            raise RuntimeError(f"Cannot write video: {out_path}")
        writer.write(overlay(frames_bgr[0], result0, args.text, frame_idx=0))
        if args.pipeline_backbone:
            from tracker.backbone_pipeline import BackbonePrefetchPipeline

            with BackbonePrefetchPipeline(
                model, session, frame_provider=preloaded_frames.__getitem__,
            ) as pipeline:
                consume_outputs(pipeline.run(range(1, n_total)))
        else:
            consume_outputs(serial_outputs())
        if device.type == "cuda":
            torch.cuda.synchronize()
    finally:
        writer.release()

    elapsed = time.perf_counter() - t_prop
    n_prop = n_total - 1
    print(f"\nPropagation: {n_prop} frames in {elapsed:.1f}s = {n_prop / elapsed:.2f} FPS")
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
