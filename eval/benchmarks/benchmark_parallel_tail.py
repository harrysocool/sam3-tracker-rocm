#!/usr/bin/env python3
"""A/B benchmark serial versus parallel Sam3VideoModel tail scheduling.

The same loaded model and input frames are used for both schedules.  Each run
gets a fresh inference session, and odd repeats reverse the A/B order to reduce
temperature and clock bias.  Per-object masks, IDs, prompt ownership, and
scores are compared frame by frame.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

import tracker  # noqa: E402,F401 -- install ROCm compatibility patches
from tracker.mig_detr_encoder import patch_sam3_video_model_detr_encoder  # noqa: E402
from tracker.mig_memory_attention import patch_sam3_video_model_memory_attention  # noqa: E402
from tracker.mig_vision_encoder import patch_sam3_video_model_with_mig  # noqa: E402
from tracker.migraphx_runtime import MIGraphXBackbone  # noqa: E402
from tracker.batched_mask_decoder import patch_batched_mask_decoder  # noqa: E402
from tracker.parallel_video import (  # noqa: E402
    close_parallel_video_tail,
    patch_parallel_video_tail,
)
import transformers  # noqa: E402
from transformers import AutoProcessor, Sam3VideoConfig, Sam3VideoModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("model/sam3"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("onnx_files_504"))
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--text",
        action="append",
        required=True,
        help="Prompt to add; repeat for a multi-prompt benchmark.",
    )
    parser.add_argument("--imgsz", type=int, default=504, choices=(504, 1008))
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--steady-frames",
        type=int,
        default=20,
        help="Number of final propagation frames used for the steady-state summary.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def configure_processor(processor, image_size: int) -> None:
    mask_size = 4 * image_size // 14
    for subprocessor in (
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
    ):
        if subprocessor is not None:
            if hasattr(subprocessor, "size"):
                subprocessor.size = {"height": image_size, "width": image_size}
            if hasattr(subprocessor, "mask_size"):
                subprocessor.mask_size = {"height": mask_size, "width": mask_size}
    if hasattr(processor, "target_size"):
        processor.target_size = image_size


def load_frames(path: Path, limit: int) -> list[Image.Image]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.release()
    if len(frames) < 2:
        raise RuntimeError(f"need at least two frames from {path}")
    return frames


def build_model(args):
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    config = Sam3VideoConfig.from_pretrained(str(args.checkpoint))
    if args.imgsz != 1008:
        config.image_size = args.imgsz
        config.low_res_mask_size = 4 * args.imgsz // 14
        configure_processor(processor, args.imgsz)

    model = (
        Sam3VideoModel.from_pretrained(str(args.checkpoint), config=config)
        .to("cuda")
        .half()
        .eval()
    )
    detector_dir = args.onnx_dir / "backbone_detector"
    backbone = MIGraphXBackbone(
        detector_dir / "single_simplified.onnx",
        detector_dir / "tuned.mxr",
        gpu_io_cache_path=detector_dir / "tuned_gpuio.mxr",
    )
    backbone.warmup(2)
    patch_sam3_video_model_with_mig(model, backbone)
    patch_sam3_video_model_detr_encoder(
        model,
        args.onnx_dir / "detector_modules" / "detr_encoder_simplified.onnx",
    )
    pointer_tokens = {504: 64, 1008: 48}[args.imgsz]
    patch_sam3_video_model_memory_attention(
        model,
        args.onnx_dir
        / "tracker_modules"
        / f"memory_attention_fixed_S7_P{pointer_tokens}.onnx",
    )
    patch_batched_mask_decoder(model)
    return processor, model


def run_once(processor, model, frames, prompts, parallel: bool, steady_frames: int):
    if parallel:
        patch_parallel_video_tail(model)
    else:
        close_parallel_video_tail(model)

    session = processor.init_video_session(
        video=frames,
        inference_device=torch.device("cuda"),
        dtype=torch.float16,
    )
    for prompt in prompts:
        processor.add_text_prompt(session, prompt)

    timings = []
    outputs = []
    for frame_idx in range(len(frames)):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = model(inference_session=session, frame_idx=frame_idx)
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)

        prompt_by_object = {
            object_id: session.prompts[session.obj_id_to_prompt_id[object_id]]
            for object_id in output.object_ids
        }
        masks = {
            object_id: output.obj_id_to_mask[object_id]
            .detach()
            .float()
            .cpu()
            .numpy()
            .squeeze()
            > 0
            for object_id in output.object_ids
            if object_id in output.obj_id_to_mask
        }
        outputs.append(
            {
                "ids": list(output.object_ids),
                "prompts": prompt_by_object,
                "scores": {
                    object_id: float(output.obj_id_to_score[object_id])
                    for object_id in output.object_ids
                },
                "masks": masks,
            }
        )

    propagation = timings[1:]
    steady = propagation[-min(steady_frames, len(propagation)) :]
    return {
        "schedule": "parallel" if parallel else "serial",
        "mean_ms": statistics.mean(propagation),
        "median_ms": statistics.median(propagation),
        "p95_ms": float(np.percentile(propagation, 95)),
        "steady_mean_ms": statistics.mean(steady),
        "steady_median_ms": statistics.median(steady),
        "timings_ms": timings,
        "outputs": outputs,
    }


def compare_outputs(serial, parallel):
    ious = []
    maximum_score_difference = 0.0
    ids_equal = True
    prompts_equal = True
    for serial_frame, parallel_frame in zip(serial["outputs"], parallel["outputs"]):
        ids_equal &= serial_frame["ids"] == parallel_frame["ids"]
        prompts_equal &= serial_frame["prompts"] == parallel_frame["prompts"]
        for object_id in set(serial_frame["masks"]) | set(parallel_frame["masks"]):
            left = serial_frame["masks"].get(object_id)
            right = parallel_frame["masks"].get(object_id)
            if left is None or right is None:
                ious.append(0.0)
                continue
            union = np.logical_or(left, right).sum()
            ious.append(
                1.0 if union == 0 else float(np.logical_and(left, right).sum() / union)
            )
        for object_id in set(serial_frame["scores"]) | set(parallel_frame["scores"]):
            left = serial_frame["scores"].get(object_id, float("inf"))
            right = parallel_frame["scores"].get(object_id, float("-inf"))
            maximum_score_difference = max(maximum_score_difference, abs(left - right))
    return {
        "ids_equal": ids_equal,
        "prompts_equal": prompts_equal,
        "mask_count": len(ious),
        "mean_mask_iou": statistics.mean(ious) if ious else 1.0,
        "minimum_mask_iou": min(ious, default=1.0),
        "maximum_score_abs_diff": maximum_score_difference,
    }


def strip_outputs(result):
    return {key: value for key, value in result.items() if key != "outputs"}


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.steady_frames < 1:
        raise ValueError("--steady-frames must be at least 1")
    frames = load_frames(args.video, args.max_frames)
    processor, model = build_model(args)

    # Exercise every shape-specialized session before collecting timings.
    run_once(processor, model, frames, args.text, False, args.steady_frames)
    run_once(processor, model, frames, args.text, True, args.steady_frames)

    pairs = []
    for repeat in range(args.repeats):
        order = (False, True) if repeat % 2 == 0 else (True, False)
        results = {
            parallel: run_once(
                processor,
                model,
                frames,
                args.text,
                parallel,
                args.steady_frames,
            )
            for parallel in order
        }
        serial = results[False]
        parallel = results[True]
        comparison = compare_outputs(serial, parallel)
        pairs.append(
            {
                "serial": strip_outputs(serial),
                "parallel": strip_outputs(parallel),
                "correctness": comparison,
            }
        )
        print(
            f"repeat {repeat + 1}: "
            f"serial={serial['steady_mean_ms']:.2f} ms "
            f"parallel={parallel['steady_mean_ms']:.2f} ms "
            f"min_iou={comparison['minimum_mask_iou']:.6f}"
        )

    serial_mean = statistics.mean(pair["serial"]["steady_mean_ms"] for pair in pairs)
    parallel_mean = statistics.mean(pair["parallel"]["steady_mean_ms"] for pair in pairs)
    summary = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
            "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(),
            "device": torch.cuda.get_device_name(),
        },
        "video": str(args.video),
        "prompts": args.text,
        "imgsz": args.imgsz,
        "frames": len(frames),
        "steady_frames": min(args.steady_frames, len(frames) - 1),
        "repeats": args.repeats,
        "serial_mean_ms": serial_mean,
        "parallel_mean_ms": parallel_mean,
        "latency_reduction_pct": 100.0 * (1.0 - parallel_mean / serial_mean),
        "throughput_gain_pct": 100.0 * (serial_mean / parallel_mean - 1.0),
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "pairs"}, indent=2))
    print(f"Saved: {args.out}")
    close_parallel_video_tail(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
