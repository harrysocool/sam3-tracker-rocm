#!/usr/bin/env python3
"""A/B benchmark serial versus parallel Sam3VideoModel tail scheduling.

The same loaded model and input frames are used for both schedules.  Each run
gets a fresh inference session, and odd repeats reverse the A/B order to reduce
temperature and clock bias.  Per-object masks, IDs, prompt ownership, and
scores are compared frame by frame.
"""
from __future__ import annotations

import argparse
import hashlib
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
from tracker.backbone_pipeline import BackbonePrefetchPipeline  # noqa: E402
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
        "--pipeline-backbone",
        action="store_true",
        help="Also prefetch frame N+1's backbone in the parallel candidate.",
    )
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


def run_once(
    processor,
    model,
    frames,
    prompts,
    parallel: bool,
    steady_frames: int,
    pipeline_backbone: bool = False,
):
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

    def record(frame_idx, output, elapsed_ms):
        timings.append(elapsed_ms)
        prompt_by_object = {
            object_id: session.prompts[session.obj_id_to_prompt_id[object_id]]
            for object_id in output.object_ids
        }
        masks = {
            object_id: output.obj_id_to_mask[object_id]
            .detach()
            .cpu()
            .numpy()
            .squeeze()
            for object_id in output.object_ids
            if object_id in output.obj_id_to_mask
        }
        outputs.append(
            {
                "frame_idx": frame_idx,
                "ids": list(output.object_ids),
                "prompts": prompt_by_object,
                "scores": {
                    object_id: float(output.obj_id_to_score[object_id])
                    for object_id in output.object_ids
                },
                "masks": masks,
            }
        )

    # Frame 0 cannot overlap a tracker tail or a future frame's backbone.
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(inference_session=session, frame_idx=0)
    torch.cuda.synchronize()
    record(0, output, (time.perf_counter() - start) * 1000)

    if pipeline_backbone:
        with BackbonePrefetchPipeline(model, session) as pipeline:
            iterator = iter(pipeline.run(range(1, len(frames))))
            while True:
                torch.cuda.synchronize()
                start = time.perf_counter()
                try:
                    frame_idx, output = next(iterator)
                except StopIteration:
                    break
                torch.cuda.synchronize()
                record(frame_idx, output, (time.perf_counter() - start) * 1000)
    else:
        for frame_idx in range(1, len(frames)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                output = model(inference_session=session, frame_idx=frame_idx)
            torch.cuda.synchronize()
            record(frame_idx, output, (time.perf_counter() - start) * 1000)

    propagation = timings[1:]
    # A pipelined sequence has a fill sample at the front and a drain sample
    # at the end. Exclude both when reporting steady-state output cadence;
    # keeping only the drain would systematically overstate the speedup.
    if len(propagation) > 2:
        steady_pool = propagation[1:-1]
        steady_index_pool = list(range(2, len(frames) - 1))
    else:
        steady_pool = propagation
        steady_index_pool = list(range(1, len(frames)))
    steady_count = min(steady_frames, len(steady_pool))
    steady = steady_pool[-steady_count:]
    steady_indices = steady_index_pool[-steady_count:]
    return {
        "schedule": (
            "parallel+backbone-prefetch"
            if pipeline_backbone
            else ("parallel" if parallel else "serial")
        ),
        "mean_ms": statistics.mean(propagation),
        "median_ms": statistics.median(propagation),
        "p95_ms": float(np.percentile(propagation, 95)),
        "steady_mean_ms": statistics.mean(steady),
        "steady_median_ms": statistics.median(steady),
        "steady_p95_ms": float(np.percentile(steady, 95)),
        "steady_frame_indices": steady_indices,
        "pipeline_fill_ms": propagation[0] if pipeline_backbone else None,
        "pipeline_drain_ms": propagation[-1] if pipeline_backbone else None,
        "timings_ms": timings,
        "outputs": outputs,
    }

def compare_outputs(serial, parallel):
    ious = []
    maximum_score_difference = 0.0
    maximum_mask_abs_difference = 0.0
    frame_indices_equal = len(serial["outputs"]) == len(parallel["outputs"])
    ids_equal = frame_indices_equal
    prompts_equal = frame_indices_equal
    mask_shapes_equal = frame_indices_equal
    score_keys_equal = frame_indices_equal
    for serial_frame, parallel_frame in zip(serial["outputs"], parallel["outputs"]):
        frame_indices_equal &= serial_frame["frame_idx"] == parallel_frame["frame_idx"]
        ids_equal &= serial_frame["ids"] == parallel_frame["ids"]
        prompts_equal &= serial_frame["prompts"] == parallel_frame["prompts"]
        for object_id in set(serial_frame["masks"]) | set(parallel_frame["masks"]):
            left = serial_frame["masks"].get(object_id)
            right = parallel_frame["masks"].get(object_id)
            if left is None or right is None:
                mask_shapes_equal = False
                ious.append(0.0)
                continue
            if left.shape != right.shape:
                mask_shapes_equal = False
                ious.append(0.0)
                continue
            maximum_mask_abs_difference = max(
                maximum_mask_abs_difference,
                float(np.max(np.abs(left.astype(np.float32) - right.astype(np.float32)))),
            )
            left_binary = left > 0
            right_binary = right > 0
            union = np.logical_or(left_binary, right_binary).sum()
            ious.append(
                1.0
                if union == 0
                else float(np.logical_and(left_binary, right_binary).sum() / union)
            )
        serial_score_ids = set(serial_frame["scores"])
        parallel_score_ids = set(parallel_frame["scores"])
        score_keys_equal &= serial_score_ids == parallel_score_ids
        for object_id in serial_score_ids & parallel_score_ids:
            left = serial_frame["scores"][object_id]
            right = parallel_frame["scores"][object_id]
            maximum_score_difference = max(maximum_score_difference, abs(left - right))
    return {
        "frame_indices_equal": frame_indices_equal,
        "ids_equal": ids_equal,
        "prompts_equal": prompts_equal,
        "mask_shapes_equal": mask_shapes_equal,
        "score_keys_equal": score_keys_equal,
        "mask_count": len(ious),
        "mean_mask_iou": statistics.mean(ious) if ious else 1.0,
        "minimum_mask_iou": min(ious, default=1.0),
        "maximum_mask_abs_diff": maximum_mask_abs_difference,
        "maximum_score_abs_diff": maximum_score_difference,
    }


def strip_outputs(result):
    summary = {key: value for key, value in result.items() if key != "outputs"}
    summary["output_signatures"] = []
    for frame in result["outputs"]:
        summary["output_signatures"].append(
            {
                "frame_idx": frame["frame_idx"],
                "ids": frame["ids"],
                "prompts": frame["prompts"],
                "scores": frame["scores"],
                "masks": {
                    object_id: {
                        "shape": list(mask.shape),
                        "positive_pixels": int((mask > 0).sum()),
                        "sha256": hashlib.sha256(
                            np.ascontiguousarray(mask).view(np.uint8)
                        ).hexdigest(),
                    }
                    for object_id, mask in frame["masks"].items()
                },
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if args.steady_frames < 1:
        raise ValueError("--steady-frames must be at least 1")
    frames = load_frames(args.video, args.max_frames)
    processor, model = build_model(args)

    modes = [("serial", False, False), ("parallel", True, False)]
    if args.pipeline_backbone:
        modes.append(("parallel_pipeline", True, True))

    # Exercise every shape-specialized session and schedule before timing.
    for _, parallel, pipeline_backbone in modes:
        run_once(
            processor,
            model,
            frames,
            args.text,
            parallel,
            args.steady_frames,
            pipeline_backbone=pipeline_backbone,
        )

    pairs = []
    for repeat in range(args.repeats):
        order = modes if repeat % 2 == 0 else list(reversed(modes))
        results = {
            name: run_once(
                processor,
                model,
                frames,
                args.text,
                parallel,
                args.steady_frames,
                pipeline_backbone=pipeline_backbone,
            )
            for name, parallel, pipeline_backbone in order
        }
        serial = results["serial"]
        comparisons = {
            name: compare_outputs(serial, result)
            for name, result in results.items()
            if name != "serial"
        }
        pairs.append(
            {
                "runs": {name: strip_outputs(result) for name, result in results.items()},
                "correctness_vs_serial": comparisons,
            }
        )
        timing = " ".join(
            f"{name}={result['steady_mean_ms']:.2f}ms"
            for name, result in results.items()
        )
        minimum_iou = min(
            comparison["minimum_mask_iou"] for comparison in comparisons.values()
        )
        print(f"repeat {repeat + 1}: {timing} min_iou={minimum_iou:.6f}")

    means = {
        name: statistics.mean(
            pair["runs"][name]["steady_mean_ms"] for pair in pairs
        )
        for name, _, _ in modes
    }
    amortized_means = {
        name: statistics.mean(pair["runs"][name]["mean_ms"] for pair in pairs)
        for name, _, _ in modes
    }
    candidate_name = "parallel_pipeline" if args.pipeline_backbone else "parallel"
    serial_mean = means["serial"]
    candidate_mean = means[candidate_name]
    source_paths = (
        "tracker/parallel_video.py",
        "tracker/backbone_pipeline.py",
        "tracker/migraphx_runtime.py",
        "tracker/mig_vision_encoder.py",
        "tracker/ort_gpu_io.py",
        "eval/benchmarks/benchmark_parallel_tail.py",
    )
    summary = {
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=WORKSPACE
        ).strip(),
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "transformers": transformers.__version__,
            "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(),
            "device": torch.cuda.get_device_name(),
        },
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, cwd=WORKSPACE
        ).splitlines(),
        "source_sha256": {
            path: hashlib.sha256((WORKSPACE / path).read_bytes()).hexdigest()
            for path in source_paths
        },
        "video": str(args.video),
        "prompts": args.text,
        "imgsz": args.imgsz,
        "frames": len(frames),
        "steady_frames": len(
            pairs[0]["runs"]["serial"]["steady_frame_indices"]
        ),
        "steady_frame_indices": pairs[0]["runs"]["serial"][
            "steady_frame_indices"
        ],
        "repeats": args.repeats,
        "pipeline_backbone": args.pipeline_backbone,
        "steady_mean_ms": means,
        "steady_fps": {name: 1000.0 / value for name, value in means.items()},
        "amortized_mean_ms": amortized_means,
        "amortized_fps": {
            name: 1000.0 / value for name, value in amortized_means.items()
        },
        "latency_reduction_pct": 100.0 * (1.0 - candidate_mean / serial_mean),
        "throughput_gain_pct": 100.0 * (serial_mean / candidate_mean - 1.0),
        "pairs": pairs,
    }
    if args.pipeline_backbone:
        parallel_mean = means["parallel"]
        summary["prefetch_incremental_latency_reduction_pct"] = 100.0 * (
            1.0 - candidate_mean / parallel_mean
        )
        summary["prefetch_incremental_throughput_gain_pct"] = 100.0 * (
            parallel_mean / candidate_mean - 1.0
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "pairs"}, indent=2))
    print(f"Saved: {args.out}")
    close_parallel_video_tail(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
