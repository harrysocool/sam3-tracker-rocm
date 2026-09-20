#!/usr/bin/env python3
"""Canonical arrival-paced benchmark for the production latest-frame path.

The harness intentionally fixes the released S7/C4 tracker-memory policy and
rejects the historical BENCH_* overrides. It is a throughput/latency benchmark,
not a correctness substitute; run the documented PT-vs-MIG mask gate first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import traceback
from pathlib import Path

from tracker.rocm_env import apply as _apply_rocm_env

_apply_rocm_env()

import cv2
import numpy as np
import onnxruntime as ort
import torch

from tracker.latest_frame import LatestFramePipeline
from tracker.live_inference import SAM3Live


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def _load_artifact_identity(onnx_dir: Path) -> tuple[dict, str]:
    manifest_path = onnx_dir / "ARTIFACT_MANIFEST.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"canonical benchmark requires a valid artifact manifest: {manifest_path}"
        ) from exc
    if manifest.get("schema") != 2:
        raise RuntimeError(
            f"unsupported artifact manifest schema: {manifest.get('schema')!r}"
        )
    if manifest.get("source", {}).get("dirty") is not False:
        raise RuntimeError("canonical benchmark refuses artifacts built from dirty source")
    logical = "backbone_detector/tuned_gpuio.mxr"
    rows = {
        row.get("path"): row for row in manifest.get("files", [])
        if isinstance(row, dict)
    }
    try:
        expected = rows[logical]["sha256"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"manifest does not identify {logical}") from exc
    return manifest, expected


def _reject_noncanonical_environment(environ=None) -> None:
    env = os.environ if environ is None else environ
    forbidden = [
        name for name in (
            "BENCH_NUM_MASKMEM",
            "BENCH_MAX_COND_FRAMES",
            "BENCH_KEEP_RECENT",
        )
        if env.get(name)
    ]
    if forbidden:
        raise RuntimeError(
            "canonical benchmark does not allow tracker-memory overrides: "
            + ", ".join(forbidden)
        )


def _current_ec_power_mode(environ=None) -> tuple[str | None, str]:
    env = os.environ if environ is None else environ
    override = env.get("SAM3_EC_POWER_MODE", "").strip().lower()
    if override:
        return override, "SAM3_EC_POWER_MODE"
    path = Path(
        env.get(
            "SAM3_EC_POWER_MODE_PATH",
            "/sys/class/ec_su_axb35/apu/power_mode",
        )
    )
    try:
        return path.read_text(encoding="ascii").strip().lower(), str(path)
    except OSError:
        return None, str(path)


def _validate_execution_identity(manifest: dict, checkpoint: Path,
                                 environ=None) -> None:
    env = os.environ if environ is None else environ
    checkpoint_file = checkpoint / "model.safetensors"
    expected_checkpoint = manifest.get("checkpoint", {}).get("sha256")
    if not expected_checkpoint or _sha256(checkpoint_file) != expected_checkpoint:
        raise RuntimeError("checkpoint SHA256 does not match artifact manifest")

    expected_image = manifest.get("build", {}).get("image_id")
    current_image = env.get("SAM3_DOCKER_IMAGE_ID", "").strip()
    if not current_image or current_image != expected_image:
        raise RuntimeError(
            f"container image does not match artifact manifest: "
            f"{current_image!r} != {expected_image!r}"
        )

    expected_commit = manifest.get("source", {}).get("commit")
    current_commit = env.get("SAM3_SOURCE_COMMIT", "").strip()
    if not current_commit or current_commit != expected_commit:
        raise RuntimeError(
            f"source revision does not match artifact manifest: "
            f"{current_commit!r} != {expected_commit!r}"
        )
    if env.get("SAM3_SOURCE_DIRTY", "0") == "1":
        raise RuntimeError("canonical benchmark refuses a dirty source checkout")

    mode, source = _current_ec_power_mode(env)
    if mode != "performance":
        raise RuntimeError(
            f"canonical benchmark requires EC performance mode, got "
            f"{mode!r} via {source}"
        )


def _warmup_and_reset(live: SAM3Live, video: Path, prompt: str) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open warmup video: {video}")
    try:
        for index in range(2):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"warmup frame {index} unavailable")
            live.infer(frame, full_detection=True)
        torch.cuda.synchronize(device=live.device)
    finally:
        cap.release()
    # Full reset: reset_tracking() alone retains object-id/fusion bookkeeping.
    live.reset_prompts([prompt])


def _capture(
    video: Path,
    loops: int,
    fps: float,
    pipeline: LatestFramePipeline,
    state: dict,
) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        state["error"] = f"cannot open {video}"
        pipeline.abort()
        return
    sequence = 0
    period = 1.0 / fps
    started_at = time.perf_counter()
    state["capture_started_at"] = started_at
    try:
        for loop_index in range(loops):
            if loop_index and not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                raise RuntimeError("failed to seek source video")
            source_index = 0
            while True:
                scheduled_at = started_at + sequence * period
                delay = scheduled_at - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                ok, frame = cap.read()
                if not ok:
                    break
                arrived_at = time.perf_counter()
                state["arrivals"].append(arrived_at)
                if not pipeline.submit(
                    frame,
                    sequence=sequence,
                    captured_at=arrived_at,
                    full_detection=True,
                    metadata={
                        "loop_index": loop_index,
                        "source_index": source_index,
                        "scheduled_at": scheduled_at,
                    },
                ):
                    return
                state["captured"] += 1
                sequence += 1
                source_index += 1
    except BaseException as exc:
        state["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        pipeline.abort()
    finally:
        cap.release()
        state["capture_finished_at"] = time.perf_counter()
        pipeline.finish_input()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--text", default="swan")
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--capture-fps", type=float, default=24.0)
    parser.add_argument("--warm-outputs", type=int, default=5)
    parser.add_argument("--tail-outputs", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.loops < 1 or args.capture_fps <= 0 or args.warm_outputs < 0:
        raise ValueError("invalid loop/fps/warm-output configuration")
    _reject_noncanonical_environment()
    manifest, expected_backbone_sha = _load_artifact_identity(args.onnx_dir)
    _validate_execution_identity(manifest, args.checkpoint)
    backbone = args.onnx_dir / "backbone_detector" / "tuned_gpuio.mxr"
    backbone_sha = _sha256(backbone)
    if backbone_sha != expected_backbone_sha:
        raise RuntimeError(
            f"unexpected backbone artifact: {backbone_sha} != "
            f"{expected_backbone_sha}"
        )
    providers = ort.get_available_providers()
    if ort.__version__ != "1.24.2" or not providers \
            or providers[0] != "MIGraphXExecutionProvider":
        raise RuntimeError(
            "canonical benchmark requires ONNX Runtime 1.24.2 with MIGraphX "
            f"as the primary provider, got {ort.__version__} / {providers}"
        )

    live = SAM3Live(
        checkpoint=args.checkpoint,
        prompts=[args.text],
        onnx_dir=args.onnx_dir,
        imgsz=504,
        dtype=torch.float16,
        device="cuda",
        mig=True,
        parallel_tail=True,
        redetect_every=1,
        max_objects_per_prompt=1,
        bootstrap_frames=0,
        periodic_rebootstrap_seconds=0,
    )
    tracker = live.model.tracker_model
    memory_contract = {
        "num_maskmem": int(tracker.num_maskmem),
        "max_cond_frame_num": int(tracker.config.max_cond_frame_num),
    }
    if memory_contract != {"num_maskmem": 7, "max_cond_frame_num": 4}:
        live.close()
        raise RuntimeError(
            f"canonical benchmark requires original S7/C4, got {memory_contract}"
        )
    memory_shim = tracker.memory_attention._mig_shim
    _warmup_and_reset(live, args.video, args.text)

    pipeline = LatestFramePipeline(live)
    pipeline.start()
    state = {"captured": 0, "arrivals": [], "error": None}
    producer = threading.Thread(
        target=_capture,
        args=(args.video, args.loops, args.capture_fps, pipeline, state),
        name="sam3-latest-frame-benchmark-capture",
    )
    records = []
    wall_start = time.perf_counter()
    producer.start()
    try:
        while True:
            result = pipeline.infer_next()
            if result is None:
                break
            records.append(
                {
                    "source_sequence": result.packet.sequence,
                    "source_loop": result.packet.metadata["loop_index"],
                    "source_index": result.packet.metadata["source_index"],
                    "model_frame_idx": result.output["frame_idx"],
                    "captured_at": result.packet.captured_at,
                    "inference_started_at": result.inference_started_at,
                    "completed_at": result.completed_at,
                    "queue_wait_ms": result.queue_wait_ms,
                    "service_ms": result.service_ms,
                    "frame_age_ms": result.age_ms,
                    "object_count": len(result.output["object_ids"]),
                }
            )
    finally:
        pipeline.close()
        producer.join(timeout=5.0)
        live.close()
    wall_end = time.perf_counter()
    if producer.is_alive():
        raise RuntimeError("capture thread did not terminate")
    if state["error"] is not None:
        raise RuntimeError(f"capture failed: {state['error']}")

    warm = records[min(args.warm_outputs, len(records)) :]
    completion = [record["completed_at"] for record in records]
    intervals = [
        (completion[index] - completion[index - 1]) * 1000.0
        for index in range(max(1, args.warm_outputs), len(completion))
    ]
    tail = records[-min(args.tail_outputs, len(records)) :]
    tail_intervals = [
        (tail[index]["completed_at"] - tail[index - 1]["completed_at"]) * 1000.0
        for index in range(1, len(tail))
    ]
    interval_summary = _summary(intervals)
    tail_interval_summary = _summary(tail_intervals)
    pipeline_stats = pipeline.stats()
    errors = []
    if pipeline_stats["failed_frames"] or pipeline_stats["drain_failed"]:
        errors.append(f"pipeline failure: {pipeline_stats}")
    if memory_shim._pt_fallback_calls:
        errors.append(
            f"memory attention used PyTorch fallback {memory_shim._pt_fallback_calls} times"
        )
    if any(record["object_count"] != 1 for record in records):
        errors.append("expected exactly one tracked object on every output")
    report = {
        "runtime": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "onnxruntime": ort.__version__,
            "providers": providers,
        },
        "config": {
            "video": str(args.video.resolve()),
            "prompt": args.text,
            "loops": args.loops,
            "capture_fps": args.capture_fps,
            "copy_frames": True,
            "full_detection_every_consumed_frame": True,
            "nplus1_gpu_work": False,
            "parallel_tail": True,
            "fixed_detr_decoder": True,
            "tracker_memory": memory_contract,
        },
        "artifact": {
            "onnx_dir": str(args.onnx_dir.resolve()),
            "backbone_gpuio": str(backbone.resolve()),
            "backbone_gpuio_sha256": backbone_sha,
            "manifest": str((args.onnx_dir / "ARTIFACT_MANIFEST.json").resolve()),
            "source": manifest["source"],
            "checkpoint": manifest["checkpoint"],
            "build": manifest["build"],
        },
        "captured_frames": state["captured"],
        "output_frames": len(records),
        "pipeline_stats": pipeline_stats,
        "memory_attention": {
            "mig_calls": int(memory_shim._mig_calls),
            "pytorch_fallback_calls": int(memory_shim._pt_fallback_calls),
        },
        "validation": {"passed": not errors, "errors": errors},
        "first_source_sequence": records[0]["source_sequence"] if records else None,
        "last_source_sequence": records[-1]["source_sequence"] if records else None,
        "wall_seconds": wall_end - wall_start,
        "warm_discard_outputs": min(args.warm_outputs, len(records)),
        "warm_output_interval_ms": interval_summary,
        "warm_output_hz": (
            1000.0 / interval_summary["mean"] if interval_summary["mean"] else None
        ),
        "warm_frame_age_ms": _summary([r["frame_age_ms"] for r in warm]),
        "warm_queue_wait_ms": _summary([r["queue_wait_ms"] for r in warm]),
        "warm_service_ms": _summary([r["service_ms"] for r in warm]),
        "tail_output_interval_ms": tail_interval_summary,
        "tail_output_hz": (
            1000.0 / tail_interval_summary["mean"]
            if tail_interval_summary["mean"]
            else None
        ),
        "capture_interval_ms": _summary(
            [
                (state["arrivals"][index] - state["arrivals"][index - 1]) * 1000.0
                for index in range(1, len(state["arrivals"]))
            ]
        ),
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "captured": report["captured_frames"],
                "outputs": report["output_frames"],
                "last_sequence": report["last_source_sequence"],
                "hz": report["warm_output_hz"],
                "age_p50_ms": report["warm_frame_age_ms"]["p50"],
                "age_p95_ms": report["warm_frame_age_ms"]["p95"],
                "queue_p95_ms": report["warm_queue_wait_ms"]["p95"],
                "service_p50_ms": report["warm_service_ms"]["p50"],
                "service_mean_ms": report["warm_service_ms"]["mean"],
                "service_p95_ms": report["warm_service_ms"]["p95"],
                "stats": pipeline_stats,
                "memory_attention": report["memory_attention"],
                "errors": errors,
            },
            indent=2,
        ),
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
