#!/usr/bin/env python3
"""Headless installation smoke, not a throughput or mask-quality benchmark.

Run inside docker/rocm714/run.sh with an installed 504px artifact directory.
Frames are submitted one at a time and consumed synchronously: there is no
arrival pacing, capture thread, N+1 preprocessing, rendering, or intentional
dropping. Hybrid detection requests use explicit frame indices, not a timer.
Use a known-positive video/prompt; each mode must produce a non-empty mask.

GPU and video dependencies are imported only after argument parsing, so help
and the validation/orchestration helpers can be tested on a CPU-only host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _frame_count(value: str) -> int:
    count = int(value)
    if count < 3:
        raise argparse.ArgumentTypeError("--frames must be at least 3")
    return count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=ROOT / "assets/blackswan.mp4")
    parser.add_argument("--text", default="swan", help="One known-positive text prompt")
    parser.add_argument("--frames", type=_frame_count, default=12)
    parser.add_argument("--mode", choices=("full", "hybrid", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True, help="Installation-smoke JSON")
    args = parser.parse_args(argv)
    args.text = args.text.strip()
    if not args.text:
        parser.error("--text must not be empty")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict:
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _git(root: Path, *args):
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def source_identity(root: Path = ROOT) -> dict:
    version_file = root / "VERSION"
    result = {
        "version": version_file.read_text().strip() if version_file.is_file() else None,
        "git_head": None, "git_tree": None, "git_diff_sha256": None,
        "git_status_short": None,
        "git_diff_scope": "tracked changes versus HEAD; untracked files are listed in status",
        "source_files_sha256": {},
    }
    for relative in (
        "tools/smoke_live_release.py", "tracker/live_inference.py",
        "tracker/hybrid_inference.py", "tracker/latest_frame.py",
        "tracker/mig_detr_decoder.py",
    ):
        path = root / relative
        if path.is_file():
            result["source_files_sha256"][relative] = _sha256(path)
    git_root = _git(root, "rev-parse", "--show-toplevel")
    if git_root is not None and Path(git_root.decode().strip()).resolve() == root.resolve():
        for key, command in (
            ("git_head", ("rev-parse", "HEAD")),
            ("git_tree", ("rev-parse", "HEAD^{tree}")),
        ):
            value = _git(root, *command)
            result[key] = value.decode().strip() if value is not None else None
        diff = _git(root, "diff", "--binary", "HEAD", "--")
        result["git_diff_sha256"] = hashlib.sha256(diff).hexdigest() if diff is not None else None
        status = _git(root, "status", "--short", "--untracked-files=all")
        result["git_status_short"] = status.decode().splitlines() if status is not None else None
    return result


def artifact_identity(checkpoint: Path, onnx_dir: Path, video: Path) -> dict:
    result = {
        "checkpoint_weights": _file_identity(checkpoint / "model.safetensors"),
        "checkpoint_config": _file_identity(checkpoint / "config.json"),
        "backbone": _file_identity(onnx_dir / "backbone_detector/tuned_gpuio.mxr"),
        "fixed_decoder": _file_identity(onnx_dir / "detr_decoder_fixed/direct_gpuio.mxr"),
        "video": _file_identity(video),
    }
    manifest = onnx_dir / "manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        _require(isinstance(data, dict), "artifact manifest must be a JSON object")
        result["manifest"] = _file_identity(manifest)
        result["manifest_metadata"] = {
            key: data.get(key) for key in ("format", "format_version", "release", "profile", "source", "runtime")
        }
    return result


def validate_runtime(version: str, providers) -> None:
    _require(version == "1.24.2", f"ORT 1.24.2 is required, got {version}")
    _require("MIGraphXExecutionProvider" in providers,
             f"MIGraphXExecutionProvider is required, got {list(providers)}")


def create_live(args, mode: str, full_class, hybrid_class):
    kwargs = dict(
        checkpoint=args.checkpoint, prompts=[args.text], onnx_dir=args.onnx_dir,
        imgsz=504, mig=True, device="cuda", bootstrap_frames=0,
        periodic_rebootstrap_seconds=0,
    )
    # Deliberately use the production defaults for fixed decoder and parallel tail.
    if mode == "hybrid":
        return hybrid_class(**kwargs, redetect_interval_ms=1000.0)
    return full_class(**kwargs)


def _load_runtime(args):
    import onnxruntime as ort

    providers = ort.get_available_providers()
    validate_runtime(ort.__version__, providers)
    import torch
    import migraphx

    _require(torch.cuda.is_available() and torch.version.hip is not None,
             "A ROCm GPU is required; run through docker/rocm714/run.sh")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tracker.live_inference import SAM3Live
    from tracker.hybrid_inference import SAM3HybridLive
    from tracker.latest_frame import LatestFramePipeline

    properties = torch.cuda.get_device_properties(0)
    return SimpleNamespace(
        environment={
            "python": platform.python_version(), "platform": platform.platform(),
            "onnxruntime": ort.__version__, "providers": providers,
            "torch": torch.__version__, "hip": torch.version.hip,
            "migraphx_version": getattr(migraphx, "__version__", None),
            "migraphx_module": migraphx.__file__,
            "gpu_name": properties.name, "gpu_arch": getattr(properties, "gcnArchName", None),
        },
        create_live=lambda mode: create_live(args, mode, SAM3Live, SAM3HybridLive),
        pipeline_class=LatestFramePipeline,
        synchronize=torch.cuda.synchronize,
    )


def read_frames(video: Path, count: int) -> list:
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        _require(capture.isOpened(), f"Cannot open video: {video}")
        frames = []
        for index in range(count):
            ok, frame = capture.read()
            _require(ok and frame is not None,
                     f"Video has fewer than {count} decodable frames (stopped at {index})")
            frames.append(frame)
        return frames
    finally:
        capture.release()


def _ids(values, label: str) -> list[int]:
    _require(isinstance(values, (list, tuple)), f"{label} must be a list/tuple")
    _require(all(isinstance(value, numbers.Integral) and not isinstance(value, bool)
                 and value >= 0 for value in values), f"{label} must contain non-negative integer IDs")
    result = [int(value) for value in values]
    _require(len(result) == len(set(result)), f"{label} contains duplicate IDs")
    return result


def validate_output(output, frame_shape, prompts) -> dict:
    import numpy as np

    _require(isinstance(output, dict), "Inference output must be a dict")
    ids = _ids(output.get("object_ids"), "object_ids")
    ids_set = set(ids)
    for name in ("scores", "masks", "boxes"):
        values = output.get(name)
        _require(isinstance(values, dict), f"{name} must be a dict")
        _require(set(_ids(list(values), name)) == ids_set, f"{name} keys do not match object_ids")
    ownership = output.get("prompt_to_obj_ids")
    _require(isinstance(ownership, dict), "prompt_to_obj_ids must be a dict")
    _require(set(ownership) <= set(prompts), "Unexpected prompt ownership")
    owned = []
    for prompt, values in ownership.items():
        owned.extend(_ids(values, f"prompt {prompt!r}"))
    _require(len(owned) == len(set(owned)), "Object ID belongs to multiple prompts")
    _require(set(owned) == ids_set, "Prompt ownership does not match object_ids")
    nonempty = 0
    for object_id in ids:
        score = output["scores"][object_id]
        _require(isinstance(score, numbers.Real) and not isinstance(score, bool)
                 and math.isfinite(float(score)), f"Non-finite or invalid score for {object_id}")
        mask = output["masks"][object_id]
        _require(isinstance(mask, np.ndarray) and mask.dtype == np.bool_,
                 f"Mask {object_id} must be a bool ndarray")
        _require(mask.shape == tuple(frame_shape[:2]), f"Mask {object_id} has wrong shape {mask.shape}")
        box = np.asarray(output["boxes"][object_id], dtype=float)
        _require(box.shape == (4,) and bool(np.isfinite(box).all()), f"Invalid box for {object_id}")
        nonempty += int(mask.any())
    detected = output.get("detected")
    negative = output.get("negative_evidence_valid")
    _require(isinstance(detected, (bool, np.bool_)), "detected must be a bool")
    _require(isinstance(negative, (bool, np.bool_)) and bool(negative) == bool(detected),
             "negative_evidence_valid must match actual detection")
    frame_idx = output.get("frame_idx")
    _require(isinstance(frame_idx, numbers.Integral) and not isinstance(frame_idx, bool)
             and frame_idx >= 0, "frame_idx must be a non-negative integer")
    return {
        "frame_idx": int(frame_idx), "object_ids": ids, "nonempty_masks": nonempty,
        "detected": bool(detected), "negative_evidence_valid": bool(negative),
        "redetect_reason": output.get("redetect_reason"),
    }


def validate_pipeline_stats(stats: dict, count: int) -> None:
    for key in ("submitted_frames", "output_frames"):
        _require(stats.get(key) == count, f"Unexpected pipeline {key}: {stats.get(key)}")
    for key in ("dropped_frames", "failed_frames", "aborted_frames", "queued_frames"):
        _require(stats.get(key) == 0, f"Unexpected pipeline {key}: {stats.get(key)}")
    _require(stats.get("drain_failed") is False, "Pipeline GPU drain failed or was not reported")


def _error(exc: Exception, stage: str) -> dict:
    return {"stage": stage, "type": type(exc).__name__, "message": str(exc)}


def run_mode(mode: str, frames: list, text: str, runtime) -> dict:
    _require(mode in ("full", "hybrid") and len(frames) >= 3, "Smoke needs a valid mode and at least 3 frames")
    keyframes = {0, max(2, len(frames) // 2)}
    requests = [mode == "full" or index in keyframes for index in range(len(frames))]
    result = {
        "mode": mode, "passed": False, "errors": [], "frames_processed": 0,
        "detection_requests": requests, "warmup_requests": [True, mode == "full"],
        "warmup": [], "records": [], "pipeline_stats": None,
        "lifecycle": {
            "live_created": False, "warmup_completed": False,
            "tracking_reset_after_warmup": False, "pipeline_started": False,
            "input_drained": False, "inference_synchronized": False,
            "pipeline_closed": False, "live_closed": False, "close_order": [],
        },
    }
    life = result["lifecycle"]
    live = pipeline = None
    completed = False
    try:
        live = runtime.create_live(mode)
        life["live_created"] = True
        backend = getattr(live, "live", live)
        _require(getattr(backend, "parallel_tail", False) is True, "Default parallel tail is not enabled")
        decoder = getattr(backend, "_fixed_detr_decoder", None)
        _require(decoder is not None, "Default fixed DETR decoder was not loaded")
        result["loaded_defaults"] = {"parallel_tail": True, "fixed_decoder_class": type(decoder).__name__}
        for index, request in enumerate(result["warmup_requests"]):
            output = live.infer(frames[index], full_detection=request)
            record = validate_output(output, frames[index].shape, [text])
            result["warmup"].append(record)
            _require(record["detected"] == request,
                     "Warmup did not exercise the requested detection/propagation path; use a known-positive clip")
        runtime.synchronize()
        life["warmup_completed"] = True
        live.reset_tracking()
        life["tracking_reset_after_warmup"] = True
        pipeline = runtime.pipeline_class(live, copy_frames=True)
        pipeline.start()
        life["pipeline_started"] = True
        for index, (frame, request) in enumerate(zip(frames, requests)):
            _require(pipeline.submit(frame, sequence=index, full_detection=request), "Pipeline refused a frame")
            item = pipeline.infer_next()
            _require(item is not None and item.packet.sequence == index, "Missing or out-of-order pipeline result")
            record = validate_output(item.output, frame.shape, [text])
            _require(mode != "full" or record["detected"], "Full mode unexpectedly skipped detection")
            result["records"].append({"source_sequence": index, **record})
            result["frames_processed"] += 1
        pipeline.finish_input()
        _require(pipeline.infer_next() is None, "Pipeline did not reach end of input")
        life["input_drained"] = True
        runtime.synchronize()
        life["inference_synchronized"] = True
        result["detected_frames"] = sum(row["detected"] for row in result["records"])
        result["propagated_frames"] = len(frames) - result["detected_frames"]
        result["nonempty_masks"] = sum(row["nonempty_masks"] for row in result["records"])
        _require(result["nonempty_masks"] > 0, "No non-empty masks; use a known-positive video/prompt")
        _require(result["detected_frames"] > 0, "No measured full detection")
        _require(mode != "hybrid" or result["propagated_frames"] > 0, "No measured hybrid propagation")
        validate_pipeline_stats(pipeline.stats(), len(frames))
        completed = True
    except Exception as exc:
        result["errors"].append(_error(exc, "inference_smoke"))
    finally:
        if pipeline is not None:
            life["close_order"].append("pipeline")
            try:
                pipeline.close()
                life["pipeline_closed"] = True
            except Exception as exc:
                result["errors"].append(_error(exc, "pipeline_close"))
            try:
                result["pipeline_stats"] = pipeline.stats()
                if completed:
                    validate_pipeline_stats(result["pipeline_stats"], len(frames))
            except Exception as exc:
                result["errors"].append(_error(exc, "pipeline_stats"))
        if live is not None:
            life["close_order"].append("live")
            try:
                live.close()
                life["live_closed"] = True
            except Exception as exc:
                result["errors"].append(_error(exc, "live_close"))
    result["passed"] = completed and not result["errors"]
    return result


def main(argv=None) -> int:
    args = parse_args(argv)
    report = {
        "schema_version": 1, "kind": "installation_smoke", "passed": False,
        "scope": "structural/lifecycle smoke only; not a performance benchmark or mask-quality regression",
        "scheduling": "serial submit/infer_next; explicit detection requests; no arrival pacing or N+1 work",
        "config": {"checkpoint": str(args.checkpoint), "onnx_dir": str(args.onnx_dir),
                   "video": str(args.video), "text": args.text, "frames": args.frames,
                   "mode": args.mode, "imgsz": 504, "bootstrap_frames": 0},
        "modes": [], "errors": [],
    }
    exit_code = 1
    try:
        report["source"] = source_identity()
        runtime = _load_runtime(args)
        report["environment"] = runtime.environment
        report["artifacts"] = artifact_identity(args.checkpoint, args.onnx_dir, args.video)
        frames = read_frames(args.video, args.frames)
        modes = ("full", "hybrid") if args.mode == "both" else (args.mode,)
        for mode in modes:
            result = run_mode(mode, frames, args.text, runtime)
            report["modes"].append(result)
            if not result["passed"]:
                break  # Do not start another model after a failed GPU/lifecycle check.
        report["passed"] = len(report["modes"]) == len(modes) and all(row["passed"] for row in report["modes"])
        exit_code = 0 if report["passed"] else 1
    except KeyboardInterrupt:
        report["errors"].append({"stage": "setup_or_run", "type": "KeyboardInterrupt", "message": "Interrupted"})
        exit_code = 130
    except Exception as exc:
        report["errors"].append(_error(exc, "setup_or_run"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Installation smoke {'PASS' if report['passed'] else 'FAIL'}: {args.output}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
