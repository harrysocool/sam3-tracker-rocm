#!/usr/bin/env python3
"""Canonical SAM3 ViT benchmark for the two validated XDNA2 NPU routes.

The flexml route executes the complete detector vision-encoder ONNX through
the ONNX Runtime VitisAI EP.  The IRON route keeps patch embedding and the FPN
neck on the GPU and executes the 32 ViT blocks in the frozen NPU server.

Both routes are compared with the same PyTorch FP16 vision encoder and emit a
machine-readable JSON record.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_FLEXML_ONNX = "onnx_files_504/backbone_detector/single_simplified.onnx"
DEFAULT_FLEXML_CACHE_DIR = "npu_artifacts/voe_cache_504"
DEFAULT_FLEXML_CACHE_KEY = "backbone_detector_504_v1"
DEFAULT_IRON_BIN = (
    "/home/amd/project/npu_iron/releases/"
    "sam3-vit-p14-m1536-power-v1/bin/sam3-vit-p14-m1536"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--route", choices=("flexml", "iron"), required=True)
    p.add_argument("--checkpoint", default="model/sam3")
    p.add_argument("--image", default="assets/truck.jpg")
    p.add_argument("--imgsz", type=int, default=504)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--output", default="")
    p.add_argument("--flexml-onnx", default=DEFAULT_FLEXML_ONNX)
    p.add_argument("--flexml-cache-dir", default=DEFAULT_FLEXML_CACHE_DIR)
    p.add_argument("--flexml-cache-key", default=DEFAULT_FLEXML_CACHE_KEY)
    p.add_argument("--vaip-config", default="")
    p.add_argument("--npu-bin", default=DEFAULT_IRON_BIN)
    p.add_argument("--omp-threads", type=int, default=8)
    return p.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command_output(argv: list[str]) -> str:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def current_tdr_timeout_ms() -> int | None:
    try:
        log = subprocess.check_output(
            ["journalctl", "-k", "-b", "--no-pager"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    marker = "TDR enabled, timeout "
    for line in reversed(log.splitlines()):
        if marker in line:
            tail = line.split(marker, 1)[1]
            try:
                return int(tail.split(" ms", 1)[0])
            except (ValueError, IndexError):
                return None
    return None


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def timing_stats(values: list[float]) -> dict[str, float | int]:
    return {
        "runs": len(values),
        "min_ms": min(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 50),
        "p95_ms": percentile(values, 95),
        "max_ms": max(values),
    }


def find_vaip_config(explicit: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    prefix = Path(sys.prefix)
    candidates = list(prefix.glob("lib/python*/site-packages/voe-*/vaip_config.json"))
    if not candidates:
        raise FileNotFoundError("cannot locate voe-*/vaip_config.json; pass --vaip-config")
    return sorted(candidates)[-1]


def create_flexml_session(args: argparse.Namespace):
    # This must happen before loading the GPU model.  The validated in-process
    # coexistence order is XRT/VitisAI first, then HIP.
    import onnxruntime as ort

    onnx_path = resolve_repo_path(args.flexml_onnx)
    cache_dir = resolve_repo_path(args.flexml_cache_dir)
    cache_artifact = cache_dir / args.flexml_cache_key / f"{args.flexml_cache_key}.rai"
    config_file = find_vaip_config(args.vaip_config)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    if not cache_artifact.is_file():
        raise FileNotFoundError(
            f"compiled VitisAI artifact missing: {cache_artifact}; "
            "run compile_flexml_cache.py with a new cache key"
        )

    options = {
        "config_file": str(config_file),
        "cacheDir": str(cache_dir),
        "cacheKey": args.flexml_cache_key,
    }
    t0 = time.perf_counter()
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["VitisAIExecutionProvider"],
        provider_options=[options],
    )
    session_create_ms = (time.perf_counter() - t0) * 1000.0
    if "VitisAIExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"VitisAI EP not active: {session.get_providers()}")
    return session, session_create_ms, onnx_path, cache_artifact, config_file, ort.__version__


def preprocess_input(args: argparse.Namespace) -> np.ndarray:
    # CPU-only preprocessing.  Importing torch/transformers is safe here as
    # long as no CUDA tensor or model is created before the flexml warmup.
    import cv2
    from transformers import AutoProcessor

    image_bgr = cv2.imread(str(resolve_repo_path(args.image)))
    if image_bgr is None:
        raise FileNotFoundError(resolve_repo_path(args.image))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    processor = AutoProcessor.from_pretrained(str(resolve_repo_path(args.checkpoint)))
    new_size = {"height": args.imgsz, "width": args.imgsz}
    new_mask = {"height": 4 * args.imgsz // 14, "width": 4 * args.imgsz // 14}
    for sub in (
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
    ):
        if sub is not None:
            if hasattr(sub, "size"):
                sub.size = new_size
            if hasattr(sub, "mask_size"):
                sub.mask_size = new_mask
    pixel_values = processor(images=[[image_rgb]], return_tensors="pt")["pixel_values"]
    return np.ascontiguousarray(pixel_values.float().cpu().numpy())


def load_reference(args: argparse.Namespace, np_input: np.ndarray):
    from tracker.rocm_env import apply as apply_rocm_env

    apply_rocm_env()
    import torch
    from tracker.live_inference import SAM3Live

    if args.imgsz != 504:
        raise ValueError("the validated NPU artifacts require --imgsz 504")
    live = SAM3Live(
        checkpoint=str(resolve_repo_path(args.checkpoint)),
        prompts=["benchmark"],
        imgsz=args.imgsz,
        mig=False,
        redetect_every=1,
    )
    pixel_values = torch.from_numpy(np_input).to("cuda", live.model.dtype)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        reference = live.model.detector_model.vision_encoder(pixel_values)
    torch.cuda.synchronize()
    reference_ms = (time.perf_counter() - t0) * 1000.0
    return live, pixel_values, reference, reference_ms


def warmup_flexml(session, np_input: np.ndarray, warmup: int) -> list[float]:
    input_name = session.get_inputs()[0].name
    times = []
    for _ in range(max(1, warmup)):
        t0 = time.perf_counter()
        session.run(None, {input_name: np_input})
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def run_flexml(session, np_input: np.ndarray, runs: int):
    input_name = session.get_inputs()[0].name

    times: list[float] = []
    outputs = None
    for i in range(runs):
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: np_input})
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)
        print(f"run {i:02d}: {elapsed:.3f} ms", flush=True)

    assert outputs is not None
    names = [o.name for o in session.get_outputs()]
    result = dict(zip(names, outputs))
    expected = {"fpn_0", "fpn_1", "fpn_2", "fpn_3", "last_hidden_state"}
    if set(result) != expected:
        raise RuntimeError(f"unexpected flexml outputs: {names}")
    return result["last_hidden_state"], [result[f"fpn_{i}"] for i in range(4)], times, {}


def run_iron(args: argparse.Namespace, live, pixel_values):
    import torch
    from tracker.npu_backbone_service import patch_sam3_with_npu_backbone

    npu_bin = Path(args.npu_bin)
    if not npu_bin.is_file():
        raise FileNotFoundError(npu_bin)
    encoder = patch_sam3_with_npu_backbone(
        live.model,
        npu_bin=str(npu_bin),
        omp_threads=args.omp_threads,
    )
    for _ in range(args.warmup):
        with torch.inference_mode():
            encoder(pixel_values)

    times: list[float] = []
    components = {"embed_ms": [], "npu_ms": [], "neck_ms": []}
    output = None
    for i in range(args.runs):
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = encoder(pixel_values)
        elapsed = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed)
        timing = encoder.timing
        components["embed_ms"].append(float(timing["embed_ms"]))
        components["npu_ms"].append(float(timing["npu_ms"]))
        components["neck_ms"].append(
            float(timing["total_ms"] - timing["embed_ms"] - timing["npu_ms"])
        )
        print(
            f"run {i:02d}: total={elapsed:.3f} ms "
            f"embed={components['embed_ms'][-1]:.3f} "
            f"npu={components['npu_ms'][-1]:.3f} "
            f"neck={components['neck_ms'][-1]:.3f}",
            flush=True,
        )

    assert output is not None
    encoder.shutdown()
    component_stats = {name: timing_stats(vals) for name, vals in components.items()}
    return output.last_hidden_state, list(output.fpn_hidden_states), times, component_stats


def cosine(a, b) -> float:
    import torch
    import torch.nn.functional as F

    ta = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.asarray(a))
    tb = b if isinstance(b, torch.Tensor) else torch.from_numpy(np.asarray(b))
    ta = ta.detach().float().cpu().flatten()
    tb = tb.detach().float().cpu().flatten()
    if ta.shape != tb.shape:
        raise ValueError(f"shape mismatch: {tuple(ta.shape)} vs {tuple(tb.shape)}")
    return float(F.cosine_similarity(ta.unsqueeze(0), tb.unsqueeze(0)).item())


def accuracy_record(last_hidden, fpn, reference) -> dict:
    import torch
    import torch.nn.functional as F

    lhs = last_hidden if isinstance(last_hidden, torch.Tensor) else torch.from_numpy(last_hidden)
    ref = reference.last_hidden_state
    lhs_tokens = lhs.detach().float().cpu().squeeze(0)
    ref_tokens = ref.detach().float().cpu().squeeze(0)
    token_cos = F.cosine_similarity(lhs_tokens, ref_tokens, dim=-1)
    fpn_cos = [cosine(n, p) for n, p in zip(fpn, reference.fpn_hidden_states)]
    return {
        "last_hidden_cos": cosine(last_hidden, reference.last_hidden_state),
        "per_token_cos_mean": float(token_cos.mean().item()),
        "per_token_cos_min": float(token_cos.min().item()),
        "fpn_cos": {f"p{i + 2}": value for i, value in enumerate(fpn_cos)},
        "fpn_cos_min": min(fpn_cos),
    }


def default_output(route: str) -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / "npu_vit_benchmark" / f"{route}_{stamp}.json"


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.runs < 1:
        raise ValueError("--warmup must be >= 0 and --runs must be >= 1")

    flexml = None
    flexml_warmup_times: list[float] = []
    if args.route == "flexml":
        flexml = create_flexml_session(args)

    np_input = preprocess_input(args)
    if args.route == "flexml":
        # Session construction may defer device initialization until the first
        # run.  Complete at least one NPU run before any CUDA model/forward.
        flexml_warmup_times = warmup_flexml(flexml[0], np_input, args.warmup)

    live, pixel_values, reference, reference_ms = load_reference(args, np_input)

    if args.route == "flexml":
        session, session_create_ms, onnx_path, cache_artifact, config_file, ort_version = flexml
        last_hidden, fpn, times, components = run_flexml(session, np_input, args.runs)
        artifact = {
            "onnx_path": str(onnx_path),
            "onnx_sha256": sha256(onnx_path),
            "cache_artifact": str(cache_artifact),
            "cache_sha256": sha256(cache_artifact),
            "vaip_config": str(config_file),
        }
        route_metadata = {
            "variant": "vitisai_ep_backbone_detector_504_v1",
            "session_create_ms": session_create_ms,
            "pre_gpu_warmup_ms": flexml_warmup_times,
            "onnxruntime_version": ort_version,
        }
    else:
        last_hidden, fpn, times, components = run_iron(args, live, pixel_values)
        npu_bin = Path(args.npu_bin)
        artifact = {
            "npu_binary": str(npu_bin),
            "npu_binary_sha256": sha256(npu_bin),
        }
        route_metadata = {
            "variant": "p14_m1536_common_atb_affine",
            "omp_threads": args.omp_threads,
        }

    accuracy = accuracy_record(last_hidden, fpn, reference)
    output_path = Path(args.output) if args.output else default_output(args.route)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    import transformers

    record = {
        "schema_version": 1,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workload": {
            "model": "SAM3 ViT detector backbone",
            "image_size": args.imgsz,
            "input_shape": list(pixel_values.shape),
            "layers": 32,
            "tokens": 1296,
            "hidden_size": 1024,
            "image": str(resolve_repo_path(args.image)),
            "image_sha256": sha256(resolve_repo_path(args.image)),
        },
        "route": args.route,
        "route_metadata": route_metadata,
        "protocol": {"warmup": args.warmup, "runs": args.runs},
        "latency": timing_stats(times),
        "components": components,
        "accuracy": accuracy,
        "reference": {"framework": "PyTorch FP16", "latency_ms": reference_ms},
        "artifacts": artifact,
        "environment": {
            "kernel": platform.release(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "xrt_smi_version": command_output(["xrt-smi", "--version"]),
            "amdxdna_module": command_output(["modinfo", "-n", "amdxdna"]),
            "tdr_timeout_ms": current_tdr_timeout_ms(),
            "tracker_git_head": command_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]
            ),
        },
    }
    output_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print("\nSAM3 ViT NPU benchmark")
    print(f"route={args.route} variant={route_metadata['variant']}")
    print(
        f"latency p50={record['latency']['p50_ms']:.3f} ms "
        f"p95={record['latency']['p95_ms']:.3f} ms"
    )
    print(
        f"accuracy hidden={accuracy['last_hidden_cos']:.6f} "
        f"fpn_min={accuracy['fpn_cos_min']:.6f}"
    )
    print(f"result={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
