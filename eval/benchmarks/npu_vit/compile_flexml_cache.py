#!/usr/bin/env python3
"""Create a new VitisAI EP cache for the canonical SAM3 504 px backbone.

The script never deletes or overwrites an existing cache key.  Constructing
the session is the VitisAI EP compile step; the resulting .rai is then hashed.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--cache-key", required=True)
    p.add_argument("--vaip-config", default="")
    return p.parse_args()


def find_config(value: str) -> Path:
    if value:
        return Path(value)
    matches = sorted(Path(sys.prefix).glob("lib/python*/site-packages/voe-*/vaip_config.json"))
    if not matches:
        raise FileNotFoundError("cannot locate vaip_config.json")
    return matches[-1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    import onnxruntime as ort

    onnx = Path(args.onnx).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_dir = cache_dir / args.cache_key
    output = output_dir / f"{args.cache_key}.rai"
    config = find_config(args.vaip_config).resolve()
    if not onnx.is_file():
        raise FileNotFoundError(onnx)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse existing cache key: {output_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "config_file": str(config),
        "cacheDir": str(cache_dir),
        "cacheKey": args.cache_key,
    }
    t0 = time.perf_counter()
    session = ort.InferenceSession(
        str(onnx),
        providers=["VitisAIExecutionProvider"],
        provider_options=[options],
    )
    elapsed = time.perf_counter() - t0
    if "VitisAIExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"VitisAI EP not active: {session.get_providers()}")
    if not output.is_file():
        raise FileNotFoundError(f"session completed but cache artifact is missing: {output}")
    print(f"compile_seconds={elapsed:.3f}")
    print(f"artifact={output}")
    print(f"sha256={sha256(output)}")
    print("FLEXML_COMPILE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
