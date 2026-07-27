#!/usr/bin/env python3
"""Statically validate a six-shape dynamic-K shared GEMM build."""

import argparse
import hashlib
import re
from pathlib import Path


SHAPES = {
    "qkv_w": 4,
    "qkv_g": 4,
    "o_w": 4,
    "o_g": 4,
    "ffn1": 4,
    "ffn2": 20,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    shared_hashes = {"elf_cdo": set(), "init_cdo": set(), "enable_cdo": set()}
    for shape, k_tiles in SHAPES.items():
        directory = args.root / shape
        mlir = (directory / "patched.mlir").read_text()
        required = [directory / "final.xclbin", directory / "insts.bin"]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise AssertionError(f"missing runtime artifact for {shape}")

        writes = re.findall(r"aiex\.npu\.rtp_write\(@rtp_k_\d_\d, 0, (\d+)\)", mlir)
        acquires = re.findall(r"aie\.use_lock\(%rtp_lock_\d_\d, Acquire, 1\)", mlir)
        releases = re.findall(r"aie\.use_lock\(%rtp_lock_\d_\d, Release, 0\)", mlir)
        compares = re.findall(r"arith\.cmpi slt, %9, %rtp_k_idx_\d_\d : index", mlir)
        release_pairs = re.findall(
            r"aie\.use_lock\(%rtp_lock_(\d_\d), Release, 0\)\n"
            r"\s*aie\.use_lock\(%lock_\1(?:_\d+)?, Release, 1\)",
            mlir,
        )
        if len(writes) != 32 or set(writes) != {str(k_tiles)}:
            raise AssertionError(f"bad RTP writes for {shape}: {len(writes)}, {set(writes)}")
        counts = (len(acquires), len(releases), len(compares), len(release_pairs))
        if counts != (32, 32, 32, 32):
            raise AssertionError(f"bad barrier/core counts for {shape}: {counts}")

        project = directory / "patched.mlir.prj"
        shared_hashes["elf_cdo"].add(sha256(project / "matmul_seg_aie_cdo_elfs.bin"))
        shared_hashes["init_cdo"].add(sha256(project / "matmul_seg_aie_cdo_init.bin"))
        shared_hashes["enable_cdo"].add(sha256(project / "matmul_seg_aie_cdo_enable.bin"))
        print(
            f"{shape}: k_tiles={k_tiles} writes=32 barriers=32 "
            f"xclbin={sha256(directory / 'final.xclbin')} "
            f"insts={sha256(directory / 'insts.bin')}"
        )

    for name, hashes in shared_hashes.items():
        if len(hashes) != 1:
            raise AssertionError(f"{name} differs across shapes: {sorted(hashes)}")
        print(f"shared_{name}={next(iter(hashes))}")
    print("DYNAMIC_K_STATIC_CHECK=PASS")


if __name__ == "__main__":
    main()
