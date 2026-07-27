#!/usr/bin/env python3
"""Exercise NPU runtime autosuspend/resume with one persistent backbone server."""

import argparse
import glob
import os
import select
import struct
import subprocess
import sys
import time


MAGIC = 0x0000BF16
TOKEN_BYTES = 1296 * 1024 * 4
OUTPUT_BYTES = TOKEN_BYTES


def find_runtime_status():
    matches = glob.glob("/sys/bus/pci/drivers/amdxdna/*:*/power/runtime_status")
    if len(matches) != 1:
        raise RuntimeError(f"expected one amdxdna runtime_status, found {matches}")
    return matches[0]


def read_status(path):
    with open(path, encoding="ascii") as stream:
        return stream.read().strip()


def read_exact(proc, size, timeout_s):
    result = bytearray()
    deadline = time.monotonic() + timeout_s
    fd = proc.stdout.fileno()
    while len(result) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"read timeout at {len(result)}/{size}")
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            raise TimeoutError(f"read timeout at {len(result)}/{size}")
        chunk = os.read(fd, min(size - len(result), 1024 * 1024))
        if not chunk:
            raise EOFError(f"server EOF at {len(result)}/{size}")
        result.extend(chunk)
    return bytes(result)


def stop_process(proc):
    if proc.poll() is not None:
        return
    try:
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    try:
        proc.wait(timeout=3)
        return
    except subprocess.TimeoutExpired:
        proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        default="/home/amd/project/npu_iron/bh_validq_hostopt_20260727",
    )
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--pause", type=float, default=6.5)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.cycles < 2 or args.pause < 5.5:
        raise SystemExit("cycles must be >=2 and pause >=5.5 seconds")

    status_path = find_runtime_status()
    env = os.environ.copy()
    env["PATH"] = "/opt/xilinx/xrt/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/opt/xilinx/xrt/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["OMP_NUM_THREADS"] = "8"
    proc = subprocess.Popen(
        [args.binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env
    )
    payload = struct.pack("<I", MAGIC) + bytes(TOKEN_BYTES)
    try:
        ready = struct.unpack("<I", read_exact(proc, 4, args.timeout))[0]
        if ready != MAGIC:
            raise RuntimeError(f"unexpected ready magic 0x{ready:08x}")
        print(f"server_ready=1 initial_runtime_status={read_status(status_path)}")

        suspended_count = 0
        for cycle in range(args.cycles):
            before = read_status(status_path)
            start = time.monotonic()
            proc.stdin.write(payload)
            proc.stdin.flush()
            echoed = struct.unpack("<I", read_exact(proc, 4, args.timeout))[0]
            if echoed != MAGIC:
                raise RuntimeError(f"unexpected output magic 0x{echoed:08x}")
            read_exact(proc, OUTPUT_BYTES, args.timeout)
            elapsed_ms = (time.monotonic() - start) * 1000
            after = read_status(status_path)
            print(
                f"cycle={cycle} frame_ms={elapsed_ms:.1f} "
                f"status_before={before} status_after={after}",
                flush=True,
            )
            if cycle + 1 == args.cycles:
                continue
            time.sleep(args.pause)
            paused = read_status(status_path)
            print(f"cycle={cycle} status_after_pause={paused}", flush=True)
            if paused == "suspended":
                suspended_count += 1

        expected = args.cycles - 1
        if suspended_count != expected:
            print(
                f"AUTOSUSPEND_PROBE=FAIL suspended={suspended_count}/{expected}",
                file=sys.stderr,
            )
            return 1
        print(f"AUTOSUSPEND_PROBE=PASS suspended={suspended_count}/{expected}")
        return 0
    finally:
        stop_process(proc)


if __name__ == "__main__":
    raise SystemExit(main())
