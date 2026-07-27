#!/usr/bin/env python3
"""Patch a placed MLIR-AIE GEMM design with per-core dynamic K-loop RTPs."""

import argparse
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--k-tiles", type=int, required=True)
    args = parser.parse_args()

    text = args.input.read_text()
    lines = text.splitlines()
    compute_tiles = [(col, row) for row in range(2, 6) for col in range(8)]

    tile_anchor = None
    for index, line in enumerate(lines):
        if re.match(r"\s*%tile_7_5 = aie\.tile\(7, 5\)", line):
            tile_anchor = index + 1
            break
    if tile_anchor is None:
        raise RuntimeError("cannot find final compute-tile declaration")

    buffer_lines = []
    for col, row in compute_tiles:
        buffer_lines.extend(
            [
                f'    %rtp_k_{col}_{row} = aie.buffer(%tile_{col}_{row}) '
                f'{{sym_name = "rtp_k_{col}_{row}"}} : memref<1xi32>',
                f"    %rtp_lock_{col}_{row} = aie.lock(%tile_{col}_{row}, 6) "
                "{init = 0 : i32}",
            ]
        )
    lines[tile_anchor:tile_anchor] = buffer_lines

    core_starts = []
    for index, line in enumerate(lines):
        match = re.match(
            r"\s*%core_(\d)_(\d) = aie\.core\(%tile_\1_\2\) \{", line
        )
        if match:
            core_starts.append((index, int(match.group(1)), int(match.group(2))))
    if len(core_starts) != 32:
        raise RuntimeError(f"expected 32 cores, found {len(core_starts)}")

    modified = 0
    # Work backwards so insertions do not invalidate later start offsets.
    for start, col, row in reversed(core_starts):
        end = next(
            (
                i
                for i in range(start + 1, len(lines))
                if re.match(r"\s*\} \{.*stack_size = \d+ : i32\}", lines[i])
            ),
            None,
        )
        if end is None:
            raise RuntimeError(f"cannot find end of core {col},{row}")

        c0_line = next(
            (
                i
                for i in range(start + 1, end)
                if re.match(r"\s*%c0 = arith\.constant 0 : index", lines[i])
            ),
            None,
        )
        if c0_line is None:
            raise RuntimeError(f"cannot find constants in core {col},{row}")
        acquire_line = next(
            (
                i
                for i in range(c0_line + 1, end)
                if re.match(
                    r"\s*aie\.use_lock\(.*AcquireGreaterEqual, 1\)", lines[i]
                )
            ),
            None,
        )
        if acquire_line is None:
            raise RuntimeError(f"cannot find entry acquire in core {col},{row}")
        indent = re.match(r"(\s*)", lines[acquire_line]).group(1)
        load_lines = [
            f"{indent}aie.use_lock(%rtp_lock_{col}_{row}, Acquire, 1)",
            f"{indent}%rtp_k_i32_{col}_{row} = memref.load "
            f"%rtp_k_{col}_{row}[%c0] : memref<1xi32>",
            f"{indent}%rtp_k_idx_{col}_{row} = arith.index_cast "
            f"%rtp_k_i32_{col}_{row} : i32 to index",
        ]
        lines[acquire_line + 1 : acquire_line + 1] = load_lines
        end += len(load_lines)

        compare = next(
            (
                i
                for i in range(acquire_line + 1 + len(load_lines), end)
                if re.match(r"\s*%10 = arith\.cmpi slt, %9, %c\d+ : index", lines[i])
            ),
            None,
        )
        if compare is None:
            raise RuntimeError(f"cannot find K-loop compare in core {col},{row}")
        static_k_match = re.search(r"%9, %c(\d+)", lines[compare])
        if static_k_match is None or int(static_k_match.group(1)) != args.k_tiles:
            raise RuntimeError(
                f"unexpected static K-loop bound in core {col},{row}: "
                f"{lines[compare].strip()} (expected {args.k_tiles})"
            )
        lines[compare] = re.sub(
            r"%9, %c\d+", f"%9, %rtp_k_idx_{col}_{row}", lines[compare]
        )

        # Keep the dispatch barrier held for the complete core iteration.  The
        # final output release makes the output-DMA completion a reliable
        # boundary: a following runtime sequence cannot release this core with
        # a new RTP value until the previous iteration has returned the lock
        # to zero.
        final_output_release = next(
            (
                i
                for i in range(end - 1, compare, -1)
                if re.match(
                    rf"\s*aie\.use_lock\(%lock_{col}_{row}(?:_\d+)?, Release, 1\)",
                    lines[i],
                )
            ),
            None,
        )
        if final_output_release is None:
            raise RuntimeError(
                f"cannot find final output release in core {col},{row}"
            )
        final_indent = re.match(r"(\s*)", lines[final_output_release]).group(1)
        lines[final_output_release:final_output_release] = [
            f"{final_indent}aie.use_lock(%rtp_lock_{col}_{row}, Release, 0)"
        ]
        modified += 1

    runtime_start = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(r"\s*aie\.runtime_sequence @matmul_bf16\(", line)
        ),
        None,
    )
    if runtime_start is None:
        raise RuntimeError("cannot find matmul runtime sequence")
    runtime_indent = re.match(r"(\s*)", lines[runtime_start]).group(1) + "  "
    writes = [
        f"{runtime_indent}aiex.npu.rtp_write(@rtp_k_{col}_{row}, 0, {args.k_tiles})"
        for col, row in compute_tiles
    ]
    releases = [
        f"{runtime_indent}aiex.set_lock(%rtp_lock_{col}_{row}, 1)"
        for col, row in compute_tiles
    ]
    lines[runtime_start + 1 : runtime_start + 1] = writes + releases

    if modified != 32:
        raise RuntimeError(f"expected 32 modified cores, got {modified}")
    output = "\n".join(lines) + "\n"
    args.output.write_text(output)
    print(
        f"patched {args.input} -> {args.output}: "
        f"cores={modified}, rtp_writes={len(writes)}, "
        f"barriers={len(releases)}, k_tiles={args.k_tiles}"
    )


if __name__ == "__main__":
    main()
