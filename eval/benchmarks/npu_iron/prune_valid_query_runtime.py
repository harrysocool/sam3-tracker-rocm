#!/usr/bin/env python3
"""Prune unused SAM3 window-query launch blocks from placed flash MLIR."""

import argparse
import re
from pathlib import Path


BLOCK_RE = re.compile(
    r"\s*%(\d+) = aiex\.dma_configure_task_for @air_QKIn_0_0_0 \{"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.input.read_text().splitlines()
    runtime_start = next(
        i
        for i, line in enumerate(lines)
        if re.match(r"\s*aie\.runtime_sequence @attention_bf16\(", line)
    )

    depth = 0
    runtime_end = None
    for i in range(runtime_start, len(lines)):
        depth += lines[i].count("{") - lines[i].count("}")
        if i > runtime_start and depth == 0:
            runtime_end = i
            break
    if runtime_end is None:
        raise RuntimeError("cannot find runtime-sequence end")

    starts = []
    for i in range(runtime_start + 1, runtime_end):
        match = BLOCK_RE.match(lines[i])
        if match:
            task_id = int(match.group(1))
            if task_id % 24 == 0:
                starts.append((i, task_id))
    if len(starts) != 96:
        raise RuntimeError(f"expected 96 launch blocks, found {len(starts)}")

    for block, (_, task_id) in enumerate(starts):
        if task_id != block * 24:
            raise RuntimeError(
                f"unexpected first task id for block {block}: {task_id}"
            )

    kept = []
    removed = []
    body = []
    for block, (start, _) in enumerate(starts):
        end = starts[block + 1][0] if block + 1 < len(starts) else runtime_end
        q_iter = block // 32
        head_group = block % 32
        max_iters = 3 if head_group < 8 else 2 if head_group < 24 else 1
        if q_iter < max_iters:
            body.extend(lines[start:end])
            kept.append((q_iter, head_group))
        else:
            removed.append((q_iter, head_group))

    if len(kept) != 64 or len(removed) != 32:
        raise RuntimeError(f"unexpected keep/remove counts: {len(kept)}/{len(removed)}")

    output_lines = lines[: starts[0][0]] + body + lines[runtime_end:]
    output = "\n".join(output_lines) + "\n"

    expected_counts = {
        "configure": 64 * 24,
        "start": 64 * 24,
        "await": 64 * 6,
        "free": 64 * 18,
    }
    actual_counts = {
        "configure": output.count("aiex.dma_configure_task_for"),
        "start": output.count("aiex.dma_start_task"),
        "await": output.count("aiex.dma_await_task"),
        "free": output.count("aiex.dma_free_task"),
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"unexpected operation counts: {actual_counts}")

    args.output.write_text(output)
    print(
        f"pruned {args.input} -> {args.output}: "
        f"kept={len(kept)} removed={len(removed)} ops={actual_counts}"
    )


if __name__ == "__main__":
    main()
