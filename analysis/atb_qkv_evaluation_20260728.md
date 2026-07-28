# Chess ATB QKV evaluation — 2026-07-28

## Candidate

- Upstream MLIR-AIE config1 asymmetric tile buffering.
- QKV shape padded from M=1536 to M=2048; K=1024, N=3072.
- BF16 activation, BFP16EBS8 weight, BF16 output.
- All 32 ViT layers use ATB QKV; attention/O/FFN remain on the validated path.

Toolchain:

```text
mlir-aie c69e4c3 / wheel 1.3.5.dev115+gc69e4c3
Vitis AIE Essentials 1.9.0.dev20260611, Chess X-2025.06
AIE2P guarded acquire/release compatibility shim
```

Artifacts:

```text
xclbin a587a26cea2e8244010f3be04083aaf715ee67f1af9818a70acd6d74f86ce014
insts  ad5b509d20033307dbe03c3bb18a0f26bcd6de2623dfd9ab302a16db40f3c780
candidate binary 9d50d9fcf2eeffab9fc0dc34c893323f779327e4c3c0b50c5817a2261361a28c
```

## Primitive and transition gates

```text
random numerical gate: PASS
same-context warm: avg 810 us, min/max 796/838 us
FFN2 -> ATB transition: mean/median 3577/3567 us
D-state: none
```

## Full-backbone result

Representative timed frame:

```text
C++ wall/dispatch: 743 / 624 ms
QKV dispatch/read: 112.3 / 14.2 ms
image-to-FPN:      761 ms
```

Validated hostfused baseline:

```text
C++ wall/dispatch p50: 761.5 / 652.0 ms
QKV dispatch:          about 140.5 ms
image-to-FPN p50:      776.1 ms
```

Accuracy against PyTorch:

```text
last-hidden 0.989694
FPN p2      0.999181
FPN p3      0.997588
FPN p4      0.996031
FPN p5      0.995805
```

The current gate is last-hidden 0.993296 with FPN no lower than 0.997220.
The candidate therefore fails accuracy despite a real 15-19 ms end-to-end
gain. The extra BFP16 weight quantization and BF16 projection output accumulate
across all 32 layers.

## Decision

Reject full-layer ATB QKV integration. Keep the binary and source as a bounded
negative result. Partial-layer ATB would save only a few milliseconds before
meeting the existing accuracy floor and is not a priority for the 500 ms goal.
