# AMD XDNA runtime-PM periodic stall root cause

**Date:** 2026-07-27

## Observation

The optimized backbone showed one arbitrary 220-270 ms dispatch stall about
every five seconds. A same-design microbenchmark did not reproduce the stall,
and the affected operator/layer changed each time.

A low-overhead profiler recorded global dispatch ordinal, frame, layer, stage,
current/previous design, transition state, and runtime-PM state only when a
dispatch exceeded 20 ms.

With `power/control=auto`:

```text
slow frames: 0, 6, 12, 18, 24, 30
ordinal gaps: 911, 997, 991, 994, 1001
PM state at every stall: active
affected stages: FFN1, FFN2, flash_w, O, flash_g
one stall occurred with transition=0 inside the common GEMM design
```

The faster valid-query backbone changed the submission interval from the old
~682 to ~1000, but elapsed time remained about 4.6-5.0 seconds. This matches the
driver's five-second runtime autosuspend delay and rules out an operator,
instruction stream, fixed submission count, or required design transition.

## Controlled comparison

The same 30-frame workload was run with runtime PM temporarily forced to `on`.
The script restored `auto` on every exit path.

| Mode | slow count | C++ p50 | C++ p95 | Full p50 | Full p95 |
|---|---:|---:|---:|---:|---:|
| auto | 6 | 793.5 ms | 1018.3 ms | 808.0 ms | 1032.1 ms |
| on | 0 | 791.5 ms | 807.6 ms | 807.3 ms | 823.4 ms |

Forcing `on` did not improve normal p50. It removed only the periodic lock
contention, proving that this is a latency-stability mode rather than a compute
acceleration mode.

## Host traversal fusion

The next candidate combined residual snapshot with CPU LayerNorm traversal and
wrote attention bias+residual directly into the backbone state. Two standalone
residual copies and the 5 MB attention temporary were removed.

Compact FFN1 H=4864 was also tested, but FFN1 dispatch became 6-9 ms slower due
to less favorable launch/tiling efficiency. It was rejected; host fusion was
retained with the original H=5120 FFN1 path.

Accuracy after host fusion remains above the project gate:

```text
last_hidden_state  0.993296
FPN p2             0.999416
FPN p3             0.998490
FPN p4             0.997723
FPN p5             0.997220
```

Final hostfused + performance-mode 30-frame result:

| Metric | min | mean | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| C++ backbone wall ms | 758 | 762.5 | **761.5** | 768.1 | **769.5** | 776 |
| dispatch ms | 650 | 653.2 | **652.0** | 657.2 | **659.5** | 664 |
| Python-visible NPU ms | 764.1 | 769.4 | **768.5** | 774.9 | **776.7** | 782.9 |
| full image-to-FPN ms | 770.8 | 777.4 | **776.1** | 783.7 | **784.2** | 789.6 |

No slow dispatch, D-state, or watchdog/monitor issue was observed.

## Operational policy

Keep the default `power/control=auto` while idle. Set `power/control=on` for the
lifetime of a latency-sensitive NPU service, then restore `auto` when the
service exits. Do not globally disable runtime PM without measuring idle-power
impact.

## Decision

- Runtime autosuspend is the confirmed p95 stall source.
- Performance mode solves p95 without changing compute or accuracy.
- Compact FFN1 is rejected.
- Host traversal fusion becomes the next performance baseline.
- Remaining path toward 500 ms is BF16/BFP16 microkernel improvement and device
  chaining, not further host copy tuning.
