# Performance records and correctness evidence

This page separates live freshness measurements, offline throughput, and
historical box-tracker results. Published latency and throughput values retain
their original measurement windows; newer correctness checks are labeled
separately. Runtime code and local model artifacts may evolve, so record their
identities when reproducing a result. These numbers are reference observations,
not deployment deadlines.

## Measurement terms

- **Output Hz:** completed inference cadence in the harness's measured window.
  It is not camera FPS or the fraction of source frames processed.
- **Service time:** inference work after the consumer starts an owned frame.
- **Frame age:** host-monotonic arrival to completed inference, including time
  waiting in the latest slot. It excludes exposure-to-host transport and any
  downstream ROS/DDS, TF, rendering, or occupancy-grid publication delay.
- **Emitted / captured / dropped:** separate counts; newest-frame replacement
  intentionally discards stale waiting frames.
- **Offline FPS:** sequential video throughput under that experiment's window
  and rendering settings. It is not comparable to a live cadence number without
  matching those conditions.

Do not derive service time by inverting camera FPS, or interpret an age p95 as
a hard maximum. Use pose/TF at sensor exposure time and enforce an application
result-age budget separately.

## Default full-detection live reference

Recorded September 1, 2026 on Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`):

- ROCm 7.14, MIGraphX 2.17, ORT 1.24.2 with `MIGraphXExecutionProvider`.
- Torch 2.11.0 gfx1151 wheel; 504px FP16 input.
- Accepted FC1-sink backbone, fixed direct-MXR decoder, same-frame parallel
  detector/tracker tails.
- `LatestFramePipeline(copy_frames=True)`, one ordered inference owner,
  full detection on every consumed frame, no N+1 GPU work.
- `assets/blackswan.mp4`, prompt `swan`, one object; 250 source arrivals paced
  at 24 FPS by a looped-source harness. No overlay or video encoding.
- Headline interval, service, and age statistics use the recorded warm window
  with `warm_discard_outputs=5`; counts below include all outputs.

| Metric | Recorded value |
|---|---:|
| Output rate | **9.1275 Hz** |
| Emitted / captured / dropped | 96 / 250 / 154 |
| Mean completion interval | 109.56 ms |
| Mean service time | 109.53 ms |
| Frame age p50 / p95 | 131.82 / **152.73 ms** |
| Queue wait p95 | 39.95 ms |
| Last consumed source sequence | 249 |
| Inference failures / aborts / drain failures | 0 / 0 / 0 |

The finite-file README demo renders and encodes output, has no explicit
prewarm by default, and does not loop the clip. It is not the original
250-arrival headless benchmark. The original image label was
`sam3-gpu714-ort1242-mgx217-gfx1151:torch211`; current release assembly is
documented in the [container guide](../docker/rocm714/README.md). This is not
a fresh measurement of each locally rebuilt rc4 model artifact.

## Fixed-decoder controlled A/B

The September 1 experiment used complementary ABBA and BAAB orders, each with
250 arrivals per arm under the same latest-frame workload. The control used
the native decoder; the candidate used the fixed direct-MXR decoder. Both
shared the accepted backbone and parallel-tail configuration.

| Combined eight-arm metric | Control | Candidate | Change |
|---|---:|---:|---:|
| Mean completion interval | 122.060 ms | 114.372 ms | 6.30% lower |
| Output rate | 8.193 Hz | 8.743 Hz | 6.72% higher |
| Mean service | 121.923 ms | 114.219 ms | 6.32% lower |
| Service p95 | 140.131 ms | 129.914 ms | 7.29% lower |
| Frame age p95 | 165.066 ms | 160.163 ms | 2.97% lower |

The interval and rate percentages differ because rate is reciprocal to the
interval. Every position-matched comparison favored the candidate. The
separate 9.1275 Hz integration run above is **not** another arm of this A/B.
Neither establishes 10 Hz full-detection live operation.

## Optional hybrid live reference

Recorded September 2, 2026 with the same GPU/runtime family and a 1000 ms
full-detection interval. Clean keyframes replace the inner session, reuse
prompt embeddings, and associate public IDs by same-prompt mask IoU.
Native detector-skip tracking runs between keyframes; the latest-frame
scheduler and no-N+1 policy do not change.

### One object

`blackswan.mp4`, `swan`, 504px FP16, 250 arrivals at 24 FPS, no rendering or
video encoding. Each run used explicit prewarm and tracking reset. All 110
outputs per run are included in the reported aggregate statistics; unlike
the full-mode reference, the first five measured outputs were not discarded.

| Run | Output Hz | Mean service | Frame age p50 / p95 | Emitted / captured / dropped |
|---|---:|---:|---:|---:|
| 1 | 10.4719 | 95.478 ms | 110.216 / 136.995 ms | 110 / 250 / 140 |
| 2 | 10.4724 | 95.446 ms | 110.911 / 135.634 ms | 110 / 250 / 140 |
| 3 | 10.4749 | 95.432 ms | 111.025 / 137.764 ms | 110 / 250 / 140 |

Each run included a first-propagation outlier at source sequence 2, with
maximum service times of 496.716 / 488.601 / 498.082 ms. These outliers were
**retained**, not removed from the statistics. Each run reached sequence 249,
with ten keyframes and no inference failure, abort, drain failure, or
propagation object-loss event.

These are separate runs with different detection work and measurement windows
from default full detection. Do not report the headline ratio as a controlled
full-model speedup.

### Multiple objects

`two_person_dog_lawn.mp4`, 300 arrivals at 25 FPS, two explicit warmup frames
followed by tracking reset, no overlay or encoding, fresh container process
per cell. This clip is a local experiment input, not a tracked bundled asset.

| Prompts | Objects/output | Output Hz | Mean service | Age p50 / p95 | Emitted / captured / dropped |
|---|---:|---:|---:|---:|---:|
| `people` | 2 | 9.4621 | 105.60 ms | 126.91 / 150.74 ms | 115 / 300 / 185 |
| `people,dog` | 3 | 8.3682 | 119.44 ms | 139.49 / 172.52 ms | 102 / 300 / 198 |
| `people,dog,lawn,sidewalk` | 6–7 | 6.2287 | 160.47 ms | 178.26 / 237.65 ms | 76 / 300 / 224 |

Active object count, not prompt count alone, drives tracker cost. P4 normally
retained six objects and had seven on 15 of 76 outputs. All cells reached
source sequence 299 and had zero inference failures, aborts, drain failures,
and propagation-frame object-loss events. P4's age p95 exceeded 200 ms;
application age gating remains necessary.

## Correctness and resource checks

### Fixed decoder versus native decoder

With the accepted MIG backbone, a 30-sequential-frame full-path comparison
between native and fixed decoder retained identical metadata, IDs, and prompt
ownership. Mean/minimum mask IoU was approximately **0.999994 / 0.999878**;
maximum score absolute error was 0.0009765625 and box error was zero.

A dropped-frame test selected 22 of 50 source frames. Candidate latest-frame
outputs were bit-exact against candidate direct replay of the selected
subsequence. Candidate versus native decoder replay on that same subsequence
had mean/minimum IoU **0.9999933 / 0.9999242**, matching metadata and zero box
error. This isolates decoder/scheduling equivalence; it is **not** a pure-PT
versus full-MIG regression.

### PT versus MIG offline masks

The August 25–27 offline regression reported mean IoU about **0.994** at 504px.
Two serial runs recorded 0.994174 and 0.994109 mean IoU, with minimum 0.989274
and no frame below 0.95. Pipelined regression runs retained mean 0.994142
and minimum 0.989274. These measurements apply to that offline configuration
and must not be relabeled as fixed-decoder or hybrid accuracy.

The [original full-stack evaluation](rocm714_fullstack_evaluation.md) preserves
the test sequence and synchronization findings. Current commands and the
coverage limits of each test are in the [evaluation guide](evaluation.md).

### Live/offline output unification

The September 14 review validation used the same loaded 504px FP16 MIG model
for offline prefetch and direct `SAM3Live.infer` replay of identical frames.
Both used the fixed decoder, same-frame parallel tails, and a 0.5 output-score
threshold in the supported ROCm 7.14 / MIGraphX 2.17 / ORT 1.24.2 runtime.

| Input and policy | Compared outputs | Result |
|---|---:|---|
| Local `two_person_dog_lawn.mp4`, `people dog`, cap 1 per prompt | 30 frames / 60 masks | Masks, boxes, scores, IDs, and prompt ownership identical |
| `blackswan.mp4`, `swan`, uncapped, `--max-frames 0` | 50 frames / 50 masks | All outputs identical; offline encoded all 50 frames |

The multi-prompt run evicted 30 excess objects in each path and checked the
per-prompt session limit after every frame. This is a persistent tracking cap,
not just a rendering filter. Offline input storage is now separate from the
tracking session, avoiding the upstream preloaded-versus-streaming differences
in tracker temporal encoding and hotstart policy.

A separate 30-frame swan run through the actual offline CLI compared pure
PyTorch with the optimized MIG path. Original-resolution binary mask IoU was
**0.994427 mean / 0.992279 minimum**, with matching IDs and ownership; maximum
score difference was **0.041504**. PT and MIG outputs are not bit-exact.

These are correctness checks, not latency measurements or a general guarantee
of identical results across separately loaded models or different frame
sequences. The initial output-only refactor still differed between preloaded
and live sessions; the accepted run includes frame-by-frame session alignment.
Scripts, source/input/artifact hashes, logs, and `acceptance.json` are retained
in maintainer storage under
`gpu/experiments/unified-output-causal-20260914.PIDaTgnj/`. The multi-person
fixture is local and is not distributed with the repository.

### Hybrid versus full detection on a real scene

A separate 50-frame `office_hallway_two_way` test used a local ROS-bag-derived
input with `floor,wall` prompts. It compared full detection on every sequential
frame against scheduled clean keyframes at `[0,10,20,30,40]`; loss on frame 47
forced an additional detection on frame 48. This was not a source-paced run.

| Prompt | Mean / minimum union IoU on nominal propagation frames | False-free mask pixels | Added mask pixels |
|---|---:|---:|---:|
| floor | 0.981233 / 0.949965 | 1.4169% | 0.4546% |
| wall | 0.963659 / 0.933420 | 1.7963% | 2.2107% |

These are workload-specific mask differences, not a guarantee of occupancy
map quality. Clean-session keyframes are not bit-exact with a stateful
every-frame reference. **Tracker-only absence never authorizes free-space
clearing**, regardless of these IoUs. The input bag is not included in the
repository.

### Fresh-session resource soak

On September 2, 120 measured full-detection calls repeatedly used the first
`blackswan` frame while replacing the inner session, with no explicit garbage
collection. One unmeasured detection populated the prompt embedding first.

| Metric | Change over the measured run |
|---|---:|
| Torch allocated memory | +507,904 bytes |
| Torch reserved memory | +4 MiB |
| Process RSS | +72 KiB |

All three plateaued after iteration 10. Old sessions were released; prompt
tensor identity/content, empty replacement state, and public-ID checks passed.
This short repeated-frame soak is not proof of indefinite memory stability or
a throughput/freshness benchmark. Long streams still need a bounded reset
policy.

## Offline reference runs

These August 25–27 measurements predate the integrated fixed-decoder live
path. They use the ROCm 7.14 / MIGraphX 2.17 container family, one swan, 504px,
with the offline tool's native decoder and rendering/encoding in its reported
end-to-end propagation window.

| Offline configuration | Recorded FPS |
|---|---:|
| Serial MIG | 8.51 |
| Same-frame parallel tail | 8.93–9.09; median 9.03 |
| Parallel tail + next-frame backbone + per-instance position-encoding cache | 10.63 / 10.84 / 10.78; median 10.78 |

The current offline 504px CLI enables the fixed decoder, parallel tails, and
backbone prefetch for MIG videos by default. For the native-decoder schedules
above, add `--no-fixed-detr-decoder`; use `--no-parallel-tail` for serial MIG or
`--no-pipeline-backbone` for same-frame overlap only. `--no-mig` selects the
PyTorch reference and disables all MIG optimizations. These historical results
retain their original artifacts and measurement windows; changing defaults
does not remeasure them.

The current CLI also shares live's postprocessing, frame-by-frame tracking
session semantics, and persistent object limits (five per prompt by default,
`--max-objects 0` for uncapped sessions). The historical offline figures above
predate that unification.

An earlier pipeline-only record was 10.21–10.27 FPS, before the final
position-encoding cache improvement. It is not a conflicting measurement of
the same final configuration. The instrumented serial module profile was
111.65 ms/frame; its synchronization hooks serialize work and do not measure
parallel-tail speedup.

See the [historical full-stack report](rocm714_fullstack_evaluation.md) for
per-stage timings, matched-frame A/B windows, excluded pipeline fill/drain,
and multi-object correctness. These FPS values are not default live output
rates.

## Historical host and box results

The old native text path's 7.06 FPS, pure-PT reference around 2.6 FPS, and
box-only 12.21 FPS / DAVIS mean J 81.6% are preserved in the
[legacy-runtime archive](historical/legacy-runtime.md). They do not share
the default live runtime, prompt protocol, or measurement window.

DAVIS J and J&F are different metrics. A box-derived initial mask and a ground
truth initial mask also define different protocols. Do not attribute the gap
to prompt quality alone or present those results as a like-for-like comparison.

## Provenance

The tables above are the checked-in summaries for readers without access to
the development machine. Original JSON, harnesses, and large model artifacts
remain in maintainer storage, outside Git; they are not required to run the
public demo. These identifiers are provenance, not downloadable repository
paths:

| Record | Maintainer artifact identifier, relative to the GPU artifact root |
|---|---|
| Default live | `experiments/latest-fixed-decoder/results/production_250_integrated.json` |
| Decoder A/B and equivalence | `experiments/latest-fixed-decoder/REPORT.md`, `SUMMARY.json`, and `results/` |
| Final hybrid runs and matrix | `experiments/unified-hybrid-clean-keyframe-20260902/REPORT.md`, `swan_run{1,2,3}.json`, and `results/` |
| Resource soak | `experiments/unified-reset-soak/REPORT.md` and `results/swan_full_keyframe_120.json` |
| Offline 7.14 work | Checked-in [full-stack evaluation](rocm714_fullstack_evaluation.md) and its evidence index |

The original fixed-decoder experiment recorded these artifact SHA256 values:

```text
backbone: ae563fa53e15b2a24f823a1c5f6af40329120d49d102e05ef7f4af0198e1cfec
decoder:  141bab3fb6940327dff8548c907ad205267ff6546d24aaae3f413fc6eb30ded3
```

Do not assume a fresh local compilation is byte-identical to those benchmark
files. Re-run the relevant correctness gates and measure the target workload
before accepting a new performance claim.
