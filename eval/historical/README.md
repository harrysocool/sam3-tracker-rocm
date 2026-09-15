# Historical evaluation tools

These scripts preserve one-off diagnostics, probes, and profilers from earlier
SAM3 ROCm investigations. They are not part of the supported runtime, release
smoke, or automatic pytest suite. Some depend on historical APIs, shapes, or
artifacts and may require the matching source revision to run.

- `debug/` contains targeted compiler and numerical investigations.
- `probes/` contains the early text-detector integration probes and their
  private postprocessing helper.
- `profilers/` contains pre-full-model profiling entry points with historical
  resolution and artifact assumptions.

Use the maintained tools instead:

- [`tools/text_baseline.py`](../../tools/text_baseline.py) for current offline
  image/video inference and PyTorch diagnosis.
- [`eval/datasets/mask_diff_pt_vs_mig.py`](../datasets/mask_diff_pt_vs_mig.py)
  for the current PT-vs-MIG mask regression.
- [`eval/benchmarks/benchmark_parallel_tail.py`](../benchmarks/benchmark_parallel_tail.py)
  and [`profile_full_mig.py`](../benchmarks/profile_full_mig.py) for maintained
  scheduling and module checks.

Do not interpret timings or correctness conclusions here as current release
results. Current measurement scope and accepted evidence are documented in the
[`evaluation guide`](../../docs/evaluation.md).
