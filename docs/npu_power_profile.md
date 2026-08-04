# SAM3 NPU one-FPS power profile

This profile is the validated low-power configuration for the SAM3 ViT
backbone on the current XDNA2 development system.

## Validated configuration

```text
Backbone: P14 M1536 common-ATB + affine
Binary SHA256: 4b2e69ef82cc3f7758ba4e5c77227a27d71ba5abb1324ca411614a5d3f71a998
Host OpenMP threads: 1
NPU runtime power: scoped performance mode while inference is active
Target cadence: 1 FPS
```

The 100-frame validation produced:

| Metric | P14 NPU | MIGraphX GPU |
|---|---:|---:|
| Mean PPT | 29.660 W | 32.265 W |
| Peak PPT | 32.040 W | 49.071 W |
| Energy/frame | 29.872 J | 32.899 J |
| Latency p50 | 887.502 ms | 71.541 ms |
| Latency p95 | 898.512 ms | 74.664 ms |
| TDR / D-state | 0 / 0 | 0 / 0 |

The NPU profile is intended for approximately one-frame-per-second,
power-constrained operation. It is not a replacement for MIGraphX when high
throughput is required.

## Running a command under the profile

The wrapper starts the installed scoped performance-mode service, exports the
validated NPU binary and thread count, runs the requested command, and restores
PCI runtime power control to `auto` on exit:

```bash
scripts/run_npu_power_profile.sh python demo_npu_streaming.py \
  --checkpoint model/sam3 --video assets/blackswan.mp4 \
  --text swan --imgsz 504
```

It requests sudo authentication once because the systemd performance-mode
service changes the NPU PCI runtime-power setting.

The lower-latency fallback uses two host threads:

```bash
SAM3_NPU_OMP_THREADS=2 scripts/run_npu_power_profile.sh COMMAND [ARGS...]
```

The wrapper checks for an existing amdxdna D-state task before starting and
after the command completes. It does not reboot the machine.

## Environment overrides

```text
SAM3_NPU_BIN          alternate IRON backbone binary
SAM3_NPU_OMP_THREADS  host OpenMP thread count; wrapper default is 1
```

Without the wrapper or these environment variables, the existing tracker
defaults remain unchanged for backward compatibility.
