#!/usr/bin/env python3
"""Single-image latency benchmark for the NPU ViT backbone path.

Measures the per-image cost of the NPU backbone (patch-embed on GPU →
NPU subprocess for 32 ViT blocks → FPN neck on GPU), matching a
single-image benchmark setup rather than a streaming video.

Usage:
    python eval/benchmarks/npu_iron/bench_npu_single_image.py \
        --checkpoint model/sam3 --image assets/truck.jpg --imgsz 504 --runs 20
"""
import argparse, sys, time, statistics
sys.path.insert(0, '.')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='model/sam3')
    p.add_argument('--image', default='assets/truck.jpg')
    p.add_argument('--imgsz', type=int, default=504)
    p.add_argument('--runs', type=int, default=20)
    p.add_argument('--warmup', type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    from tracker import rocm_patches  # noqa: F401  (applies ROCm patches)
    import torch
    from PIL import Image
    from tracker.live_inference import SAM3Live
    from tracker.npu_backbone_service import patch_sam3_with_npu_backbone

    print(f"Loading model {args.checkpoint} @ {args.imgsz}px ...")
    live = SAM3Live(checkpoint=args.checkpoint, prompts=['x'],
                    imgsz=args.imgsz, mig=False, redetect_every=1)
    npu_enc = patch_sam3_with_npu_backbone(live.model)

    # Preprocess the single image the same way the streaming path does.
    print(f"Preprocessing {args.image} ...")
    pil = Image.open(args.image).convert('RGB')
    processed = live.processor.video_processor(
        videos=[pil], device=live.device, return_tensors='pt')
    pixel_values = processed.pixel_values_videos[0][0].to(live.dtype)  # [C,H,W]
    pixel_values = pixel_values.unsqueeze(0)                            # [1,C,H,W]
    print(f"  pixel_values: {tuple(pixel_values.shape)} {pixel_values.dtype}")

    # Warmup — first run starts the persistent server + loads weights.
    print(f"\nWarmup ({args.warmup} runs, first is slow: server start + weight load)...")
    for i in range(args.warmup):
        with torch.no_grad():
            npu_enc(pixel_values)
        t = npu_enc.timing
        print(f"  warmup {i}: total={t['total_ms']:.1f}ms  "
              f"embed={t['embed_ms']:.1f}  npu={t['npu_ms']:.1f}  "
              f"neck={t.get('neck_ms',0)-t.get('npu_ms',0)-t.get('embed_ms',0):.1f}")

    # Timed runs.
    print(f"\nTimed runs ({args.runs}):")
    rec = {'total_ms': [], 'embed_ms': [], 'npu_ms': [], 'neck_ms': []}
    for i in range(args.runs):
        with torch.no_grad():
            npu_enc(pixel_values)
        t = npu_enc.timing
        neck_only = t['total_ms'] - t['npu_ms'] - t['embed_ms']
        rec['total_ms'].append(t['total_ms'])
        rec['embed_ms'].append(t['embed_ms'])
        rec['npu_ms'].append(t['npu_ms'])
        rec['neck_ms'].append(neck_only)
        print(f"  run {i:2d}: total={t['total_ms']:7.1f}ms  "
              f"embed={t['embed_ms']:5.1f}  npu={t['npu_ms']:7.1f}  neck={neck_only:5.1f}")

    def stat(name, xs):
        return (f"  {name:10s} mean={statistics.mean(xs):7.1f}ms  "
                f"std={statistics.pstdev(xs):5.1f}  "
                f"min={min(xs):7.1f}  max={max(xs):7.1f}")

    print(f"\n{'='*60}")
    print(f"NPU backbone single-image latency  ({args.image}, {args.imgsz}px, "
          f"{args.runs} runs)")
    print('='*60)
    for k in ('embed_ms', 'npu_ms', 'neck_ms', 'total_ms'):
        print(stat(k.replace('_ms', ''), rec[k]))
    fps = 1000.0 / statistics.mean(rec['total_ms'])
    print(f"\n  throughput: {fps:.2f} img/s  (total, single image, no batching)")

    npu_enc.shutdown()


if __name__ == '__main__':
    main()
