#!/usr/bin/env python3
"""
SG eval using Sam3VideoModel text prompts — proper cgF1-compatible eval.
Uses noun_phrase from GT annotations as text prompt → detect → track.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json, os, random, sys, time

_TS = datetime.now().strftime("%Y%m%d_%H%M%S")  # one stamp per run; passed in default --out names
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))
from tracker.rocm_env import apply as _apply_rocm_env; _apply_rocm_env()

import torch
from transformers import Sam3VideoModel, AutoProcessor


def encode_rle(mask: np.ndarray, h: int, w: int) -> dict:
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def zero_rle(h: int, w: int) -> dict:
    rle = mask_utils.encode(np.asfortranarray(np.zeros((h, w), dtype=np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def mask_to_binary(mask_t, img_h, img_w):
    mask = mask_t.detach().float().cpu().numpy().squeeze()
    mask = (mask > 0).astype(bool)
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-json", type=Path,
                    default=WORKSPACE/"dataset/gt-annotations/saco_veval_smartglasses_val.json")
    ap.add_argument("--img-root", type=Path,
                    default=WORKSPACE/"dataset/saco_sg/JPEGImages_6fps")
    ap.add_argument("--checkpoint", type=Path, default=WORKSPACE/"model/sam3")
    ap.add_argument("--n-seqs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Max frames per annotation (0=all, recommend 20 for quick test)")
    ap.add_argument("--out", type=Path,
                    default=WORKSPACE/f"results/eval/saco_sg/saco_sg_30seq_textprompt_preds_{_TS}.json")
    ap.add_argument("--imgsz", type=int, default=1008,
                    help="Input resolution (504 or 1008)")
    ap.add_argument("--mig", action="store_true",
                    help="Use MIGraphX acceleration (requires LD_PRELOAD)")
    ap.add_argument("--onnx-dir", type=Path, default=None,
                    help="ONNX artefacts root (default: onnx_files_<imgsz>)")
    args = ap.parse_args()
    if args.onnx_dir is None:
        args.onnx_dir = WORKSPACE / f"onnx_files_{args.imgsz}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.float16

    print(f"Loading Sam3VideoModel (imgsz={args.imgsz}, mig={args.mig}) ...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))

    from transformers import Sam3VideoConfig
    config = Sam3VideoConfig.from_pretrained(str(args.checkpoint))
    if args.imgsz != 1008:
        config.image_size = args.imgsz
        config.low_res_mask_size = 4 * args.imgsz // 14
        new_size = {"height": args.imgsz, "width": args.imgsz}
        new_mask = {"height": 4 * args.imgsz // 14, "width": 4 * args.imgsz // 14}
        for sub in (getattr(processor, "image_processor", None),
                    getattr(processor, "video_processor", None)):
            if sub is not None:
                if hasattr(sub, "size"):      sub.size = new_size
                if hasattr(sub, "mask_size"): sub.mask_size = new_mask
        if hasattr(processor, "target_size"): processor.target_size = args.imgsz

    model = Sam3VideoModel.from_pretrained(str(args.checkpoint), config=config).to(device).to(dtype).eval()

    if args.mig:
        from tracker.migraphx_runtime import MIGraphXBackbone
        from tracker.mig_vision_encoder import patch_sam3_video_model_with_mig
        from tracker.mig_detr_encoder import patch_sam3_video_model_detr_encoder
        from tracker.mig_memory_attention import patch_sam3_video_model_memory_attention
        det_dir = args.onnx_dir / "backbone_detector"
        mxr = MIGraphXBackbone(det_dir / "single_simplified.onnx", det_dir / "tuned.mxr")
        patch_sam3_video_model_with_mig(model, mxr)
        detr_onnx = args.onnx_dir / "detector_modules" / "detr_encoder_simplified.onnx"
        if detr_onnx.exists():
            patch_sam3_video_model_detr_encoder(model, detr_onnx)
        mem_onnx = args.onnx_dir / "tracker_modules" / "memory_attention_fixed_S7_P32.onnx"
        if mem_onnx.exists():
            patch_sam3_video_model_memory_attention(model, mem_onnx)
        print("  MIG patches applied")

    print(f"  loaded in {time.perf_counter()-t0:.1f}s\n")

    with open(args.gt_json) as f:
        gt = json.load(f)
    vid_map = {v["id"]: v for v in gt["videos"]}
    anns = gt["annotations"]

    random.seed(args.seed)
    if args.n_seqs < len(anns):
        anns = random.sample(anns, args.n_seqs)

    max_f = args.max_frames if args.max_frames > 0 else None
    print(f"Text-prompt eval: {len(anns)} anns, seed={args.seed}, max_frames={max_f or 'all'}\n")

    predictions = []
    t_global = time.perf_counter()

    for si, ann in enumerate(anns):
        vid = vid_map[ann["video_id"]]
        img_h, img_w = ann["height"], ann["width"]
        all_fnames = vid["file_names"]
        fnames = all_fnames[:max_f] if max_f else all_fnames
        n_frames_total = len(all_fnames)
        n_frames = len(fnames)
        noun = ann["noun_phrase"]

        pred_segs   = [zero_rle(img_h, img_w)] * n_frames_total
        pred_bboxes = [[0.0, 0.0, 0.0, 0.0]] * n_frames_total
        pred_areas  = [0] * n_frames_total
        pred_score  = 0.0

        try:
            frames = [Image.open(args.img_root / fn).convert("RGB") for fn in fnames]

            session = processor.init_video_session(
                video=frames, inference_device=device, dtype=dtype)
            processor.add_text_prompt(session, noun)

            with torch.inference_mode():
                out0 = model(inference_session=session, frame_idx=0)

            if out0.object_ids:
                best_id = max(out0.object_ids,
                              key=lambda oid: float(out0.obj_id_to_score.get(oid, 0.0)))
                pred_score = float(out0.obj_id_to_score.get(best_id, 0.0))

                def record(fi, out):
                    m = out.obj_id_to_mask.get(best_id)
                    if m is not None:
                        bm = mask_to_binary(m, img_h, img_w)
                        pred_segs[fi] = encode_rle(bm, img_h, img_w)
                        if bm.any():
                            ys, xs = np.where(bm)
                            x1,y1,x2,y2 = xs.min(),ys.min(),xs.max(),ys.max()
                            pred_bboxes[fi] = [float(x1),float(y1),float(x2-x1),float(y2-y1)]
                            pred_areas[fi] = int(bm.sum())

                record(0, out0)
                for fi in range(1, n_frames):
                    with torch.inference_mode():
                        out = model(inference_session=session, frame_idx=fi)
                    record(fi, out)

        except Exception as e:
            print(f"  ERROR ann {ann['id']} ({noun}): {e}")

        predictions.append({
            "video_id":      ann["video_id"],
            "category_id":   ann["category_id"],
            "segmentations": pred_segs,
            "bboxes":        pred_bboxes,
            "areas":         pred_areas,
            "score":         pred_score,
        })

        elapsed = time.perf_counter() - t_global
        eta = elapsed / (si + 1) * (len(anns) - si - 1)
        n_det = sum(1 for a in pred_areas[:n_frames] if a > 0)
        print(f"  [{si+1:3d}/{len(anns)}] vid={ann['video_id']:4d} "
              f"{noun[:28]:28s} | score={pred_score:.2f} "
              f"det={n_det}/{n_frames} | ETA {eta/60:.0f}m")

    with open(args.out, "w") as f:
        json.dump(predictions, f)
    print(f"\nSaved {len(predictions)} → {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
