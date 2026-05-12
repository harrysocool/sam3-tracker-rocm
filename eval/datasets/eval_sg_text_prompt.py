#!/usr/bin/env python3
"""
SG eval using Sam3VideoModel text prompts — proper cgF1-compatible eval.
Uses noun_phrase from GT annotations as text prompt → detect → track.
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.5.1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

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
                    default=WORKSPACE/"results/eval/saco_sg/saco_sg_30seq_textprompt_preds.json")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.float16

    print(f"Loading Sam3VideoModel ...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(args.checkpoint))
    model = Sam3VideoModel.from_pretrained(str(args.checkpoint)).to(device).to(dtype).eval()
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
