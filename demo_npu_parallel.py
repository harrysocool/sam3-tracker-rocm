#!/usr/bin/env python3
"""NPU + GPU parallel streaming demo.

Architecture:
  Frame 0  : PyTorch GPU text detection (~380ms) → init tracker immediately
  Frames 1+: GPU tracker propagation at full speed (~9 FPS, MIGraphX)
  NPU bg   : async re-detection every ~4s
             → if tracker lost objects: reinit from NPU result
             → if tracker still active: skip (let tracker continue)
"""
import argparse, time, cv2, numpy as np, threading, queue
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--text', required=True, nargs='+')
    p.add_argument('--checkpoint', default='model/sam3')
    p.add_argument('--onnx-dir', default='onnx_files_504')
    p.add_argument('--imgsz', type=int, default=504)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--output', default='')
    return p.parse_args()


def overlay_text(frame, lines, y0=30, color=(0,255,0)):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, y0+i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3)
        cv2.putText(frame, line, (10, y0+i*26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)


_COLORS = [
    (0, 200, 80), (255, 80, 0), (0, 80, 255), (0, 220, 220),
    (200, 0, 200), (0, 200, 255), (180, 255, 100), (255, 100, 180),
]

def draw_masks(frame, masks, obj_to_prompt, prompt_color):
    """Draw masks with per-prompt consistent colors (matching demo_live.py style)."""
    for oid, mask in masks.items():
        if not mask.any(): continue
        prompt = obj_to_prompt.get(oid, '')
        color = prompt_color.get(prompt, _COLORS[0])
        overlay = frame.copy()
        overlay[mask > 0.5] = color
        cv2.addWeighted(frame, 0.55, overlay, 0.45, 0, frame)
    return frame


def upscale_mask(mask_imgsz, H, W):
    return cv2.resize(mask_imgsz.astype(np.uint8), (W,H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def mask_to_bbox(mask):
    if not mask.any(): return (0.,0.,0.,0.)
    ys, xs = np.where(mask)
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def scale_box(box, src_w, src_h, imgsz):
    sx, sy = imgsz/src_w, imgsz/src_h
    x1,y1,x2,y2 = box
    return [x1*sx, y1*sy, x2*sx, y2*sy]


def init_trackers_from_detection(sam3_result, bbf, onnx_dir, imgsz, shared,
                                  W, H, next_obj_id):
    """Create SAM3OnnxTracker instances from SAM3 detection result. Returns new dicts."""
    from tracker.sam3_onnx_tracker import SAM3OnnxTracker
    new_trackers, new_scores, new_prompts = {}, {}, {}
    for prompt, oids in sam3_result.get('prompt_to_obj_ids', {}).items():
        for oid in oids:
            bbox = sam3_result['boxes'].get(oid)
            mask = sam3_result['masks'].get(oid)
            score = float(sam3_result['scores'].get(oid, 0.0))
            if bbox is None or mask is None or not np.any(mask):
                continue
            bbox_imgsz = scale_box(bbox, W, H, imgsz)
            new_id = next_obj_id[0]; next_obj_id[0] += 1
            trk = SAM3OnnxTracker(checkpoint=None, onnx_dir=onnx_dir,
                                   imgsz=imgsz, shared=shared)
            trk.init_with_features(*bbf, box=bbox_imgsz)
            new_trackers[new_id] = trk
            new_scores[new_id] = score
            new_prompts[new_id] = prompt
    return new_trackers, new_scores, new_prompts


def main():
    args = parse_args()
    import sys; sys.path.insert(0, '.')
    from tracker.rocm_env import apply as _a; _a()
    import torch
    from tracker.live_inference import SAM3Live
    from tracker.sam3_onnx_tracker import SAM3OnnxTracker, SharedTrackerResources
    from tracker.migraphx_runtime import preprocess_image
    from tracker.npu_backbone_service import patch_sam3_with_npu_backbone

    print('Loading models...')
    t0 = time.perf_counter()

    # mig=True: MIG patches DETR encoder + memory_attention (faster detection).
    # patch_sam3_with_npu_backbone then overrides vision_encoder with NPU backbone,
    # preserving DETR/mem_attn MIG patches.
    live = SAM3Live(checkpoint=args.checkpoint, prompts=args.text,
                    onnx_dir=args.onnx_dir, imgsz=args.imgsz, dtype=torch.float16,
                    mig=True, redetect_every=1)
    npu_enc = patch_sam3_with_npu_backbone(live.model)

    # Shared MIGraphX tracker resources (independent of NPU)
    shared = SharedTrackerResources(onnx_dir=args.onnx_dir, imgsz=args.imgsz, backbone='auto')
    print(f'Models loaded in {time.perf_counter()-t0:.1f}s')

    # Warmup: trigger JIT + NPU binary + MIGraphX init
    print('Warming up...')
    t_w = time.perf_counter()
    dummy = np.zeros((504, 504, 3), dtype=np.uint8)
    live.infer(dummy, full_detection=True)
    shared.run_backbone(preprocess_image(dummy, args.imgsz))
    live.reset_tracking()
    print(f'Warmup done in {time.perf_counter()-t_w:.1f}s')

    cap = cv2.VideoCapture(args.video)
    fps_cap = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Use FFmpeg subprocess for video output — avoids ROCm-OpenCV VideoWriter deadlock
    ffmpeg_proc = None
    if args.output:
        import subprocess as _sp
        ffmpeg_proc = _sp.Popen([
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{W}x{H}', '-pix_fmt', 'bgr24', '-r', str(fps_cap), '-i', 'pipe:0',
            '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', args.output
        ], stdin=_sp.PIPE, stderr=_sp.DEVNULL)

    # Read frame 0 and do immediate PyTorch GPU detection
    ret, frame0 = cap.read()
    assert ret, "Can't read video"
    print(f'\nFrame 0: PyTorch GPU detection (text="{args.text}")...')
    t_det = time.perf_counter()
    init_result = live.infer(frame0, full_detection=True)
    det_ms = (time.perf_counter()-t_det)*1000
    n_init = len(init_result.get('object_ids', []))
    print(f'  {n_init} objects detected in {det_ms:.0f}ms (PyTorch GPU)')

    img_np0 = preprocess_image(frame0, args.imgsz)
    bbf0 = shared.run_backbone(img_np0)
    next_obj_id = [0]
    trackers, tracker_scores, tracker_prompts = init_trackers_from_detection(
        init_result, bbf0, args.onnx_dir, args.imgsz, shared, W, H, next_obj_id)
    # Assign one color per prompt (consistent across frames)
    all_prompts = args.text
    prompt_color = {p: _COLORS[i % len(_COLORS)] for i, p in enumerate(all_prompts)}
    print(f'  Tracker initialized with {len(trackers)} objects')

    # NPU state
    npu_frame_queue = queue.Queue(maxsize=1)
    npu_result_queue = queue.Queue(maxsize=1)
    npu_status = {'kf_count': 0, 'kf_ms': 0.0, 'running': True}

    # Live power monitoring via rocm-smi
    power_status = {'watts': 0.0}
    def power_monitor():
        import subprocess as _sp, re as _re
        while npu_status['running']:
            try:
                out = _sp.run(['rocm-smi', '--showpower'], capture_output=True, text=True, timeout=1).stdout
                m = _re.search(r'Power \(W\): ([\d.]+)', out)
                if m:
                    power_status['watts'] = float(m.group(1))
            except Exception:
                pass
            time.sleep(0.5)
    power_thread = threading.Thread(target=power_monitor, daemon=True)
    power_thread.start()

    def npu_worker():
        """Background: NPU re-detection. Only PyTorch/NPU, no MIGraphX."""
        while npu_status['running']:
            try:
                frame_bgr = npu_frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if frame_bgr is None:
                break
            t_kf = time.perf_counter()
            result = live.infer(frame_bgr, full_detection=True)
            kf_ms = (time.perf_counter()-t_kf)*1000
            npu_status['kf_count'] += 1
            npu_status['kf_ms'] = kf_ms
            n = len(result.get('object_ids', []))
            print(f'  [NPU KF #{npu_status["kf_count"]}] {kf_ms:.0f}ms → {n} objects', flush=True)
            # Push to main thread (evict stale if needed)
            try:
                npu_result_queue.put_nowait((result, frame_bgr.copy()))
            except queue.Full:
                try: npu_result_queue.get_nowait()
                except queue.Empty: pass
                try: npu_result_queue.put_nowait((result, frame_bgr.copy()))
                except queue.Full: pass

    npu_thread = threading.Thread(target=npu_worker, daemon=True)
    npu_thread.start()

    frame_times, prop_times = [], []
    frame_n = 1  # frame0 already processed above
    print(f'\nProcessing {Path(args.video).name} | text="{args.text}"')
    print(f'{"Frame":>6}  {"Time":>7}  {"Obj":>4}  {"FPS":>6}  {"Status"}')

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_n += 1
        if args.max_frames and frame_n > args.max_frames: break

        t_frame = time.perf_counter()

        # Feed latest frame to NPU (evict stale if busy)
        try:
            npu_frame_queue.put_nowait(frame.copy())
        except queue.Full:
            try: npu_frame_queue.get_nowait()
            except queue.Empty: pass
            try: npu_frame_queue.put_nowait(frame.copy())
            except queue.Full: pass

        # GPU tracker propagation (always runs)
        img_np = preprocess_image(frame, args.imgsz)
        bbf = shared.run_backbone(img_np)

        # Check NPU result: reinit ONLY if tracker has no active objects
        try:
            npu_result, kf_frame = npu_result_queue.get_nowait()
            if not trackers:
                # Tracker lost everything → reinit from NPU detection
                kf_bbf = shared.run_backbone(preprocess_image(kf_frame, args.imgsz))
                trackers, tracker_scores, tracker_prompts = init_trackers_from_detection(
                    npu_result, kf_bbf, args.onnx_dir, args.imgsz, shared,
                    W, H, next_obj_id)
                print(f'  → NPU reinit: {len(trackers)} objects', flush=True)
            else:
                pass  # Tracker active → keep running, ignore NPU result
        except queue.Empty:
            pass

        # Propagate all active trackers
        out_masks, out_boxes, out_scores = {}, {}, {}
        dead = []
        for oid, trk in list(trackers.items()):
            mask_imgsz, trk_score = trk.propagate_with_features(*bbf)
            mask = upscale_mask(mask_imgsz, H, W)
            if not mask.any():
                dead.append(oid)
                continue
            out_masks[oid] = mask
            out_boxes[oid] = mask_to_bbox(mask)
            out_scores[oid] = tracker_scores.get(oid, float(trk_score))
        for oid in dead:
            trackers.pop(oid, None)
            tracker_scores.pop(oid, None)

        elapsed_ms = (time.perf_counter()-t_frame)*1000
        frame_times.append(elapsed_ms)
        avg_fps = 1000.0/(sum(frame_times[-10:])/min(len(frame_times),10))
        kf_n = npu_status['kf_count']
        status = f'NPU KF#{kf_n}' if kf_n > 0 else 'GPU only'

        if frame_n % 10 == 0 or frame_n <= 3:
            print(f'{frame_n:6d}  {elapsed_ms:6.0f}ms  {len(out_masks):>4}  {avg_fps:6.1f}  {status}')

        if ffmpeg_proc:
            out_frame = draw_masks(frame.copy(), out_masks, tracker_prompts, prompt_color)
            pw = power_status['watts']
            overlay_text(out_frame, [
                f'Frame {frame_n} | {len(out_masks)} obj | {avg_fps:.1f} FPS',
                f'NPU KF#{kf_n}: {npu_status["kf_ms"]:.0f}ms' if kf_n else 'NPU running...',
                f'GPU tracking: {elapsed_ms:.0f}ms  |  Power: {pw:.0f}W',
            ])
            ffmpeg_proc.stdin.write(out_frame.tobytes())

    # Print summary BEFORE cleanup (os._exit may skip it otherwise)
    print(f'\n=== Summary ===')
    print(f'Frames: {frame_n}')
    print(f'NPU KFs: {npu_status["kf_count"]}  avg={npu_status["kf_ms"]:.0f}ms')
    if frame_times:
        avg_t = sum(frame_times)/len(frame_times)
        print(f'GPU tracking avg: {avg_t:.0f}ms  ({1000/avg_t:.1f} FPS)')
    if args.output:
        print(f'Saved: {args.output}')
    import sys; sys.stdout.flush()

    import signal, os as _os

    # Set 20s hard watchdog covering all cleanup
    def _force_exit(sig, frame):
        _os._exit(0)
    signal.signal(signal.SIGALRM, _force_exit)
    signal.alarm(30)

    cap.release()

    # 1. Finalize FFmpeg FIRST (pure CPU pipe, no GPU conflict)
    if ffmpeg_proc:
        try:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait(timeout=10)
        except Exception:
            ffmpeg_proc.kill()

    # 2. Stop NPU thread
    npu_status['running'] = False
    npu_frame_queue.put(None)
    proc = getattr(npu_enc, '_current_proc', None)
    if proc is not None:
        try: proc.kill()
        except Exception: pass
    npu_thread.join(timeout=6)

    _os._exit(0)


if __name__ == '__main__':
    main()
