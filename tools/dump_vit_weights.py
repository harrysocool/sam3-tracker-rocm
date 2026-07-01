#!/usr/bin/env python3
"""Dump SAM3 ViT backbone weights + RoPE to /home/amd/project/npu_iron/weights/vit_full/ as float32 numpy arrays.

Run once before export_weights_bf16.py.
Usage:
    python eval/benchmarks/npu_iron/dump_vit_weights.py [--checkpoint model/sam3] [--imgsz 504] [--out /home/amd/project/npu_iron/weights/vit_full]
"""
import argparse, sys, os
import numpy as np
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='model/sam3')
    p.add_argument('--imgsz', type=int, default=504)
    p.add_argument('--out', default='/home/amd/project/npu_iron/weights/vit_full')
    return p.parse_args()

def main():
    args = parse_args()
    sys.path.insert(0, '.')
    from tracker import rocm_patches
    from tracker.live_inference import SAM3Live

    print(f'Loading model from {args.checkpoint}...')
    live = SAM3Live(checkpoint=args.checkpoint, prompts=['x'],
                    imgsz=args.imgsz, mig=False, redetect_every=1,
                    dtype=torch.float32)
    bb = live.model.detector_model.vision_encoder.backbone
    os.makedirs(args.out, exist_ok=True)

    def save(name, t):
        np.save(f'{args.out}/{name}.npy', t.detach().float().cpu().numpy())

    # ── weights ───────────────────────────────────────────────────────────
    print(f'Dumping weights for {len(bb.layers)} blocks...')
    for i, layer in enumerate(bb.layers):
        a = layer.attention
        save(f'L{i}_qw', a.q_proj.weight); save(f'L{i}_qb', a.q_proj.bias)
        save(f'L{i}_kw', a.k_proj.weight); save(f'L{i}_kb', a.k_proj.bias)
        save(f'L{i}_vw', a.v_proj.weight); save(f'L{i}_vb', a.v_proj.bias)
        save(f'L{i}_ow', a.o_proj.weight); save(f'L{i}_ob', a.o_proj.bias)
        m = layer.mlp
        save(f'L{i}_fc1w', m.fc1.weight); save(f'L{i}_fc1b', m.fc1.bias)
        save(f'L{i}_fc2w', m.fc2.weight); save(f'L{i}_fc2b', m.fc2.bias)
        save(f'L{i}_ln1w', layer.layer_norm1.weight); save(f'L{i}_ln1b', layer.layer_norm1.bias)
        save(f'L{i}_ln2w', layer.layer_norm2.weight); save(f'L{i}_ln2b', layer.layer_norm2.bias)

    # ── RoPE: read from rotary_emb buffers ─────────────────────────────────
    # Window blocks (0-23): S=576, Global blocks (24-31): S=1296
    # Find a window block (window_size>0) and a global block (window_size=0)
    win_layer  = next(l for l in bb.layers if getattr(l, 'window_size', 0) > 0)
    glob_layer = next(l for l in bb.layers if getattr(l, 'window_size', 0) == 0)
    save('rope_win_cos',  win_layer.rotary_emb.rope_embeddings_cos)
    save('rope_win_sin',  win_layer.rotary_emb.rope_embeddings_sin)
    save('rope_glob_cos', glob_layer.rotary_emb.rope_embeddings_cos)
    save('rope_glob_sin', glob_layer.rotary_emb.rope_embeddings_sin)
    print('RoPE win:', win_layer.rotary_emb.rope_embeddings_cos.shape,
          ' glob:', glob_layer.rotary_emb.rope_embeddings_cos.shape)

    # ── reference activations (for C++ smoke test) ────────────────────────
    print('Running reference forward pass for block0_in / final_feat...')
    captured = {}
    def hook_in(module, args, kwargs):
        if 'block0_in' not in captured and args:
            captured['block0_in'] = args[0].detach()
    def hook_out(module, args, output):
        captured['final_feat'] = output[0].detach() if isinstance(output, (tuple,list)) else output.detach()

    h_in  = bb.layers[0].register_forward_pre_hook(hook_in, with_kwargs=True)
    h_out = bb.layers[-1].register_forward_hook(hook_out)

    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz,
                        dtype=torch.float16, device=next(bb.parameters()).device)
    with torch.no_grad():
        bb(dummy)

    h_in.remove(); h_out.remove()

    # block0_in is [B,H,W,C] spatial — reshape to [S,C] for C++ binary
    b0 = captured['block0_in']
    if b0.dim() == 4:
        b0 = b0.reshape(b0.shape[0], -1, b0.shape[-1])
    save('block0_in', b0.squeeze(0))

    ff = captured['final_feat']
    if ff.dim() == 4:
        ff = ff.reshape(ff.shape[0], -1, ff.shape[-1])
    save('final_feat', ff.squeeze(0))

    n = len(os.listdir(args.out))
    print(f'Done. {n} files written to {args.out}/')
    print('Next step: python tools/export_weights_bf16.py')

if __name__ == '__main__':
    main()
