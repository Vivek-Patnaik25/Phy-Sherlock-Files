#!/usr/bin/env python3
"""
inference.py — Standalone CLI inference for sherlock-physics-2 (0.1326 baseline)
Usage:
    python inference.py --weights model_weights.pth --test_dir /path/to/test --out submission.csv
"""

import os, sys, json, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from PIL import Image
import torchvision.transforms as transforms

try:
    import networkx as nx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'networkx'])
    import networkx as nx

# ── CONFIG ────────────────────────────────────────────────────────────────────
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
FEAT_DIM     = 384
FLOW_RESIZE  = (64, 64)
FLOW_GAPS    = [1, 3, 5]
MOTION_DIM   = 6
MOTION_TOTAL = len(FLOW_GAPS) * MOTION_DIM   # 18
INPUT_DIM    = FEAT_DIM * 3 + MOTION_TOTAL   # 1170
INFER_BATCH  = 2048
BEAM_SIZE    = 5

# ── MODEL ─────────────────────────────────────────────────────────────────────
class TemporalModel(nn.Module):
    """
    Physics-aware pairwise temporal classifier.
    Input : [feat_i | feat_j | feat_j-feat_i | motion_ij]  (1170-dim)
    Output: P(frame i comes before frame j)
    """
    def __init__(self, in_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512,    256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256,     64), nn.ReLU(),
            nn.Linear(64,       1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

# ── DINOV2 ────────────────────────────────────────────────────────────────────
def load_dino():
    print('Loading DINOv2 ViT-S/14...')
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dino = dino.to(DEVICE).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    print('DINOv2 loaded.')
    return dino

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_features(dino, frames, batch_size=32):
    """Returns (N, 384) float32 numpy."""
    feats = []
    for i in range(0, len(frames), batch_size):
        imgs = [Image.fromarray(f) for f in frames[i:i+batch_size]]
        x = torch.stack([transform(img) for img in imgs]).to(DEVICE)
        with torch.no_grad():
            feats.append(dino(x).cpu().numpy())
        torch.cuda.empty_cache()
    return np.concatenate(feats)

# ── FRAME UTILS ───────────────────────────────────────────────────────────────
def extract_frames(video_path):
    """Extract all frames as RGB numpy arrays. True count from actual reads."""
    cap, frames = cv2.VideoCapture(video_path), []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

# ── MOTION FEATURES ───────────────────────────────────────────────────────────
def compute_optical_flow_features(frame_a, frame_b):
    """6-dim motion vector between two RGB frames."""
    a  = cv2.resize(frame_a, FLOW_RESIZE)
    b  = cv2.resize(frame_b, FLOW_RESIZE)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    flow = cv2.calcOpticalFlowFarneback(
        ga, gb, None,
        pyr_scale=0.5, levels=2, winsize=8,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0
    )
    fx, fy = flow[..., 0], flow[..., 1]
    mag    = np.sqrt(fx**2 + fy**2)
    diff   = np.abs(ga - gb)
    return np.array([fx.mean(), fy.mean(), mag.mean(), mag.std(),
                     diff.mean(), diff.std()], dtype=np.float32)

def build_motion_cache(frames):
    """Returns (N, 18) multiscale motion features."""
    N     = len(frames)
    cache = np.zeros((N, MOTION_TOTAL), dtype=np.float32)
    for i in range(N):
        parts = []
        for g in FLOW_GAPS:
            j = min(i + g, N - 1)
            parts.append(compute_optical_flow_features(frames[i], frames[j]))
        cache[i] = np.concatenate(parts)
    return cache

# ── INFERENCE ─────────────────────────────────────────────────────────────────
def predict_order_graph(model, features, motion_cache):
    """Pairwise graph → topological sort → 0-indexed order."""
    n     = len(features)
    if n == 1:
        return [0]

    pairs     = [(i, j) for i in range(n) for j in range(i+1, n)]
    probs_all = []

    for start in range(0, len(pairs), INFER_BATCH):
        batch_pairs = pairs[start:start+INFER_BATCH]
        rows = []
        for i, j in batch_pairs:
            fi, fj = features[i], features[j]
            mot    = motion_cache[i]
            rows.append(np.concatenate([fi, fj, fj-fi, mot]))
        X = torch.tensor(np.array(rows, dtype=np.float32)).to(DEVICE)
        with torch.no_grad():
            probs_all.append(model(X).squeeze(1).cpu().numpy())
        del X
        torch.cuda.empty_cache()

    probs = np.concatenate(probs_all)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for (i, j), p in zip(pairs, probs):
        if p > 0.5: G.add_edge(i, j, weight=float(p))
        else:       G.add_edge(j, i, weight=float(1-p))

    try:
        order = list(nx.topological_sort(G))
    except Exception:
        order = sorted(range(n), key=lambda x: -G.out_degree(x))

    seen    = set(order)
    missing = [i for i in range(n) if i not in seen]
    return order + missing


def motion_penalty(seq, motion_cache):
    penalty = 0.0
    for k in range(len(seq) - 2):
        a, b, c = seq[k], seq[k+1], seq[k+2]
        penalty += np.linalg.norm(motion_cache[a, :6] - motion_cache[b, :6])
    return penalty


def beam_search_refinement(base_order, motion_cache, beam_size=BEAM_SIZE):
    """Adjacent-swap beam search to minimise motion penalty."""
    n = len(base_order)
    if n <= 2:
        return base_order

    beam = [list(base_order)]
    for _ in range(min(n - 1, 20)):
        candidates = []
        for seq in beam:
            candidates.append(seq)
            for k in range(len(seq) - 1):
                swapped        = seq[:]
                swapped[k], swapped[k+1] = swapped[k+1], swapped[k]
                candidates.append(swapped)

        scored     = sorted(candidates, key=lambda s: motion_penalty(s, motion_cache))
        seen_keys  = set()
        beam       = []
        for s in scored:
            key = tuple(s)
            if key not in seen_keys:
                seen_keys.add(key)
                beam.append(s)
            if len(beam) >= beam_size:
                break

    return beam[0]

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Sherlock physics-2 inference')
    parser.add_argument('--weights',  required=True,  help='Path to model_weights.pth')
    parser.add_argument('--test_dir', required=True,  help='Directory of test .mp4 files')
    parser.add_argument('--out',      default='submission.csv', help='Output CSV path')
    args = parser.parse_args()

    print(f'Device : {DEVICE}')

    # Load model
    model = TemporalModel().to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.eval()
    print(f'Model loaded from {args.weights}')

    # Load DINOv2
    dino = load_dino()

    test_videos = sorted(f for f in os.listdir(args.test_dir) if f.endswith('.mp4'))
    print(f'Running inference on {len(test_videos)} test videos...')

    results = []

    for vid in test_videos:
        vid_path = os.path.join(args.test_dir, vid)
        vid_id   = vid.replace('.mp4', '')

        frames     = extract_frames(vid_path)   # true count from actual reads
        true_N     = len(frames)

        if true_N == 0:
            results.append([vid_id, '[1]'])
            continue

        feats        = get_features(dino, frames)
        motion_cache = build_motion_cache(frames)

        order = predict_order_graph(model, feats, motion_cache)
        order = beam_search_refinement(order, motion_cache)

        # Guarantee correct length
        seen    = set(order)
        missing = [i for i in range(true_N) if i not in seen]
        order   = (order + missing)[:true_N]

        # Safety net — crash loudly not silently
        assert len(order) == true_N and len(set(order)) == true_N, \
            f'Order integrity failed for {vid_id}: len={len(order)} unique={len(set(order))} expected={true_N}'

        # 0-indexed → 1-indexed
        order_str = '[' + ','.join(str(x+1) for x in order) + ']'
        results.append([vid_id, order_str])

        del feats, motion_cache
        torch.cuda.empty_cache()

    # Build and sort submission
    df = pd.DataFrame(results, columns=['ID', 'order'])
    df['_n'] = df['ID'].str.extract(r'(\d+)$').astype(int)
    df = df.sort_values('_n').drop(columns='_n').reset_index(drop=True)
    df.to_csv(args.out, index=False)

    print(f'Saved : {args.out}  ({len(df)} rows)')
    print(df.head(3).to_string())

    # Hard format assertions
    for _, row in df.iterrows():
        o    = row['order']
        assert o.startswith('[') and o.endswith(']'), f"Bad format: {row['ID']}"
        vals = list(map(int, o[1:-1].split(',')))
        assert min(vals) >= 1,            f"0-indexed value in {row['ID']}"
        assert len(vals) == len(set(vals)), f"Duplicate indices in {row['ID']}"

    print('All format assertions passed. Safe to submit.')


if __name__ == '__main__':
    main()
