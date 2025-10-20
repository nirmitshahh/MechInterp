"""Misdirection OOD: swap query SPL to wrong system, check 1/2/3-after MSE."""
# Test trace uses SPL for system B after haystack, but continuation is from A.
# H1 (label-based): 1-after gets worse. H2 (observation-based): 2/3-after same.
# Usage: python -m eval --ckpt ... --N 2 --misdirection

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from model.transformer import TraceTransformer
from data.dataset import needle_haystack_dataloader
from data.traces import CONTEXT_LEN, STATE_DIM
import config


def run_misdirection(ckpt_path: str, N: int = 2, n_samples: int = 200, batch_size: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = TraceTransformer(
        vocab_size=ckpt.get("args", {}).get("vocab_size", config.VOCAB_SIZE),
        state_dim=ckpt.get("args", {}).get("state_dim", config.STATE_DIM),
        d_model=ckpt.get("args", {}).get("d_model", config.D_MODEL),
        n_layers=ckpt.get("args", {}).get("n_layers", config.N_LAYERS),
        max_len=CONTEXT_LEN,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    loader = needle_haystack_dataloader(
        n_systems=N, n_samples=n_samples, state_dim=STATE_DIM,
        batch_size=batch_size, seed=42, misdirection=True, sync=False,
    )
    sums = [0.0, 0.0, 0.0]
    counts = [0, 0, 0]
    with torch.no_grad():
        for toks, sts, targets, qpos in loader:
            toks, sts, targets, qpos = toks.to(device), sts.to(device), targets.to(device), qpos.to(device)
            pred = model(toks, sts)
            B = toks.shape[0]
            for k in (1, 2, 3):
                if k > targets.shape[1]:
                    continue
                idx = (qpos + k - 1).clamp(max=pred.shape[1] - 1)
                p = pred[torch.arange(B, device=device), idx]
                t = targets[:, k - 1]
                mse = ((p - t) ** 2).mean().item()
                sums[k - 1] += mse * B
                counts[k - 1] += B
    mses = [sums[i] / counts[i] if counts[i] else 0.0 for i in range(3)]
    return mses


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--n_samples", type=int, default=200)
    args = p.parse_args()
    m1, m2, m3 = run_misdirection(args.ckpt, args.N, args.n_samples)
    print(f"misdirection N={args.N}  1-after={m1:.4f}  2-after={m2:.4f}  3-after={m3:.4f}")
