"""Sync OOD: all haystack sequences have same state at 1-after-query index."""
# Rewind so x10 is shared. Need label to predict 2-after. Tests if 2-after uses label.
# x10 ~ N(0,I/5), then rewind each system so they all hit that x10.
# Usage: python -m experiments.sync --ckpt ... --N 2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from model.transformer import TraceTransformer
from data.dataset import needle_haystack_dataloader
from data.traces import CONTEXT_LEN, STATE_DIM
import config


def run_sync(ckpt_path: str, N: int = 2, n_samples: int = 200, batch_size: int = 16):
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
        batch_size=batch_size, seed=42, misdirection=False, sync=True,
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
    m1, m2, m3 = run_sync(args.ckpt, args.N, args.n_samples)
    print(f"sync N={args.N}  1-after={m1:.4f}  2-after={m2:.4f}  3-after={m3:.4f}")
