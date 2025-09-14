"""Eval MSE at 1-after, 2-after, 3-after query indices."""
import argparse
import torch
from model.transformer import TraceTransformer
from data.traces import CONTEXT_LEN

def eval_mse(model, loader, device):
    model.eval()
    mse_1, mse_2, mse_3 = 0.0, 0.0, 0.0
    n = 0
    # TODO: load needle-haystack batches, get preds at query+1, query+2, query+3
    return mse_1, mse_2, mse_3

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--N", type=int, default=2, help="systems in haystack")
    args = p.parse_args()
    ckpt = torch.load(args.ckpt)
    model = TraceTransformer(vocab_size=64, max_len=CONTEXT_LEN)
    model.load_state_dict(ckpt["model"])
    # m1, m2, m3 = eval_mse(model, ...)
    # print("1-after:", m1, "2-after:", m2, "3-after:", m3)
