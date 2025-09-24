"""Train transformer on interleaved traces. MSE on next-state prediction."""
# context len 251, orthogonal 5x5
import argparse
import torch
from model.transformer import TraceTransformer
from data.traces import CONTEXT_LEN

# TODO: wire up data loader, loss, optimizer, loop

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--vocab_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=100_000)
    args = p.parse_args()
    model = TraceTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_len=CONTEXT_LEN,
    )
    print(model)
