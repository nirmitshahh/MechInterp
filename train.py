"""Train transformer on interleaved traces. MSE on next-state prediction."""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.transformer import TraceTransformer
from data.traces import CONTEXT_LEN, STATE_DIM
from data.dataset import TraceDataset, collate_trace
import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=config.D_MODEL)
    p.add_argument("--n_layers", type=int, default=config.N_LAYERS)
    p.add_argument("--vocab_size", type=int, default=config.VOCAB_SIZE)
    p.add_argument("--state_dim", type=int, default=config.STATE_DIM)
    p.add_argument("--steps", type=int, default=config.DEFAULT_STEPS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--save_every", type=int, default=config.SAVE_EVERY)
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    model = TraceTransformer(
        vocab_size=args.vocab_size,
        state_dim=args.state_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_len=CONTEXT_LEN,
        dropout=config.DROPOUT,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    dataset = TraceDataset(args.steps * args.batch_size, CONTEXT_LEN, args.state_dim, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_trace, num_workers=0)

    model.train()
    step = 0
    it = iter(loader)
    while step < args.steps:
        try:
            toks, sts = next(it)
        except StopIteration:
            it = iter(loader)
            toks, sts = next(it)
        pred = model(toks, sts)
        # next-state prediction: predict state at t from context up to t-1, so we align pred[t] with sts[t]
        target = sts[:, 1:]
        pred_slice = pred[:, :-1]
        if target.shape[1] != pred_slice.shape[1]:
            target = target[:, : pred_slice.shape[1]]
            pred_slice = pred_slice[:, : target.shape[1]]
        loss = nn.functional.mse_loss(pred_slice, target)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 5000 == 0:
            print(f"step {step} loss {loss.item():.4f}")
        if step > 0 and step % args.save_every == 0:
            path = os.path.join(args.out_dir, f"ckpt_{step}.pt")
            torch.save({"model": model.state_dict(), "step": step, "args": vars(args)}, path)
            print(f"saved {path}")


if __name__ == "__main__":
    main()
