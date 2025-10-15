"""Eval MSE at 1-after, 2-after, 3-after query indices."""
import argparse
import torch
from model.transformer import TraceTransformer
from data.traces import CONTEXT_LEN, STATE_DIM
from data.dataset import needle_haystack_dataloader
import config


def eval_mse(model, loader, device, k_after: list[int] = (1, 2, 3)):
    model.eval()
    sums = [0.0] * len(k_after)
    counts = [0] * len(k_after)
    with torch.no_grad():
        for toks, sts, targets, qpos in loader:
            toks, sts, targets, qpos = toks.to(device), sts.to(device), targets.to(device), qpos.to(device)
            pred = model(toks, sts)
            B = toks.shape[0]
            for i, k in enumerate(k_after):
                if k > targets.shape[1]:
                    continue
                # pred at position qpos + k is the model's prediction for state at qpos + k
                # targets has shape (B, cont_len, d); targets[:, k-1] is the k-th state after query
                # model outputs pred[b, pos] = predicted state at pos+1, so pred[b, qpos[b]+k-1] predicts state at qpos[b]+k
                q = qpos
                idx = (q + k - 1).clamp(max=pred.shape[1] - 1)
                p = pred[torch.arange(B, device=device), idx]
                t = targets[:, k - 1]
                mse = ((p - t) ** 2).mean().item()
                sums[i] += mse * B
                counts[i] += B
    return [sums[i] / counts[i] if counts[i] else 0.0 for i in range(len(k_after))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--N", type=int, default=2, help="systems in haystack")
    p.add_argument("--n_samples", type=int, default=config.EVAL_N_SAMPLES)
    p.add_argument("--batch_size", type=int, default=config.EVAL_BATCH_SIZE)
    p.add_argument("--sweep_N", action="store_true", help="sweep N for haystack size fig")
    p.add_argument("--misdirection", action="store_true")
    p.add_argument("--sync", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("args", {})
    model = TraceTransformer(
        vocab_size=cfg.get("vocab_size", config.VOCAB_SIZE),
        state_dim=cfg.get("state_dim", config.STATE_DIM),
        d_model=cfg.get("d_model", config.D_MODEL),
        n_layers=cfg.get("n_layers", config.N_LAYERS),
        max_len=CONTEXT_LEN,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)

    if args.sweep_N:
        for n in config.HAYSTACK_N_SWEEP:
            loader = needle_haystack_dataloader(
                n_systems=n, n_samples=args.n_samples, state_dim=config.STATE_DIM,
                batch_size=args.batch_size, seed=42, misdirection=args.misdirection, sync=args.sync,
            )
            mses = eval_mse(model, loader, device, k_after=[1, 2, 3, 7, 8])
            print(f"N={n} 1-after={mses[0]:.4f} 2-after={mses[1]:.4f} 3-after={mses[2]:.4f} 7-after={mses[3]:.4f} 8-after={mses[4]:.4f}")
    else:
        loader = needle_haystack_dataloader(
            n_systems=args.N, n_samples=args.n_samples, state_dim=config.STATE_DIM,
            batch_size=args.batch_size, seed=42, misdirection=args.misdirection, sync=args.sync,
        )
        mses = eval_mse(model, loader, device)
        print(f"1-after: {mses[0]:.4f}  2-after: {mses[1]:.4f}  3-after: {mses[2]:.4f}")


if __name__ == "__main__":
    main()
