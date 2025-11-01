"""Plot MSE vs step / haystack N for figures."""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_log(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_sweep_N(results_path: str, out_path: str):
    """results_path: JSONL with lines {N, mse_1, mse_2, mse_3}."""
    rows = load_log(results_path)
    if not rows:
        return
    N = [r["N"] for r in rows]
    m1 = [r["mse_1"] for r in rows]
    m2 = [r["mse_2"] for r in rows]
    m3 = [r["mse_3"] for r in rows]
    fig, ax = plt.subplots()
    ax.plot(N, m1, "k-o", label="1-after")
    ax.plot(N, m2, "b-s", label="2-after")
    ax.plot(N, m3, "r-^", label="3-after")
    ax.set_xlabel("Systems in haystack N")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.set_title("MSE vs haystack size")
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_training_curves(log_path: str, out_path: str, keys: tuple = ("step", "loss")):
    rows = load_log(log_path)
    if not rows:
        return
    steps = [r[keys[0]] for r in rows]
    loss = [r[keys[1]] for r in rows]
    fig, ax = plt.subplots()
    ax.plot(steps, loss)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    fig.savefig(out_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true", help="plot sweep N")
    p.add_argument("--train", action="store_true", help="plot training loss")
    p.add_argument("--log", type=str)
    p.add_argument("--out", type=str, default="plot.png")
    args = p.parse_args()
    if args.sweep and args.log:
        plot_sweep_N(args.log, args.out)
    elif args.train and args.log:
        plot_training_curves(args.log, args.out)
