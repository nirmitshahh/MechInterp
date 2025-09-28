# Decomposing Prediction Mechanisms for In-Context Recall

Toy setup: interleaved linear dynamical systems + SPLs, needle-in-haystack, MSE at 1-after / 2-after / 3-after query.

- `data/` — orthogonal sys (Mezzadri), trace interleaving, needle-haystack (normal / misdirection / sync)
- `model/` — GPT-2 style transformer over mixed SPL + state sequence
- `train.py` — MSE next-state, context 251, ckpt every N steps
- `eval.py` — 1/2/3-after MSE, sweep haystack size N, `--misdirection` / `--sync` for OOD
- `experiments/` — misdirection run, sync run, edge-pruning stub, OLMo translation notes
