# mech interp iclr thing

implementing "Decomposing Prediction Mechanisms for In-Context Recall" (iclr 2026). toy setup: interleaved linear dyn systems + SPLs, needle-in-haystack, 1-after vs 2-after vs 3-after mse.

- `data/` orthogonal sys, traces, needle-haystack
- `model/` gpt2-style transformer
- `train.py` / `eval.py` 
- `experiments/` misdirection, sync ood
