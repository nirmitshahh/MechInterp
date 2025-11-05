#!/bin/bash
# Run eval with sweep_N and optional ckpt path.
set -e
cd "$(dirname "$0")/.."
CKPT="${1:-checkpoints/ckpt_100000.pt}"
python eval.py --ckpt "$CKPT" --sweep_N
