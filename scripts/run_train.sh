#!/bin/bash
# Example: train for 100k steps, save every 25k.
set -e
cd "$(dirname "$0")/.."
python train.py --steps 100000 --save_every 25000 --out_dir checkpoints --seed 0
