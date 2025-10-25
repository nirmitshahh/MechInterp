# Edge pruning (Bhaskar et al) to get 1-after vs 2-after circuits.
# L' = k * L_mse + L_edge. No KL. Target sparsity via binary search on threshold.
# Input: trained model, fixed needle-haystack config (e.g. N=5).
# Output: sparse masks per layer/head; evaluate 1-after and 2-after MSE with pruned edges.
#
# Pseudo:
#   for task in ["1_after", "2_after"]:
#     loss = mse_at_query_plus_k(model, data, k=1 or 2) + lambda_edge * |mask|
#     optimize continuous masks; then threshold to 0/1; report mse.
#   Check edge overlap between the two circuits (paper says 0% overlap).

import torch
import torch.nn as nn

# TODO: hook into TraceTransformer blocks, add learnable gates per edge, optimize L', then threshold.
