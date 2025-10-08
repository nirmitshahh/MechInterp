"""Misdirection OOD: swap query SPL to wrong system, check 1/2/3-after MSE."""
# Test trace uses SPL for system B after haystack, but continuation is from A.
# H1: 1-after gets worse (label-based). H2: 2/3-after same (observation-based).
# run eval with --misdirection to use wrong SPL
