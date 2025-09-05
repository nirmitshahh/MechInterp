"""Orthogonal matrix sampling (Mezzadri) and state evolution x_{i+1} = U x_i."""
import numpy as np


def sample_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform over O(n) via QR of Gaussian matrix."""
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    return Q


def evolve(x0: np.ndarray, U: np.ndarray, steps: int) -> np.ndarray:
    """x_{i+1} = U x_i. Returns (steps+1,) + x0.shape."""
    out = np.zeros((steps + 1,) + x0.shape)
    out[0] = x0
    for i in range(steps):
        out[i + 1] = U @ out[i]
    return out
