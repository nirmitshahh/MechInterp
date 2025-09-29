"""Orthogonal matrix sampling (Mezzadri) and state evolution x_{i+1} = U x_i."""
import numpy as np


def sample_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform over O(n) via QR of Gaussian matrix."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # avoid degenerate reflections so we get uniform Haar measure
    d = np.diag(R)
    ph = np.sign(d) + (d == 0)
    Q = Q * ph
    return Q


def evolve(x0: np.ndarray, U: np.ndarray, steps: int) -> np.ndarray:
    """x_{i+1} = U x_i. Returns (steps+1,) + x0.shape."""
    out = np.zeros((steps + 1,) + x0.shape, dtype=np.float64)
    out[0] = x0
    for i in range(steps):
        out[i + 1] = U @ out[i]
    return out


def solve_U_from_observations(observations: np.ndarray) -> np.ndarray:
    """
    Recover U from 6 consecutive observations: U = [x1..x5] @ pinv([x0..x4]).
    observations: (6, d) or (T>=6,) use first 6.
    """
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(-1, obs.shape[0] // 6)
    if obs.shape[0] < 6:
        raise ValueError("need at least 6 observations to recover U")
    X_next = obs[1:6].T   # (d, 5)
    X_curr = obs[0:5].T   # (d, 5)
    U = X_next @ np.linalg.pinv(X_curr)
    return U


def predict_next(U: np.ndarray, x: np.ndarray) -> np.ndarray:
    return U @ np.asarray(x, dtype=np.float64)
