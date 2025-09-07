"""Interleaved trace generation: segments from different systems + SPLs."""
# SPL = symbolic punctuation label, redrawn per trace
from .orthogonal import sample_orthogonal, evolve
import numpy as np

CONTEXT_LEN = 251
STATE_DIM = 5


def sample_initial_state(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(STATE_DIM) / np.sqrt(STATE_DIM)


def build_trace(
    systems: list[np.ndarray],
    segment_lengths: list[int],
    spl_pairs: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, list]:
    """Interleave segments. Each segment: (open_spl, x_0..x_k, close_spl)."""
    # stub: returns (trace_vectors, metadata) for later tokenization
    segments = []
    for i, (U, length, (o, c)) in enumerate(zip(systems, segment_lengths, spl_pairs)):
        x0 = sample_initial_state(rng)
        states = evolve(x0, U, length)
        segments.append((o, states, c))
    return segments
