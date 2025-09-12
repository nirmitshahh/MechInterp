"""Needle-in-haystack test: N systems in haystack, query SPL, predict continuation."""
import numpy as np
from .traces import build_trace
from .orthogonal import evolve

HAYSTACK_SEG_LEN = 10
NEEDLE_CONTINUATION_LEN = 10


def make_needle_haystack_trace(
    systems: list,
    needle_ix: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build test trace: haystack segments (each 10 obs) then query open + 10 continuation."""
    # returns (context, targets) for 1-after, 2-after, ... query indices
    pass
