"""Interleaved trace generation: segments from different systems + SPLs."""
from .orthogonal import sample_orthogonal, evolve
import numpy as np

CONTEXT_LEN = 251
STATE_DIM = 5


def sample_initial_state(rng: np.random.Generator, dim: int = STATE_DIM) -> np.ndarray:
    return rng.standard_normal(dim) / np.sqrt(dim)


def _zipf_max_systems(alpha: float = 1.5, n_max: int = 25, rng: np.random.Generator | None = None) -> int:
    """Sample max number of systems in trace from Zipf(alpha, n_max)."""
    if rng is None:
        rng = np.random.default_rng()
    probs = np.arange(1, n_max + 1, dtype=np.float64) ** (-alpha)
    probs /= probs.sum()
    return int(rng.choice(np.arange(1, n_max + 1), p=probs)) + 1  # at least 2


def build_trace(
    context_len: int,
    systems: list[tuple[np.ndarray, np.ndarray]] | None,
    rng: np.random.Generator,
    state_dim: int = STATE_DIM,
    n_systems_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Build one training trace: interleaved segments of length-251 sequence.
    systems: list of (U, x0) per system; if None we draw n_systems from library.
    Returns:
        token_ids: (context_len,) ints, -1 for positions that are state not token
        states: (context_len, state_dim) float, 0 where token-only
        seg_meta: list of {system_ix, open_spl, close_spl, start_pos, end_pos} per segment
    """
    n_max = n_systems_max or _zipf_max_systems(1.5, 25, rng)
    n_sys = min(n_max, 25)
    if systems is None or len(systems) < n_sys:
        systems = [(sample_orthogonal(state_dim, rng), sample_initial_state(rng, state_dim)) for _ in range(n_sys)]

    n_spl_pairs = (n_sys + 7) // 2
    spl_pool = list(rng.permutation(64))[: 2 * n_spl_pairs]
    open_close = [(spl_pool[2 * i], spl_pool[2 * i + 1]) for i in range(n_spl_pairs)]

    token_ids = np.full(context_len, -1, dtype=np.int32)
    states = np.zeros((context_len, state_dim), dtype=np.float64)
    seg_meta = []

    # system_id -> (U, x_last_ix, x_last_state) for resume
    active = {}
    pos = 0
    while pos < context_len:
        pick = rng.integers(0, n_sys)
        U, x0 = systems[pick]
        if pick in active:
            _, last_ix, last_x = active[pick]
            seg_len = min(rng.integers(1, 20), context_len - pos - 3)
            start_ix = last_ix + 1
            seg_states = evolve(last_x, U, seg_len)
            x_last = seg_states[-1]
            last_ix_new = start_ix + seg_len - 1
            active[pick] = (U, last_ix_new, x_last)
        else:
            seg_len = min(rng.integers(2, 20), context_len - pos - 3)
            seg_states = evolve(x0, U, seg_len)
            x_last = seg_states[-1]
            active[pick] = (U, seg_len - 1, x_last)

        need = 2 + seg_len
        if pos + need > context_len:
            break
        o, c = open_close[pick % len(open_close)]
        token_ids[pos] = o
        pos += 1
        token_ids[pos : pos + seg_len] = -2
        states[pos : pos + seg_len] = seg_states[1 : seg_len + 1]
        pos += seg_len
        token_ids[pos] = c
        seg_meta.append({"system_ix": pick, "open_spl": o, "close_spl": c, "start": pos - seg_len - 1, "end": pos})
        pos += 1

    return token_ids, states, seg_meta


def trace_to_sequence(token_ids: np.ndarray, states: np.ndarray, pad_token: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to (T,) token ids and (T, state_dim) states; use pad_token where token-only."""
    T = token_ids.shape[0]
    out_tok = np.where(token_ids >= 0, token_ids, pad_token)
    return out_tok, states
