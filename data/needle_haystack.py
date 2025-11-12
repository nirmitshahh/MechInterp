"""Needle-in-haystack test: N systems in haystack, query SPL, predict continuation."""
from .orthogonal import evolve
from .traces import sample_initial_state, STATE_DIM
import numpy as np

HAYSTACK_SEG_LEN = 10
NEEDLE_CONTINUATION_LEN = 10


def make_needle_haystack_trace(
    systems: list[np.ndarray],
    needle_ix: int,
    rng: np.random.Generator,
    n_obs_per_system: int = 10,
    continuation_len: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    systems: list of U (d,d). needle_ix: which system is the "needle" (query target).
    Returns:
        token_ids: (L,) -1 at state-only positions
        states: (L, d)
        targets: (continuation_len, d) true next states for query continuation
        query_pos: index of the query open-SPL token (so query_pos+1 is first target)
    """
    n = len(systems)
    d = systems[0].shape[0]
    spl_pairs = [(2 * i, 2 * i + 1) for i in range(n)]
    L_seg = n_obs_per_system

    # haystack: for each system k, emit (open_k, x0..x_{L_seg-1}, close_k)
    # we need to store (open, states, close) and flatten
    segments = []
    for k in range(n):
        x0 = sample_initial_state(rng)
        xs = evolve(x0, systems[k], L_seg - 1)
        ok, ck = spl_pairs[k]
        segments.append((ok, xs[1:], ck))

    # build haystack sequence
    tok_list = []
    st_list = []
    for k in range(n):
        ok, sts, ck = segments[k]
        tok_list.append(ok)
        tok_list.extend([-1] * len(sts))
        tok_list.append(ck)
        st_list.append(np.zeros(d))
        st_list.append(sts)
        st_list.append(np.zeros(d))

    # query: needle open-SPL + continuation from needle system
    U_needle = systems[needle_ix]
    # get last state of needle from its haystack segment
    _, sts_needle, _ = segments[needle_ix]
    x_last = sts_needle[-1]
    cont_states = evolve(x_last, U_needle, continuation_len)[1:]

    o_needle, _ = spl_pairs[needle_ix]
    tok_list.append(o_needle)
    st_list.append(np.zeros(d))
    tok_list.append(-1)
    st_list.append(cont_states[0])
    for i in range(1, continuation_len):
        tok_list.append(-1)
        st_list.append(cont_states[i])

    token_ids = np.array(tok_list, dtype=np.int32)
    states = np.array(np.vstack(st_list), dtype=np.float64)
    targets = np.array(cont_states, dtype=np.float64)
    query_pos = np.where(token_ids == o_needle)[0][-1]

    return token_ids, states, targets, query_pos


def make_misdirection_trace(
    systems: list[np.ndarray],
    needle_ix: int,
    wrong_spl_ix: int,
    rng: np.random.Generator,
    n_obs_per_system: int = 10,
    continuation_len: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Same as needle-haystack but query uses wrong_spl_ix open-SPL; continuation is still from needle_ix."""
    token_ids, states, targets, _ = make_needle_haystack_trace(
        systems, needle_ix, rng, n_obs_per_system, continuation_len
    )
    n = len(systems)
    o_needle = 2 * needle_ix
    o_wrong = 2 * wrong_spl_ix
    token_ids = np.where(token_ids == o_needle, o_wrong, token_ids)
    query_pos = np.where(token_ids == o_wrong)[0][-1]
    return token_ids, states, targets, query_pos


def make_sync_trace(
    systems: list[np.ndarray],
    needle_ix: int,
    rng: np.random.Generator,
    n_obs_per_system: int = 10,
    continuation_len: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    All haystack segments are rewound so they share the same state at index n_obs_per_system
    (i.e. the first position after query open). So 1-after-query is ambiguous; need label for 2-after.
    """
    n = len(systems)
    d = systems[0].shape[0]
    x_shared = sample_initial_state(rng)

    spl_pairs = [(2 * i, 2 * i + 1) for i in range(n)]
    segments = []
    for k in range(n):
        U = systems[k]
        # rewind: want x_{n_obs-1} = x_shared, so x0 = (U.T)^{n_obs-1} @ x_shared
        x0 = np.array(x_shared, dtype=np.float64)
        for _ in range(n_obs_per_system - 1):
            x0 = U.T @ x0
        xs = evolve(x0, U, n_obs_per_system - 1)
        segments.append((spl_pairs[k][0], xs[1:], spl_pairs[k][1]))

    tok_list = []
    st_list = []
    for k in range(n):
        ok, sts, ck = segments[k]
        tok_list.append(ok)
        tok_list.extend([-1] * len(sts))
        tok_list.append(ck)
        st_list.append(np.zeros(d))
        st_list.append(sts)
        st_list.append(np.zeros(d))

    U_needle = systems[needle_ix]
    cont_states = evolve(x_shared, U_needle, continuation_len)[1:]
    o_needle = spl_pairs[needle_ix][0]
    tok_list.append(o_needle)
    st_list.append(np.zeros(d))
    tok_list.append(-1)
    st_list.append(cont_states[0])
    for i in range(1, continuation_len):
        tok_list.append(-1)
        st_list.append(cont_states[i])

    token_ids = np.array(tok_list, dtype=np.int32)
    states = np.array(np.vstack(st_list), dtype=np.float64)
    targets = np.array(cont_states, dtype=np.float64)
    query_pos = np.where(token_ids == o_needle)[0][-1]

    return token_ids, states, targets, query_pos
