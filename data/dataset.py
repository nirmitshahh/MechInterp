"""Dataset and collate for training traces and needle-haystack eval."""
from .traces import build_trace, CONTEXT_LEN, STATE_DIM
from .needle_haystack import make_needle_haystack_trace, make_misdirection_trace, make_sync_trace
from .orthogonal import sample_orthogonal
import numpy as np
import torch


class TraceDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int, context_len: int, state_dim: int, seed: int = 0):
        self.n_samples = n_samples
        self.context_len = context_len
        self.state_dim = state_dim
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids, states, _ = build_trace(
            self.context_len, None, self.rng, state_dim=self.state_dim
        )
        T = min(token_ids.shape[0], self.context_len)
        tok = np.where(token_ids >= 0, token_ids, 0).astype(np.int64)
        st = states[:T]
        if T < self.context_len:
            tok = np.pad(tok, (0, self.context_len - T), constant_values=0)
            st = np.pad(st, ((0, self.context_len - T), (0, 0)), constant_values=0)
        return torch.from_numpy(tok[: self.context_len]), torch.from_numpy(st[: self.context_len].astype(np.float32))


def collate_trace(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    toks = torch.stack([b[0] for b in batch])
    sts = torch.stack([b[1] for b in batch])
    return toks, sts


class NeedleHaystackDataset(torch.utils.data.Dataset):
    def __init__(self, n_systems: int, n_samples: int, state_dim: int, seed: int, misdirection: bool = False, sync: bool = False):
        rng = np.random.default_rng(seed)
        systems = [sample_orthogonal(state_dim, rng) for _ in range(n_systems)]
        self.samples = []
        for _ in range(n_samples):
            needle_ix = rng.integers(0, n_systems)
            if sync:
                tok, st, targets, qpos = make_sync_trace(systems, needle_ix, rng)
            elif misdirection:
                wrong_ix = (needle_ix + 1 + rng.integers(0, max(1, n_systems - 1))) % n_systems
                tok, st, targets, qpos = make_misdirection_trace(systems, needle_ix, wrong_ix, rng)
            else:
                tok, st, targets, qpos = make_needle_haystack_trace(systems, needle_ix, rng)
            tok = np.where(tok >= 0, tok, 0).astype(np.int64)
            self.samples.append((torch.from_numpy(tok), torch.from_numpy(st.astype(np.float32)), torch.from_numpy(targets.astype(np.float32)), qpos))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]


def needle_haystack_dataloader(
    n_systems: int,
    n_samples: int,
    state_dim: int,
    batch_size: int,
    seed: int = 42,
    misdirection: bool = False,
    sync: bool = False,
) -> torch.utils.data.DataLoader:
    """Build eval batches: normal, misdirection, or sync needle-haystack traces."""
    ds = NeedleHaystackDataset(n_systems, n_samples, state_dim, seed, misdirection, sync)

    def _collate(batch):
        toks = torch.nn.utils.rnn.pad_sequence([b[0] for b in batch], batch_first=True, padding_value=0)
        sts = torch.nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True, padding_value=0)
        targets = torch.stack([b[2] for b in batch])
        qpos = torch.tensor([b[3] for b in batch], dtype=torch.long)
        return toks, sts, targets, qpos

    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
