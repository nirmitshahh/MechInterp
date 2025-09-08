"""GPT-2 style transformer for next-token prediction on interleaved traces."""
import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        q = self.Wq(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1)
        out = torch.matmul(self.dropout(scores), v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.Wo(out)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TraceTransformer(nn.Module):
    """Next-token predictor for mixed SPL + continuous observation sequence."""

    def __init__(
        self,
        vocab_size: int,
        state_dim: int = 5,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, state_dim)

    def forward(self, token_ids: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        B, T, _ = states.shape
        tok_emb = self.token_emb(token_ids)
        st_emb = self.state_proj(states)
        x = tok_emb + st_emb + self.pos_emb[:, :T]
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))
