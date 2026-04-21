import torch
from torch import nn, Tensor

from ..configs.model import ModelConfig


class RoPE(nn.Module):
    def __init__(
        self,
        n_heads: int,
        d_rope: int,
        seq_len: int,
        base: int = 10000,
        identical_rope: bool = True,
    ) -> None:
        super().__init__()
        self._build_cache(n_heads, d_rope, seq_len, base, identical_rope)

    def _build_cache(
        self, n_heads: int, d_rope: int, seq_len: int, base: int, identical: bool
    ):
        seq_idx = torch.arange(seq_len).float()

        if identical:
            inv_freq = 1.0 / (base ** (torch.arange(0, d_rope, 2).float() / d_rope))
            freqs = torch.outer(seq_idx, inv_freq)
            cos = freqs.cos()[None, :, :].expand(n_heads, -1, -1)
            sin = freqs.sin()[None, :, :].expand(n_heads, -1, -1)
        else:
            d_all = n_heads * d_rope
            inv_freq = 1.0 / (base ** (torch.arange(0, d_all, 2).float() / d_all))
            freqs = torch.outer(seq_idx, inv_freq)
            cos = freqs.cos().reshape(seq_len, n_heads, d_rope // 2).permute(1, 0, 2)
            sin = freqs.sin().reshape(seq_len, n_heads, d_rope // 2).permute(1, 0, 2)

        cos_cached = torch.concat([cos, cos], dim=-1)
        sin_cached = torch.concat([sin, sin], dim=-1)

        self.cos_cached: Tensor
        self.sin_cached: Tensor
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.register_buffer("cos_cached", cos_cached, persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq_len = x.shape[2]

        if seq_len > self.cos_cached.shape[1]:
            raise ValueError("seq_len exceeds max_seq_len inside RoPE")

        x1, x2 = x.chunk(2, dim=-1)
        neg_half_x = torch.cat([-x2, x1], dim=-1)
        cos = self.cos_cached[:, offset : offset + seq_len, :][None, :, :, :]
        sin = self.sin_cached[:, offset : offset + seq_len, :][None, :, :, :]

        return x * cos + neg_half_x * sin
