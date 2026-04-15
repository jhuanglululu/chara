import torch
from torch import nn

from ..configs.model import ModelConfig


class RoPE(nn.Module):
    def __init__(self, config: ModelConfig, base: int = 10000) -> None:
        super().__init__()
        self._build_cache(base, config.d_head, config.max_seq_len)

    def _build_cache(self, base, d_head, seq_len):
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        seq_idx = torch.arange(seq_len).float()

        freqs = torch.outer(seq_idx, inv_freq)
        emb = torch.cat([freqs, freqs], dim=1)

        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0):
        x1, x2 = x.chunk(2, dim=-1)
        neg_half_x = torch.cat([-x2, x1], dim=-1)

        seq_len = x.shape[2]

        x_rope = (
            x * self.cos_cached[offset : offset + seq_len][None, None, :, :]
            + neg_half_x * self.sin_cached[offset : offset + seq_len][None, None, :, :]
        )  # (batch_size, n_heads, seq_len, d_head)

        return x_rope
