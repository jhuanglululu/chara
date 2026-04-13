import torch.nn.functional as F
from torch import nn, Tensor

from .rope import RoPE
from ..configs.model import ModelConfig


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, rope: RoPE) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.d_heads = config.d_head

        self.rope = rope

        self.w_qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.w_qkv(x)

        qkv = qkv.reshape(batch_size, seq_len, self.n_heads, 3 * self.d_heads)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, 3 * d_heads)
        q, k, v = qkv.chunk(3, dim=-1)  # (batch_size, n_heads, seq_len, d_heads)

        q = self.rope(q)
        k = self.rope(k)

        value = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
        value = value.permute(0, 2, 1, 3).reshape(x.shape)

        return self.w_o(value)
