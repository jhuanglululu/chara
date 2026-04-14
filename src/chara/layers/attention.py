import torch
import torch.nn.functional as F
from torch import nn, Tensor


from .rope import RoPE
from ..configs import ModelConfig
from ..caches import DecoderCache


class Attention(nn.Module):
    def __init__(self, config: ModelConfig, rope: RoPE) -> None:
        super().__init__()

        self.n_heads = config.n_heads
        self.d_head = config.d_head

        self.rope = rope

        self.w_qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        cache: DecoderCache | None = None,
    ) -> tuple[Tensor, DecoderCache | None]:
        batch_size, _, _ = x.shape

        qkv = self.w_qkv(x)
        qkv = qkv.reshape(batch_size, -1, self.n_heads, 3 * self.d_head)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_len, 3 * d_head)
        q, k, v = qkv.chunk(3, dim=-1)  # (batch_size, n_heads, seq_len, d_head)

        offset = cache.attention_k.shape[2] if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k = torch.concat([cache.attention_k, k], dim=2)
            v = torch.concat([cache.attention_v, v], dim=2)
            cache = DecoderCache(attention_k=k, attention_v=v)

        value = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if mask is None else ~mask
        )
        value = value.permute(0, 2, 1, 3).reshape(x.shape)

        return self.w_o(value), cache
