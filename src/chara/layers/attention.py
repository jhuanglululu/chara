import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor


from .rope import RoPE
from ..configs import ModelConfig
from ..caches import DecoderCache


def _sdpa(q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor | None = None) -> Tensor:
    """attn_mask is a block mask: True marks blocked positions.

    pytorch has a bug with mps implementation of scaled dot product attention
    when v has a different head dim than q/k, so fall back to a manual impl.
    """
    if q.device.type == "mps":
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=None if attn_mask is None else ~attn_mask
    )


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

        offset = cache.atten_k.shape[2] if cache is not None else 0
        pos_q_rot = self.rope(q, offset=offset)
        pos_k_rot = self.rope(k, offset=offset)

        if cache is not None:
            k = torch.concat([cache.atten_k, k], dim=2)
            v = torch.concat([cache.atten_v, v], dim=2)
            pos_k_rot = torch.concat([cache.atten_pos_k_rot, pos_k_rot], dim=2)

            cache.atten_k = k
            cache.atten_v = v
            cache.atten_pos_k_rot = pos_k_rot

        q = torch.concat([q, pos_q_rot], dim=-1)
        k = torch.concat([k, pos_k_rot], dim=-1)

        value = _sdpa(q, k, v, attn_mask=mask)
        value = value.permute(0, 2, 1, 3).reshape(x.shape)

        return self.w_o(value), cache
