import torch
from torch import Tensor
from dataclasses import dataclass

from ..configs import ModelConfig


@dataclass
class DecoderCache:
    attention_k: Tensor
    attention_v: Tensor

    def clone(self) -> "DecoderCache":
        return DecoderCache(
            attention_k=self.attention_k.clone(),
            attention_v=self.attention_v.clone(),
        )


def empty_decoder_cache(
    batch_size, mconfig: ModelConfig, device: torch.types.Device
) -> DecoderCache:
    # (batch_size, n_heads, seq_len, d_head)
    return DecoderCache(
        attention_k=torch.zeros(batch_size, mconfig.n_heads, 0, mconfig.d_head).to(
            device
        ),
        attention_v=torch.zeros(batch_size, mconfig.n_heads, 0, mconfig.d_head).to(
            device
        ),
    )
