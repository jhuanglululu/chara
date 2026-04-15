import torch
from torch import Tensor
from dataclasses import dataclass

from ..configs import ModelConfig


@dataclass
class DecoderCache:
    atten_k: Tensor
    atten_v: Tensor
    atten_pos_k_rot: Tensor

    def clone(self) -> "DecoderCache":
        return DecoderCache(
            atten_k=self.atten_k.clone(),
            atten_v=self.atten_v.clone(),
            atten_pos_k_rot=self.atten_pos_k_rot.clone(),
        )


def empty_decoder_cache(
    batch_size, mconfig: ModelConfig, device: torch.types.Device
) -> DecoderCache:
    return DecoderCache(
        atten_k=torch.zeros(batch_size, mconfig.n_heads, 0, mconfig.d_head).to(device),
        atten_v=torch.zeros(batch_size, mconfig.n_heads, 0, mconfig.d_head).to(device),
        atten_pos_k_rot=torch.zeros(batch_size, mconfig.n_heads, 0, mconfig.d_rope).to(
            device
        ),
    )
