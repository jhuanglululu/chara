import torch
from torch import Tensor
from dataclasses import dataclass

from ..configs import ModelConfig


@dataclass
class DecoderCache:
    atten_k: Tensor
    atten_k_rot: Tensor

    def clone(self) -> "DecoderCache":
        return DecoderCache(
            atten_k=self.atten_k.clone(),
            atten_k_rot=self.atten_k_rot.clone(),
        )


def empty_decoder_cache(
    batch_size, config: ModelConfig, device: torch.types.Device
) -> DecoderCache:
    return DecoderCache(
        atten_k=torch.zeros(batch_size, 0, config.d_latent).to(device),
        atten_k_rot=torch.zeros(batch_size, 0, config.d_rope).to(device),
    )
