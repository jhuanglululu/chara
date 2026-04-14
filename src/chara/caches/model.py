import torch
from dataclasses import dataclass

from .decoder import DecoderCache, empty_decoder_cache
from ..configs import ModelConfig, TrainingConfig


@dataclass
class TransformerLMCache:
    decoders: list[DecoderCache]

    def clone(self) -> "TransformerLMCache":
        return TransformerLMCache(
            decoders=[decoder.clone() for decoder in self.decoders]
        )


def empty_transformer_cache(
    batch_size: int, mconfig: ModelConfig, device: torch.types.Device
):
    return TransformerLMCache(
        decoders=[
            empty_decoder_cache(batch_size, mconfig, device)
            for _ in range(mconfig.n_layers)
        ]
    )
