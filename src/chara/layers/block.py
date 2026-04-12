from torch import nn, Tensor

from src.chara.layers import Attention, SwiGluMlp, RmsNorm
from ..configs.model import ModelConfig


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.attention = Attention(config)
        self.norm1 = RmsNorm(config)
        self.mlp = SwiGluMlp(config)
        self.norm2 = RmsNorm(config)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x
