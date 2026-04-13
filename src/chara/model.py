from torch import nn, Tensor

from .configs.model import ModelConfig
from .layers import DecoderBlock, RmsNorm, RoPE


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        rope = RoPE(config)

        self.decoder_stack = nn.ModuleList(
            [DecoderBlock(config, rope) for _ in range(config.n_layers)]
        )

        self.norm = RmsNorm(config)

        self.projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # shared weight projection
        self.projection.weight = self.embedding.weight

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.embedding(x)

        for decoder in self.decoder_stack:
            x = decoder(x, mask)

        x = self.norm(x)

        return self.projection(x)
