from torch import nn, Tensor

from .configs.model import ModelConfig


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        self.decoder_stack = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model) for _ in range(config.n_layers)]
        )

        self.projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # shared weight projection
        self.projection.weight = self.embedding.weight

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)

        for decoder in self.decoder_stack:
            x = decoder(x)

        return self.projection(x)
