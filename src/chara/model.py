from torch import nn, Tensor


from .configs.model import ModelConfig
from .layers import DecoderBlock, RmsNorm, RoPE
from .caches import TransformerLMCache


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        rope = RoPE(
            config.n_heads,
            config.d_rope,
            config.max_seq_len,
            identical_rope=config.identical_rope,
        )

        self.decoder_stack = nn.ModuleList(
            [DecoderBlock(config, rope) for _ in range(config.n_layers)]
        )

        self.norm = RmsNorm(config)

        self.projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # shared weight projection
        self.projection.weight = self.embedding.weight

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        cache: TransformerLMCache | None = None,
    ) -> tuple[Tensor, TransformerLMCache | None]:
        x = self.embedding(x)
        if cache is not None:
            cache = cache.clone()

        for idx, decoder in enumerate(self.decoder_stack):
            if cache is not None:
                x, cache.decoders[idx] = decoder(x, mask, cache.decoders[idx])
            else:
                x, _ = decoder(x, mask, None)

        x = self.norm(x)

        return self.projection(x), cache
