from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model hyperparameters"""

    vocab_size: int
    """size of tokenizer vocabulary, including special tokens"""
    max_seq_len: int
    """maximum supported token sequence length"""
    d_model: int
    """hidden/residual dimension of token representations"""
    n_layers: int
    """number of decoder blocks"""
    n_heads: int = -1
    """number of attention heads per block"""
    d_ff: int = -1
    """hidden dimension of the feed-forward sublayer"""
    rms_norm_eps: float = 1e-6
    """numerical stability constant for RMSNorm"""
    dropout: float = 0.0
    """dropout probability applied during training"""
    d_latent: int = -1
    """dimension of attention latent space"""
    d_rope: int = -1
    """dimension of rotary position embedding"""
    identical_rope: bool = False
    """use per head rotary position embedding instead of continuous"""

    d_head: int = field(init=False)

    def __post_init__(self):
        if self.n_heads == -1:
            if self.d_model % 64 != 0:
                raise ValueError(
                    "d_model is not multiple of 64, can not use default value for n_heads"
                )
            self.n_heads = self.d_model // 64

        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_head = self.d_model // self.n_heads

        if self.d_latent == -1:
            self.d_latent = self.d_head

        if self.d_rope == -1:
            self.d_rope = self.d_head

        if self.d_ff == -1:
            self.d_ff = 4 * self.d_model

        if self.d_rope % 2 != 0:
            raise ValueError("d_rope must be divisible by 2")
