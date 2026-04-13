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
    n_heads: int
    """number of attention heads per block"""
    d_ff: int
    """hidden dimension of the feed-forward sublayer"""
    rms_norm_eps: float
    """numerical stability constant for RMSNorm"""
    dropout: float
    """dropout probability applied during training"""

    d_head: int = field(init=False)

    def __post_init__(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must integer multiple of n_heads")

        self.d_head = self.d_model // self.n_heads

        if self.d_head % 2 != 0:
            raise ValueError("d_head must be divisible by 2")
