from torch import nn

from ..configs.model import ModelConfig


class RmsNorm(nn.RMSNorm):
    """wrapper for rms norm"""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config.d_model, eps=config.rms_norm_eps)
