import torch.nn.functional as F
from torch import nn, Tensor

from ..configs.model import ModelConfig


class SwiGluMlp(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # for both gate and up projection
        self.w_fused = nn.Linear(config.d_model, 2 * config.d_ff)
        self.w_down = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x) -> Tensor:
        fused = self.w_fused(x)
        gate, up = fused.chunk(2, dim=-1)
        gate = F.silu(gate)
        hidden = gate * up
        return self.w_down(hidden)
