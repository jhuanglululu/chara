import torch
from torch import Tensor


def causal_mask(batch_size: int, seq_len: int) -> Tensor:
    return (
        torch.triu(torch.ones(seq_len, seq_len), diagonal=1)[None, None, :, :]
        .expand(batch_size, 1, seq_len, seq_len)
        .bool()
    )
