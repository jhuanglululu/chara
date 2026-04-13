import torch
from torch import Tensor
import torch.nn.functional as F


def cross_entropy_loss(logits: Tensor, tokens: Tensor, mask: Tensor) -> Tensor:
    if logits.dim() != 3:
        raise ValueError(f"unexpected shape for input tensor: {logits.shape}")

    (_, _, vocab_size) = logits.shape
    mask = mask[:, 1:]

    valid_count = (~mask).sum()
    if valid_count == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    per_token_loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab_size),
        tokens[:, 1:].reshape(-1),
        reduction="none",
    )

    return per_token_loss.masked_fill(mask.reshape(-1), 0.0).sum() / valid_count
