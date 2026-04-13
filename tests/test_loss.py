import torch
from torch import Tensor
import torch.nn.functional as F
import pytest

from src import chara


test_config = chara.configs.ModelConfig(
    vocab_size=5000,
    max_seq_len=256,
    d_model=128,
    n_layers=4,
    n_heads=2,
    d_ff=512,
    rms_norm_eps=1e-6,
    dropout=0.1,
    device=torch.device("cpu"),
)


def per_token_cross_entropy_loss(
    input: Tensor, expected: Tensor, mask: Tensor
) -> Tensor:
    (b, s, vocab_size) = input.shape
    mask = mask[:, 1:]

    per_token_loss = F.cross_entropy(
        input[:, :-1, :].reshape(-1, vocab_size),
        expected[:, 1:].reshape(-1),
        reduction="none",
    )
    return per_token_loss.masked_fill(mask.reshape(-1), 0.0).reshape(b, s - 1)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_cross_entropy_loss(batch_size: int, seq_len: int):
    """test whether cross entropy masking works"""
    logits = torch.randn((batch_size, seq_len, test_config.vocab_size))
    tokens = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
    mask = torch.concat(
        [torch.zeros(batch_size, seq_len // 2), torch.ones(batch_size, seq_len // 2)],
        dim=-1,
    ).bool()

    per_token = per_token_cross_entropy_loss(logits, tokens, mask)

    assert (per_token[:, : seq_len // 2 - 1] != 0.0).all(), (
        "incorrect reference per token"
    )
    assert (per_token[:, seq_len // 2 - 1 :] == 0.0).all(), (
        "incorrect reference per token"
    )

    loss = chara.cross_entropy_loss(logits, tokens, mask)

    assert torch.isclose(per_token.sum() / (~mask[:, 1:]).sum(), loss), (
        "computed does not match reference"
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_cross_entropy_loss_zero_valid_token(batch_size: int, seq_len: int):
    """test whether cross entropy loss produce no loss when no valid token"""
    logits = torch.randn((batch_size, seq_len, test_config.vocab_size))
    tokens = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))
    mask = torch.ones((batch_size, seq_len)).bool()

    assert torch.allclose(
        chara.cross_entropy_loss(logits, tokens, mask), torch.tensor(0.0)
    ), "loss is not zero"
