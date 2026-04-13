import torch
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


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_block_shape(batch_size: int, seq_len: int):
    """test whether input and output shape is the same for block"""
    block = chara.layers.DecoderBlock(test_config)
    mask = chara.causal_mask(batch_size, seq_len)

    x = torch.rand(batch_size, seq_len, test_config.d_model)

    with torch.no_grad():
        y = block(x, mask)

    assert x.shape == y.shape, f"expected {x.shape}, got {y.shape}"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_block_smoke(batch_size: int, seq_len: int):
    """test whether gradient update is correct for attention"""
    block = chara.layers.DecoderBlock(test_config)
    mask = chara.causal_mask(batch_size, seq_len)

    x = torch.rand(batch_size, seq_len, test_config.d_model, requires_grad=True)
    y = block(x, mask)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
