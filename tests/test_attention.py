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
)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_attention_shape(batch_size: int, seq_len: int):
    """test whether input and output shape is the same for attention"""
    rope = chara.layers.RoPE(test_config)
    attention = chara.layers.Attention(test_config, rope)
    mask = chara.causal_mask(batch_size, seq_len)

    x = torch.rand(batch_size, seq_len, test_config.d_model)
    with torch.no_grad():
        y = attention(x, mask)

    assert x.shape == y.shape, (
        f"attention input and output shape mismatched: expected {x.shape} got {y.shape}"
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_attention_invariance(batch_size: int, seq_len: int):
    """test whether masking is applied correctly for attention"""
    rope = chara.layers.RoPE(test_config)
    attention = chara.layers.Attention(test_config, rope)
    mask = chara.causal_mask(batch_size, seq_len)

    x1 = torch.rand(batch_size, seq_len, test_config.d_model)
    x2 = x1.clone()
    x2[:, -1, :] = torch.rand(batch_size, test_config.d_model)

    with torch.no_grad():
        y1 = attention(x1, mask)
        y2 = attention(x2, mask)

    assert torch.allclose(y1[:, : seq_len - 1, :], y2[:, : seq_len - 1, :]), (
        "earlier output changed when later input modified"
    )
    assert not torch.allclose(y1[:, seq_len - 1 :, :], y2[:, seq_len - 1 :, :]), (
        "later output unchanged when later input modified"
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_attention_smoke(batch_size: int, seq_len: int):
    """test whether gradient update is correct for attention"""
    rope = chara.layers.RoPE(test_config)
    attention = chara.layers.Attention(test_config, rope)
    mask = chara.causal_mask(batch_size, seq_len)

    x = torch.rand(batch_size, seq_len, test_config.d_model, requires_grad=True)
    y = attention(x, mask)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()

    for name, param in attention.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
