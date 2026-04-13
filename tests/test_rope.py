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
def test_rope_shape(batch_size: int, seq_len: int):
    """test whether input and output shape is the same for rope"""
    rope = chara.layers.RoPE(test_config)

    x = torch.rand(batch_size, test_config.n_heads, seq_len, test_config.d_head)

    with torch.no_grad():
        y = rope(x)

    assert x.shape == y.shape, f"expected {x.shape}, got {y.shape}"


def test_rope_relative_position():
    rope = chara.layers.RoPE(test_config)
    q = torch.randn(1, 1, 1, test_config.d_head)
    k = torch.randn(1, 1, 1, test_config.d_head)

    x_a = torch.zeros(1, 1, 16, test_config.d_head)
    x_a[:, :, 2] = q
    x_a[:, :, 5] = k

    x_b = torch.zeros(1, 1, 16, test_config.d_head)
    x_b[:, :, 7] = q
    x_b[:, :, 10] = k

    with torch.no_grad():
        y_a = rope(x_a)
        y_b = rope(x_b)

    score_a = (y_a[:, :, 2] * y_a[:, :, 5]).sum()
    score_b = (y_b[:, :, 7] * y_b[:, :, 10]).sum()

    assert torch.allclose(score_a, score_b), (
        "rope does not rotate based on relative position"
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_rope_length(batch_size: int, seq_len: int):
    """test whether rope preserve the length"""
    rope = chara.layers.RoPE(test_config)

    x = torch.rand(batch_size, test_config.n_heads, seq_len, test_config.d_head)

    with torch.no_grad():
        y = rope(x)

    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1)), (
        "length is not preserved in rope"
    )
