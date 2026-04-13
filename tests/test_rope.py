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
def test_rope(batch_size: int, seq_len: int):
    """test whether input and output shape is the same for rope"""
    rope = chara.layers.RoPE(test_config)

    x = torch.rand(batch_size, test_config.n_heads, seq_len, test_config.d_head)

    with torch.no_grad():
        y = rope(x)

    assert x.shape == y.shape, f"expected {x.shape}, got {y.shape}"
