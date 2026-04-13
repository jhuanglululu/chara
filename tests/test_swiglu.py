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
def test_swiglu_mlp(batch_size: int, seq_len: int):
    """test whether input and output shape is the same for swiglu mlp"""
    mlp = chara.layers.SwiGluMlp(test_config)

    x = torch.rand(batch_size, seq_len, test_config.d_model)

    with torch.no_grad():
        y = mlp(x)

    assert x.shape == y.shape, f"expected {x.shape}, got {y.shape}"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [16, 32, 64])
def test_swiglu_mlp_smoke(batch_size: int, seq_len: int):
    """test whether gradient update is correct for swiglu mlp"""
    mlp = chara.layers.SwiGluMlp(test_config)

    x = torch.rand(batch_size, seq_len, test_config.d_model, requires_grad=True)
    y = mlp(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()

    for name, param in mlp.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite gradient for {name}"
