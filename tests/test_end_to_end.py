import torch
import pytest

from src import chara

test_config = chara.config.ModelConfig(
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
def test_end_to_end(batch_size: int, seq_len: int):
    """test whether output shape is correct for entire model"""
    model = chara.TransformerLM(test_config)
    model.eval()

    x = torch.randint(0, test_config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(x)

    expected_shape = (
        batch_size,
        seq_len,
        test_config.vocab_size,
    )

    assert logits.shape == expected_shape, (
        f"expected ({batch_size}, {seq_len}, {test_config.vocab_size}), got {logits.shape}"
    )
