import sys

sys.path.insert(0, ".")

import torch

from src import chara
from src.chara.configs.training import TrainingConfig
from scripts.debug.dataset import get_tokenizer, get_dataset, get_loader
from scripts.debug.repl import repl


seq_len = 320
batch_size = 4
loss_thresh = 5e-4

tconfig = TrainingConfig(
    batch_size=batch_size,
    epochs=256,
    learning_rate=3e-4,
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu"),
)

print(f"using {tconfig.device}")

tokenizer = get_tokenizer("checkpoints/debug/tokenizer.json")
dataset = get_dataset(tokenizer, "data/overoverfit.txt", seq_len=seq_len)
loader = get_loader(dataset, config=tconfig)

mconfig = chara.configs.ModelConfig(
    vocab_size=tokenizer.get_vocab_size(),
    max_seq_len=seq_len,
    d_model=256,
    n_layers=4,
    d_rope=8,
    d_latent=8,
)

model = chara.TransformerLM(mconfig)
model.to(tconfig.device)

total = sum(p.numel() for p in model.parameters())
print(f"{total / 1e6:.1f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=tconfig.learning_rate)

for epoch in range(tconfig.epochs):
    min_loss = float("inf")

    for input_ids in loader:
        optimizer.zero_grad()

        pad_mask = input_ids == tokenizer.token_to_id("<pad>")
        causal = chara.causal_mask(input_ids.shape[0], input_ids.shape[1])
        mask = causal | pad_mask[:, None, None, :]

        input_ids = input_ids.to(tconfig.device)
        mask = mask.to(tconfig.device)

        logits, _ = model(input_ids, mask)
        loss = chara.cross_entropy_loss(
            logits, input_ids, pad_mask.bool().to(tconfig.device)
        )

        if loss.item() < min_loss:
            min_loss = loss.item()

        loss.backward()
        optimizer.step()

    print(f"\repoch: {epoch}, loss: {min_loss:.1e}", end="")
    if loss_thresh is not None and min_loss < loss_thresh:
        print(f"\nearly exiting", end="")
        break

print()


repl(mconfig, tconfig, tokenizer, model, seq_len)
