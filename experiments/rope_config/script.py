# frequently changed settings

n_chunk = 3
chunk_size = 3
n_data = 1024
identical_rope = False

seq_len = 310
batch_size = 32
epochs = 100

create_test = True

import sys

sys.path.insert(0, ".")

import json
import datetime
from pathlib import Path
from dataclasses import asdict

from tokenizers import Tokenizer

from experiments.rope_config.data_gen import Data, gen_data
from experiments.rope_config.tokenizer import (
    encode_testing_data,
    encode_training_data,
    train_tokenizer,
)

data_path = "experiments/rope_config/data/data_latest.json"
test_path = "experiments/rope_config/data/test.json"
tokenizer_path = "experiments/rope_config/data/tokenizer_latest.json"

if not Path(test_path).exists():
    datas = [gen_data(n_chunk, chunk_size) for _ in range(n_data)]

    with open(test_path, "w+") as file:
        json.dump(
            {
                "datetime": str(datetime.datetime.now()),
                "n_chunk": n_chunk,
                "n_data": n_data,
                "datas": [asdict(d) for d in datas],
            },
            file,
            indent=4,
        )

if not Path(data_path).exists():
    datas = [gen_data(n_chunk, chunk_size) for _ in range(n_data)]

    with open(data_path, "w+") as file:
        json.dump(
            {
                "datetime": str(datetime.datetime.now()),
                "n_chunk": n_chunk,
                "n_data": n_data,
                "datas": [asdict(d) for d in datas],
            },
            file,
            indent=4,
        )
else:
    with open(data_path, "r") as file:
        full_data = json.load(file)
        datas = [Data(**d) for d in full_data["datas"]]

if not Path(tokenizer_path).exists():
    tokenizer = train_tokenizer(datas, tokenizer_path)
else:
    tokenizer = Tokenizer.from_file(tokenizer_path)

# print max len

max = 0
for data in datas:
    for key, value in data.key_value:
        ids, _ = encode_training_data(tokenizer, data.context, key, value)
        if len(ids) > max:
            max = len(ids)

print(f"data max len: {max}")


# dataset wrapper
import torch


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(
        self, tokenizer: Tokenizer, datas: list[Data], n_chunk: int, seq_len: int
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.datas = datas
        self.n_chunk = n_chunk

        self.n_data = self.n_chunk * len(self.datas)

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        data = self.datas[idx // self.n_chunk]
        key, value = data.key_value[idx % self.n_chunk]

        ids, response_mask = encode_training_data(
            self.tokenizer, data.context, key, value
        )

        pad_id = self.tokenizer.token_to_id("<pad>")
        pad_len = self.seq_len - len(ids)
        pad_mask = [0] * len(ids) + [1] * pad_len
        ids = ids + [pad_id] * pad_len
        response_mask = response_mask + [1] * pad_len

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(pad_mask, dtype=torch.long).bool(),
            torch.tensor(response_mask, dtype=torch.long).bool(),
        )


# model start here

from src import chara

config = chara.configs.ModelConfig(
    vocab_size=tokenizer.get_vocab_size(),
    max_seq_len=seq_len,
    d_model=512,
    n_layers=4,
    d_rope=8,
    d_latent=16,
    identical_rope=identical_rope,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using {device}")

loader = torch.utils.data.DataLoader(
    RetrievalDataset(tokenizer, datas, n_chunk, seq_len),
    batch_size=batch_size,
    shuffle=True,
)


model = chara.TransformerLM(config)
model.to(device)
model.compile()

total = sum(p.numel() for p in model.parameters())
print(f"{total / 1e6:.1f}M parameters")

muon_params = []
adamw_params = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue

    is_embedding = "embedding" in name

    if is_embedding or p.ndim != 2:
        adamw_params += [p]
    else:
        muon_params += [p]

muon_optim = torch.optim.Muon(muon_params, lr=0.002)

adamw_optim = torch.optim.AdamW(adamw_params, lr=3e-4)

print("training start")

for epoch in range(epochs):
    min_loss = float("inf")

    for input_ids, pad_mask, response_mask in loader:
        muon_optim.zero_grad()
        adamw_optim.zero_grad()

        causal = chara.causal_mask(input_ids.shape[0], input_ids.shape[1])
        mask = causal | pad_mask[:, None, None, :]

        input_ids = input_ids.to(device)
        mask = mask.to(device)

        logits, _ = model(input_ids, mask)
        loss = chara.cross_entropy_loss(logits, input_ids, response_mask.to(device))

        if loss.item() < min_loss:
            min_loss = loss.item()

        loss.backward()
        muon_optim.step()
        adamw_optim.step()

    print(f"\repoch: {epoch}, loss: {min_loss:.1e}", end="")

print()

with torch.no_grad():
    model.eval()

    while True:
        context = input("context: ")
        key = input("key: ")

        input_ids = encode_testing_data(tokenizer, context, key)
        input_tensor = torch.tensor(input_ids, dtype=torch.long)[None, :]

        cache = chara.caches.empty_transformer_cache(1, config, device)
        mask = chara.causal_mask(1, len(input_ids)).to(device)

        for _ in range(seq_len - len(input_ids)):
            logits, cache = model(input_tensor.to(device), mask, cache)
            predict_id = torch.argmax(logits[0, -1, :]).item()

            input_ids += [predict_id]
            input_tensor = torch.tensor([predict_id], dtype=torch.long)[None, :]
            mask = None

            print(tokenizer.decode([predict_id]), end="", flush=True)

            if predict_id == tokenizer.token_to_id("<eos>"):
                break
        print()
