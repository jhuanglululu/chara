import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from src.chara.configs.training import TrainingConfig


class DebugDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, texts: list[str], seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.encodings = tokenizer.encode_batch(texts)
        print(f"max sequence length: {max([len(en.ids) for en in self.encodings])}")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        ids = self.encodings[idx].ids[: self.seq_len]
        ids = ids + [self.tokenizer.token_to_id("<pad>")] * (self.seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)


def get_tokenizer(tokenizer_file: str) -> Tokenizer:
    tokenizer = Tokenizer.from_file(
        tokenizer_file,
    )

    return tokenizer


def get_dataset(tokenizer: Tokenizer, data_file: str, seq_len: int) -> DebugDataset:
    with open(data_file, "r") as f:
        texts = f.read()
    texts = texts.split("\n")
    texts = [x for x in texts if x][:20]
    print(f"dataset size: {len(texts)}")
    return DebugDataset(tokenizer, texts, seq_len=seq_len)


def get_loader(dataset: DebugDataset, config: TrainingConfig) -> DataLoader:
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    return loader
