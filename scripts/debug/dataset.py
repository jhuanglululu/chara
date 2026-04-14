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


def lazy_file_split(filepath, sep, chunk_size=8192):
    remainder = ""
    with open(filepath, "r") as f:
        while chunk := f.read(chunk_size):
            remainder += chunk
            *parts, remainder = remainder.split(sep)
            yield from parts
        if remainder:
            yield remainder


def get_dataset(tokenizer: Tokenizer, data_file: str, seq_len: int) -> DebugDataset:
    texts = []
    for text in lazy_file_split(data_file, "\n\n"):
        if text:
            texts += [text.strip()]
        if len(texts) >= 10_000:
            break

    print(f"dataset size: {len(texts)}")
    return DebugDataset(tokenizer, texts, seq_len=seq_len)


def get_loader(dataset: DebugDataset, config: TrainingConfig) -> DataLoader:
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    return loader
