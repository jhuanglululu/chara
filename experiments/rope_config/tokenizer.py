from pathlib import Path

from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.pre_tokenizers import Split

from experiments.rope_config.data_gen import Data


def train_tokenizer(datas: list[Data], save_file: str):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    pattern = Regex(r" ?[A-Za-z0-9_]+|\'(?:s)|[.,!?;:'\"]")
    tokenizer.pre_tokenizer = Split(pattern, behavior="isolated", invert=True)

    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=8192,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<find>", "<result>"],
        initial_alphabet=list("0123456789"),
    )

    raw_texts = [d.context for d in datas]
    tokenizer.train_from_iterator(raw_texts, trainer)
    tokenizer.save(save_file)

    return tokenizer


def encode_training_data(
    tokenizer: Tokenizer, context: str, key: str, value: str
) -> tuple[list[int], list[int]]:
    bos_tok = tokenizer.token_to_id("<bos>")
    eos_tok = tokenizer.token_to_id("<eos>")
    find_tok = tokenizer.token_to_id("<find>")
    result_tok = tokenizer.token_to_id("<result>")

    context_ids = tokenizer.encode(context, add_special_tokens=False).ids
    key_ids = tokenizer.encode(key, add_special_tokens=False).ids
    value_ids = tokenizer.encode(value, add_special_tokens=False).ids

    ids = [bos_tok, *context_ids, find_tok, *key_ids, result_tok, *value_ids, eos_tok]

    context_len = 1 + len(context_ids) + 1 + len(key_ids) + 1
    response_len = len(value_ids) + 1
    loss_mask = [1] * context_len + [0] * response_len

    return ids, loss_mask


def encode_testing_data(tokenizer: Tokenizer, context: str, key: str) -> list[int]:
    bos_tok = tokenizer.token_to_id("<bos>")
    find_tok = tokenizer.token_to_id("<find>")
    result_tok = tokenizer.token_to_id("<result>")

    context_ids = tokenizer.encode(context, add_special_tokens=False).ids
    key_ids = tokenizer.encode(key, add_special_tokens=False).ids

    ids = [bos_tok, *context_ids, find_tok, *key_ids, result_tok]
    return ids
