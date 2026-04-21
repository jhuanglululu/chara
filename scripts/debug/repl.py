import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src import chara
from src.chara.caches import empty_transformer_cache
from src.chara.configs import ModelConfig, TrainingConfig


def repl(
    mconfig: ModelConfig,
    tconfig: TrainingConfig,
    tokenizer: Tokenizer,
    model: chara.TransformerLM,
    max_seq_len: int,
):
    with torch.no_grad():
        model.eval()
        while True:
            prompt = input(" > ")
            if prompt.strip().lower() == "exit":
                return

            print(f"response: {prompt}", end="")

            input_ids = tokenizer.encode(prompt).ids[:-1]
            input_tensor = torch.tensor(input_ids, dtype=torch.long)[None, :]
            cache = empty_transformer_cache(1, mconfig, tconfig.device)
            mask = chara.causal_mask(1, len(input_ids)).to(tconfig.device)

            for _ in range(max_seq_len - len(input_ids)):
                logits, cache = model(input_tensor.to(tconfig.device), mask, cache)
                predict_id = torch.argmax(logits[0, -1, :]).item()

                input_ids += [predict_id]
                input_tensor = torch.tensor([predict_id], dtype=torch.long)[None, :]
                mask = None

                print(tokenizer.decode([predict_id]), end="", flush=True)

                if predict_id == tokenizer.token_to_id("<eos>"):
                    break
            print()
