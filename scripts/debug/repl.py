import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src import chara


def repl(
    device: torch.types.Device,
    tokenizer: Tokenizer,
    model: chara.TransformerLM,  # pyright: ignore[reportGeneralTypeIssues]
    max_response_len: int,
):
    with torch.no_grad():
        while True:
            prompt = input(" > ")
            if prompt.strip().lower() == "exit":
                return

            print(f"response: {prompt}", end="")

            input_ids = tokenizer.encode(prompt).ids[:-1]
            for _ in range(max_response_len):
                input_tensor = torch.tensor(input_ids, dtype=torch.long)[None, :].to(
                    device
                )
                mask = chara.causal_mask(1, len(input_ids))
                logits = model(input_tensor, mask)
                predict_id = torch.argmax(logits[0, -1, :]).item()
                input_ids += [predict_id]

                print(tokenizer.decode([predict_id]), end="", flush=True)

                if predict_id == tokenizer.token_to_id("<eos>"):
                    break
            print()
