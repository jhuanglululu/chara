from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()
trainer = BpeTrainer(
    vocab_size=2048, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
)
tokenizer.train(files=["data/overfit.txt"], trainer=trainer)


tokenizer.post_processor = TemplateProcessing(
    single="<bos> $A <eos>",
    special_tokens=[
        ("<bos>", tokenizer.token_to_id("<bos>")),
        ("<eos>", tokenizer.token_to_id("<eos>")),
    ],
)

tokenizer.save("checkpoints/debug/tokenizer.json")
