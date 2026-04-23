import random
from string import Template
from dataclasses import dataclass


@dataclass
class Data:
    key_value: list[tuple[str, str]]
    context: str


def gen_key() -> str:
    name = random.choice(names)
    adj = random.choice(adjs)
    item = random.choice(items)
    return random.choice(key_templates).substitute(name=name, adj=adj, item=item)


def gen_value(value_len: int) -> str:
    return str(random.randint(0, 10**value_len - 1)).rjust(value_len, "0")


def gen_msg(key: str, value: str) -> str:
    return random.choice(string_templates).substitute(key=key, value=value)


def gen_data(n_chunk: int, chunk_size: int = 3, value_len: int = 10) -> Data:
    data = []
    key_value = []

    while len(key_value) < n_chunk:
        k = gen_key()
        if k not in key_value:
            key_value += [(k, gen_value(value_len))]

    for k, v in key_value:
        chunk_content = random.sample(fillers, chunk_size - 1)
        chunk_content += [gen_msg(k, v)]
        random.shuffle(chunk_content)
        data += chunk_content

    return Data(key_value=key_value, context=" ".join(data))


_string_templates = [
    "The $key has a $value on it.",
    "There is a $value on the $key.",
    "You can see a $value on the $key.",
    "A $value is on the $key.",
    "The $key contains a $value.",
    "The $key shows a $value.",
    "A $value appears on the $key.",
    "On the $key is a $value.",
    "The $key is marked by a $value.",
    "The $value is written on the $key.",
    "A $key carries a $value.",
    "Printed on the $key is a $value.",
]
string_templates = [Template(t) for t in _string_templates]

# name, adj, item
_key_templates = [
    "$adj $item",
    "$name's $item",
    "$name's $adj $item",
    "$adj $item near $name",
    "$adj $item next to $name",
    "$adj $item on top of $name",
]
key_templates = [Template(t) for t in _key_templates]

names = [
    "Herta",
    "Silver Wolf",
    "Castorice",
    "Hyacine",
    "Hysilen",
    "Tribbie",
    "Sparxie",
    "Cerydra",
]

adjs = [
    "red",
    "green",
    "blue",
    "colorful",
    "tall",
    "short",
    "wide",
    "U-shaped",
    "M-shaped",
    "W-shaped",
]

items = [
    "crown",
    "umbrella",
    "hat",
    "mask",
    "shirt",
    "chair",
    "table",
    "car",
    "cat",
    "book",
    "paper",
    "rock",
    "scissors",
    "banana",
    "whiteboard",
    "map",
    "computer",
    "macbook pro",
    "pajamas",
    "wall",
    "bottle",
    "guava",
    "pineapple",
    "cabbage",
    "stove",
    "microwave",
    "hair dryer",
]

fillers = [
    "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration.",
    "The best performing models also connect the encoder and decoder through an attention mechanism.",
    "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
    "Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train",
    "Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU.",
    "On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.",
    "We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
    "The decoder is also composed of a stack of N = 6 identical layers.",
    "In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.",
    "Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.",
    "We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.",
    "This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.",
    "In encoder-decoder attention layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.",
    "This allows every position in the decoder to attend over all positions in the input sequence.",
    "This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].",
    "The encoder contains self-attention layers.",
    "In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder.",
    "Each position in the encoder can attend to all positions in the previous layer of the encoder.",
]
