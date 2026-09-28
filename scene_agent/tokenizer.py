"""Train and load the 16k BPE tokenizer used by SceneFunctionTransformer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<scene>", "</scene>", "<function>"]


class SceneBPETokenizer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_id = self._special_id("<pad>")
        self.bos_id = self._special_id("<bos>")
        self.eos_id = self._special_id("<eos>")

    def _special_id(self, token: str) -> int:
        value = self.tokenizer.token_to_id(token)
        if value is None:
            raise ValueError(f"tokenizer lacks required special token {token}")
        return value

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @classmethod
    def train(cls, texts: Iterable[str], path: str | Path, vocab_size: int = 16_384) -> "SceneBPETokenizer":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.BPE(unk_token="<pad>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tokenizer.train_from_iterator(texts, trainer=trainer)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(path))
        return cls(tokenizer)

    @classmethod
    def load(cls, path: str | Path) -> "SceneBPETokenizer":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
