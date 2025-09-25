import regex as re
from typing import Iterable, Iterator
import json
from cs336_basics.tokenization.optim_bpe import (
    remove_special_tokens_and_pretokenize,
    word2bytes,
)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [i.encode("utf-8") for i in self.special_tokens]

        self.vocab_to_id = {v: k for k, v in vocab.items()}

        # Ensure special tokens are in the vocabulary
        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.vocab_to_id:
                # Add to vocab if not already present
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_to_id[token_bytes] = new_id

    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        # Load vocab (assumed to be a JSON file: {token_id: byte_string})
        with open(vocab_filepath, "r", encoding="utf-8") as vf:
            vocab_data = json.load(vf)
            # Optional: convert keys to int if stored as strings
            vocab = {
                int(k): bytes(v, "latin1") if isinstance(v, str) else bytes(v)
                for k, v in vocab_data.items()
            }

        # Load merges (assumed to be a list of pairs like: "a b")
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            lines = mf.readlines()
            # Optional: skip headers like "#version: 0.2"
            merge_pairs = [
                tuple(line.strip().split())
                for line in lines
                if not line.startswith("#") and line.strip()
            ]
            # Convert to byte-pairs
            merges = [(a.encode("utf-8"), b.encode("utf-8")) for a, b in merge_pairs]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _apply_merges(
        self,
        byte_list: list[bytes],
        merges: set[tuple[bytes, bytes]],
        vocab_to_id: dict[bytes, int],
    ) -> list[bytes]:
        while True:
            min_token_id = float("inf")
            best_pair_idx = -1
            merged = None

            for i in range(len(byte_list) - 1):
                pair = (byte_list[i], byte_list[i + 1])
                if pair in merges:
                    combined = pair[0] + pair[1]
                    token_id = vocab_to_id.get(combined)
                    if token_id is not None and token_id < min_token_id:
                        min_token_id = token_id
                        best_pair_idx = i
                        merged = combined

            if best_pair_idx == -1:
                break

            # Apply best merge
            byte_list = (
                byte_list[:best_pair_idx] + [merged] + byte_list[best_pair_idx + 2 :]
            )

        return tuple(byte_list)

    def encode(self, text: str) -> list[int]:
        # pre tokenize
        chunks = remove_special_tokens_and_pretokenize(
            text, self.special_tokens, drop_special=False
        )
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode("utf-8")])
            else:
                # apply bpe merges
                word_list = re.findall(PAT, chunk)
                chunk_tokens = []
                for word in word_list:
                    word_bytes = list(word2bytes(word))
                    merged_word_bytes = self._apply_merges(
                        word_bytes, self.merges, self.vocab_to_id
                    )
                    chunk_tokens.extend(self.vocab_to_id[i] for i in merged_word_bytes)
                tokens.extend(chunk_tokens)
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[t] for t in ids]).decode("utf-8", errors="replace")


if __name__ == "__main__":
    tokenizer = Tokenizer()
