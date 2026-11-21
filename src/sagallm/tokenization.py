"""Dependency-free byte-level BPE for local prototyping (not optimized for web-scale corpora)."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path


class ByteBPETokenizer:
    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    BASE = 260  # 4 specials + 256 bytes

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = [tuple(x) for x in (merges or [])]
        self.ranks = {pair: self.BASE+i for i, pair in enumerate(self.merges)}
        self.tokens: dict[int, bytes] = {n+4: bytes([n]) for n in range(256)}
        for i, (left, right) in enumerate(self.merges):
            self.tokens[self.BASE+i] = self.tokens[left] + self.tokens[right]

    @property
    def vocab_size(self) -> int:
        return self.BASE + len(self.merges)

    @classmethod
    def train(cls, texts: list[str], vocab_size: int = 512, min_frequency: int = 2) -> "ByteBPETokenizer":
        if not (cls.BASE <= vocab_size <= 65536):
            raise ValueError("Byte BPE vocabulary size must be between 260 and 65536")
        sequences = [[byte+4 for byte in text.encode("utf-8")] for text in texts]
        merges: list[tuple[int, int]] = []
        while cls.BASE + len(merges) < vocab_size:
            counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
            if not counts:
                break
            pair, count = min(counts.items(), key=lambda x: (-x[1], x[0]))
            if count < min_frequency:
                break
            replacement = cls.BASE + len(merges)
            merges.append(pair)
            sequences = [cls._merge(seq, pair, replacement) for seq in sequences]
        return cls(merges)

    @staticmethod
    def _merge(seq: list[int], pair: tuple[int, int], replacement: int) -> list[int]:
        output = []
        index = 0
        while index < len(seq):
            if index + 1 < len(seq) and (seq[index], seq[index+1]) == pair:
                output.append(replacement)
                index += 2
            else:
                output.append(seq[index])
                index += 1
        return output

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        seq = [byte+4 for byte in text.encode("utf-8")]
        while len(seq) > 1:
            candidates = [(self.ranks[(a,b)], (a,b)) for a,b in zip(seq, seq[1:]) if (a,b) in self.ranks]
            if not candidates:
                break
            rank, pair = min(candidates)
            seq = self._merge(seq, pair, rank)
        return ([self.SPECIAL["<bos>"]] if add_bos else []) + seq + ([self.SPECIAL["<eos>"]] if add_eos else [])

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        chunks = []
        for token_id in ids:
            if token_id in self.SPECIAL.values():
                if not skip_special:
                    chunks.append(next(k for k,v in self.SPECIAL.items() if v == token_id).encode())
                continue
            if token_id not in self.tokens:
                raise ValueError(f"Unknown token ID: {token_id}")
            chunks.append(self.tokens[token_id])
        return b"".join(chunks).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"type": "byte_bpe_v1", "merges": self.merges}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "byte_bpe_v1":
            raise ValueError("Unsupported tokenizer format")
        return cls(data["merges"])