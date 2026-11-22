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
        """Byte BPE via a rank heap and linked neighbors (avoids full rescans).

        For equal-rank matches, the leftmost pair merges first. This preserves
        the original implementation's simultaneous left-to-right merge order.
        """
        import heapq

        values = [byte + 4 for byte in text.encode("utf-8")]
        count = len(values)
        if count < 2 or not self.ranks:
            result = values
        else:
            left = [i - 1 for i in range(count)]
            right = [i + 1 for i in range(count)]
            right[-1] = -1
            queue = []

            def enqueue(i: int) -> None:
                if i < 0:
                    return
                j = right[i]
                if j >= 0:
                    pair = (values[i], values[j])
                    rank = self.ranks.get(pair)
                    if rank is not None:
                        heapq.heappush(queue, (rank, i, j))

            for i in range(count - 1):
                enqueue(i)
            while queue:
                rank, i, j = heapq.heappop(queue)
                if right[i] != j or self.ranks.get((values[i], values[j])) != rank:
                    continue  # stale pair after a neighboring merge
                values[i] = rank
                k = right[j]
                right[i] = k
                if k >= 0:
                    left[k] = i
                # Invalidate *both* links of the removed node. Stale heap
                # entries must not merge an index that has left the chain.
                right[j] = -1
                left[j] = -1
                # A newly formed token can only participate in higher-rank
                # merges, so the priority queue remains order-correct.
                enqueue(left[i])
                enqueue(i)
            result = []
            i = 0
            while i >= 0:
                result.append(values[i])
                i = right[i]
        return ([self.SPECIAL["<bos>"]] if add_bos else []) + result + ([self.SPECIAL["<eos>"]] if add_eos else [])

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


class FastBPETokenizer:
    """Optional Rust-backed ByteLevel BPE; stores its engine in one checksumable file.

    Token IDs differ from :class:`ByteBPETokenizer`. Never substitute one
    for the other in an existing checkpoint or uint32 shard dataset.
    """

    SPECIAL = ByteBPETokenizer.SPECIAL
    FORMAT = 'hf_byte_bpe_v1'

    def __init__(self, engine):
        self.engine = engine
        if any(engine.token_to_id(k) != v for k, v in self.SPECIAL.items()):
            raise ValueError('Fast tokenizer special token IDs must be 0..3')

    @staticmethod
    def _package():
        try:
            from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
        except ImportError as exc:
            raise RuntimeError('Fast BPE requires the optional dependency: pip install -e ".[fast-tokenizer]"') from exc
        return Tokenizer, models, trainers, pre_tokenizers, decoders

    @classmethod
    def train(cls, texts, vocab_size: int = 32768, min_frequency: int = 2):
        if not (260 <= vocab_size <= 262144) or min_frequency < 1:
            raise ValueError('Fast BPE vocabulary must be 260..262144 and min_frequency >= 1')
        Tokenizer, models, trainers, pre_tokenizers, decoders = cls._package()
        engine = Tokenizer(models.BPE(unk_token='<unk>'))
        engine.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        engine.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, min_frequency=min_frequency,
            special_tokens=list(cls.SPECIAL),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        # An iterator streams documents into the native library; no Python list is needed.
        engine.train_from_iterator(texts, trainer=trainer)
        return cls(engine)

    @property
    def vocab_size(self) -> int:
        return self.engine.get_vocab_size(with_added_tokens=True)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self.engine.encode(text, add_special_tokens=False).ids
        return ([self.SPECIAL['<bos>']] if add_bos else []) + ids + (
            [self.SPECIAL['<eos>']] if add_eos else [])

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if any(not isinstance(i, int) or i < 0 or i >= self.vocab_size for i in ids):
            raise ValueError('Unknown token ID')
        return self.engine.decode(ids, skip_special_tokens=skip_special)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'type': self.FORMAT, 'engine': json.loads(self.engine.to_str())}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def from_payload(cls, payload: dict):
        if payload.get('type') != cls.FORMAT or not isinstance(payload.get('engine'), dict):
            raise ValueError('Unsupported fast tokenizer format')
        Tokenizer, *_ = cls._package()
        return cls(Tokenizer.from_str(json.dumps(payload['engine'])))

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_payload(json.loads(Path(path).read_text(encoding='utf-8')))


def load_tokenizer(path: str | Path) -> ByteBPETokenizer | FastBPETokenizer:
    """Dispatch from the exact on-disk format; no implicit conversions."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('Invalid tokenizer file')
    if payload.get('type') == 'byte_bpe_v1':
        return ByteBPETokenizer.load(path)
    if payload.get('type') == FastBPETokenizer.FORMAT:
        return FastBPETokenizer.from_payload(payload)
    raise ValueError('Unsupported tokenizer format')