"""Validated, serializable model configuration."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    max_position_embeddings: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        attrs = ("vocab_size", "hidden_size", "num_layers", "num_heads", "num_kv_heads", "intermediate_size", "max_position_embeddings")
        if any(getattr(self, x) <= 0 for x in attrs):
            raise ValueError("All model dimensions must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("Normalization epsilon and RoPE base must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    def parameter_count(self) -> int:
        """Exact count for SagaTransformer with default bias-free linear layers."""
        h, f, n = self.hidden_size, self.intermediate_size, self.num_layers
        kv = h * self.num_kv_heads // self.num_heads
        attention = 2*h*h + 2*h*kv  # q/o + k/v
        mlp = 3*h*f  # gate, up, down
        norms = 2*h  # RMSNorm per block
        embeddings = self.vocab_size*h * (1 if self.tie_word_embeddings else 2)
        return n*(attention+mlp+norms) + embeddings + h

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Model configuration must be a mapping")
        return cls(**data)