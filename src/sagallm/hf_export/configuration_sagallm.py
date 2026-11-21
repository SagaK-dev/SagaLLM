"""Hugging Face configuration for the native SagaLLM architecture.

This module deliberately imports no SagaLLM package files so exported models
can be loaded as standalone Transformers custom architectures.
"""
from transformers import PretrainedConfig


class SagaLLMConfig(PretrainedConfig):
    model_type = "sagallm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 384,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        intermediate_size: int = 128,
        max_position_embeddings: int = 128,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("tie_word_embeddings", tie_word_embeddings)
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.is_decoder = True

    def native_config_kwargs(self) -> dict:
        return {key: getattr(self, key) for key in (
            "vocab_size", "hidden_size", "num_layers", "num_heads",
            "num_kv_heads", "intermediate_size", "max_position_embeddings",
            "rope_theta", "rms_norm_eps", "tie_word_embeddings",
        )}