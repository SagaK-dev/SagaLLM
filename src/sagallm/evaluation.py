"""Basic held-out language-model validation; not a standardized benchmark suite."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from .config import ModelConfig
from .model import SagaTransformer
from .tokenization import ByteBPETokenizer
from .training import checkpoint_load, load_tokens, resolve_device


@torch.inference_mode()
def evaluate_language_model(checkpoint: str | Path, tokenizer_file: str | Path, data_file: str | Path,
                            *, block_size: int = 64, batches: int = 4, batch_size: int = 2,
                            device_name: str = "auto", seed: int = 42) -> dict:
    if batches < 1 or batch_size < 1:
        raise ValueError("batches and batch_size must be positive")
    tokenizer = ByteBPETokenizer.load(tokenizer_file)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**state["config"])
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError("Tokenizer and checkpoint vocabulary mismatch")
    if not 1 <= block_size <= config.max_position_embeddings:
        raise ValueError("Invalid block size")
    device = resolve_device(device_name)
    model = SagaTransformer(config)
    checkpoint_load(checkpoint, model)
    model.to(device).eval()
    tokens, digest = load_tokens(data_file, tokenizer)
    if len(tokens) < block_size + 2:
        raise ValueError("Not enough held-out data for evaluation")
    generator = torch.Generator().manual_seed(seed)
    total_loss = 0.0
    correct = 0
    count = 0
    for _ in range(batches):
        starts = torch.randint(0, len(tokens)-block_size-1, (batch_size,), generator=generator).tolist()
        x = torch.stack([tokens[i:i+block_size] for i in starts]).to(device)
        y = torch.stack([tokens[i+1:i+block_size+1] for i in starts]).to(device)
        result = model(x, labels=y)
        count += y.numel()
        total_loss += float(result.loss.item()) * y.numel()
        correct += int((result.logits.argmax(dim=-1) == y).sum().item())
    average = total_loss / count
    return {"loss": average, "perplexity": math.exp(min(average, 80)),
            "next_token_accuracy": correct/count, "evaluated_tokens": count,
            "data_sha256": digest, "model_checkpoint": str(checkpoint),
            "device": str(device), "evaluation_type": "toy_heldout_next_token"}