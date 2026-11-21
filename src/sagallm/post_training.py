"""Initial supervised fine-tuning with assistant-only token masking."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .config import ModelConfig
from .model import SagaTransformer
from .tokenization import ByteBPETokenizer
from .training import checkpoint_load, checkpoint_save, resolve_device, verify_checkpoint_tokenizer


def format_sft_example(messages: list[dict], tokenizer: ByteBPETokenizer, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Only user→assistant single-turn format is supported in this first version."""
    if len(messages) != 2 or [m.get("role") for m in messages] != ["user", "assistant"]:
        raise ValueError("SFT v1 supports exactly one user message followed by one assistant message")
    if not all(isinstance(m.get("content"), str) and m["content"] for m in messages):
        raise ValueError("Message content must be non-empty text")
    user, assistant = messages
    prefix = [tokenizer.SPECIAL["<bos>"]] + tokenizer.encode("User: " + user["content"] + "\nAssistant: ")
    response = tokenizer.encode(assistant["content"]) + [tokenizer.SPECIAL["<eos>"]]
    combined = prefix + response
    if len(combined) - 1 > max_length:
        raise ValueError("SFT example exceeds the model context window")
    x = torch.tensor(combined[:-1], dtype=torch.long)
    y = torch.tensor(combined[1:], dtype=torch.long)
    y[:len(prefix)-1] = -100
    return x, y


def sft_train(base_checkpoint: str | Path, tokenizer_file: str | Path, dataset_file: str | Path,
              output_checkpoint: str | Path, *, steps: int = 4, lr: float = 1e-4,
              device_name: str = "auto", allow_large: bool = False, seed: int = 42) -> dict:
    if steps < 1 or lr <= 0:
        raise ValueError("steps and lr must be positive")
    state = torch.load(base_checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**state["config"])
    verify_checkpoint_tokenizer(state, tokenizer_file)
    if config.parameter_count() > 200_000_000 and not allow_large:
        raise ValueError("Large model SFT requires --allow-large")
    tokenizer = ByteBPETokenizer.load(tokenizer_file)
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds model vocabulary")
    examples = []
    for line in Path(dataset_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            examples.append(format_sft_example(json.loads(line)["messages"], tokenizer, config.max_position_embeddings))
    if not examples:
        raise ValueError("No valid SFT examples")
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    model = SagaTransformer(config)
    checkpoint_load(base_checkpoint, model)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for i in range(steps):
        x, y = examples[i % len(examples)]
        optimizer.zero_grad(set_to_none=True)
        loss = model(x[None, :].to(device), labels=y[None, :].to(device)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    checkpoint_save(output_checkpoint, model, optimizer, steps, config,
                   {"stage": "sft", "source_checkpoint": str(base_checkpoint),
                    "dataset_sha256": hashlib.sha256(Path(dataset_file).read_bytes()).hexdigest(),
                    "tokenizer_sha256": hashlib.sha256(Path(tokenizer_file).read_bytes()).hexdigest()})
    return {"steps": steps, "final_loss": losses[-1], "checkpoint": str(output_checkpoint),
            "examples": len(examples), "device": str(device), "stage": "sft"}