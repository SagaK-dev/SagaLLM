"""Single-device pretraining, validation, and resumable checkpoints."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .config import ModelConfig
from .model import SagaTransformer
from .tokenization import ByteBPETokenizer


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if device.type == "mps" and not (hasattr(torch.backends,"mps") and torch.backends.mps.is_available()):
        raise ValueError("MPS is not available")
    return device


def load_tokens(path: str | Path, tokenizer: ByteBPETokenizer) -> tuple[torch.Tensor, str]:
    raw = Path(path).read_bytes()
    data = tokenizer.encode(raw.decode("utf-8"), add_eos=True)
    if len(data) < 4:
        raise ValueError("Dataset is too short")
    return torch.tensor(data, dtype=torch.long), hashlib.sha256(raw).hexdigest()


def sample_batch(tokens: torch.Tensor, block_size: int, batch_size: int, generator: torch.Generator,
                 device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) < block_size + 2:
        raise ValueError("Dataset must contain at least block_size + 2 tokens")
    starts = torch.randint(0, len(tokens)-block_size-1, (batch_size,), generator=generator)
    x = torch.stack([tokens[i:i+block_size] for i in starts.tolist()]).to(device)
    y = torch.stack([tokens[i+1:i+block_size+1] for i in starts.tolist()]).to(device)
    return x, y


def checkpoint_save(path: str | Path, model: SagaTransformer, optimizer: torch.optim.Optimizer,
                    step: int, config: ModelConfig, train_state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
             "config": config.to_dict(), "train_state": train_state, "cpu_rng_state": torch.get_rng_state()}
    torch.save(state, temporary)
    temporary.replace(path)


def checkpoint_load(path: str | Path, model: SagaTransformer,
                    optimizer: torch.optim.Optimizer | None = None) -> dict:
    # Checkpoints are still trusted input only; never accept untrusted files.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != model.config.to_dict():
        raise ValueError("Checkpoint model configuration mismatch")
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


def pretrain(model_config: ModelConfig, tokenizer_path: str | Path, dataset_path: str | Path,
             checkpoint_path: str | Path, *, steps: int = 8, batch_size: int = 2,
             block_size: int = 32, grad_accum: int = 1, lr: float = 3e-4,
             warmup_steps: int = 1, seed: int = 42, device_name: str = "auto",
             resume: bool = False, allow_large: bool = False, log_path: str | Path | None = None) -> dict:
    if model_config.parameter_count() > 200_000_000 and not allow_large:
        raise ValueError("Models above 200M parameters require explicit --allow-large")
    if steps < 1 or batch_size < 1 or grad_accum < 1 or lr <= 0:
        raise ValueError("Invalid training hyperparameters")
    if not 1 <= block_size <= model_config.max_position_embeddings:
        raise ValueError("Invalid block_size")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    if tokenizer.vocab_size > model_config.vocab_size:
        raise ValueError("Tokenizer vocabulary exceeds model vocabulary")
    tokens, data_sha = load_tokens(dataset_path, tokenizer)
    token_sha = hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest()
    device = resolve_device(device_name)
    torch.manual_seed(seed)
    random_generator = torch.Generator(device="cpu").manual_seed(seed)
    model = SagaTransformer(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    starting_step = 0
    if resume:
        loaded = checkpoint_load(checkpoint_path, model, optimizer)
        state = loaded["train_state"]
        if state["dataset_sha256"] != data_sha or state["tokenizer_sha256"] != token_sha:
            raise ValueError("Cannot resume: dataset or tokenizer changed")
        random_generator.set_state(state["data_rng_state"])
        torch.set_rng_state(loaded["cpu_rng_state"])
        starting_step = int(loaded["step"])
        if starting_step >= steps:
            raise ValueError("Requested steps must exceed checkpoint step")
    logs = []
    start_time = time.monotonic()
    model.train()
    for step in range(starting_step, steps):
        # The schedule is a function of absolute step; keep total steps fixed when
        # exact scheduler-state reproduction across interrupted runs matters.
        rate = min(1.0, (step+1)/max(warmup_steps,1))
        if step >= warmup_steps:
            rate = 0.5 * (1+math.cos(math.pi*(step-warmup_steps)/max(1,steps-warmup_steps)))
        for group in optimizer.param_groups:
            group["lr"] = lr * rate
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(grad_accum):
            x, y = sample_batch(tokens, block_size, batch_size, random_generator, device)
            out = model(x, labels=y)
            (out.loss / grad_accum).backward()
            losses.append(out.loss.item())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        elapsed = max(time.monotonic()-start_time,1e-6)
        metrics = {"step": step+1, "loss": sum(losses)/len(losses), "lr": optimizer.param_groups[0]["lr"],
                   "grad_norm": grad_norm, "tokens_seen": (step+1)*batch_size*block_size*grad_accum,
                   "steps_per_second": (step+1-starting_step)/elapsed, "device": str(device)}
        logs.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    train_state = {"dataset_sha256": data_sha, "tokenizer_sha256": token_sha,
                   "data_rng_state": random_generator.get_state(), "base_lr": lr,
                   "schedule_total_steps": steps, "batch_size": batch_size, "block_size": block_size,
                   "grad_accum": grad_accum}
    checkpoint_save(checkpoint_path, model, optimizer, steps, model_config, train_state)
    if log_path:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            for item in logs:
                f.write(json.dumps(item) + "\n")
    return {"checkpoint": str(checkpoint_path), "final_loss": logs[-1]["loss"], "steps": steps,
            "device": str(device), "duration_sec": time.monotonic()-start_time}