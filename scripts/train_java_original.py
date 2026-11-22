"""Bounded, offline training of SagaLLM on original bilingual Java examples.

Random initialization: no external pretrained weights, APIs, models, or datasets.
This CPU-scale experiment is deliberately not marketed as a capable Java LLM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

import torch

from sagallm.config import ModelConfig
from sagallm.java_generated_data import verify_java_dataset
from sagallm.java_ai import LocalCheckpointGenerator, extract_java
from sagallm.model import SagaTransformer
from sagallm.post_training import format_sft_example
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_examples(path: Path) -> list[dict]:
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]


def avg_loss(model: SagaTransformer, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    was_train = model.training
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for x, y in pairs:
            total += float(model(x[None, :], labels=y[None, :]).loss.item())
    model.train(was_train)
    return total / len(pairs)


def compile_check(dataset_path: Path, ckpt: Path, tok_file: Path, count: int = 15) -> dict:
    """Judge compilation only; never run generated Java programs."""
    from sagallm.java_generated_data import TEST_FAMILIES
    import shutil

    if shutil.which("javac") is None:
        return {"status": "NOT_RUN", "reason": "javac not installed", "attempted": 0, "compiled": 0}
    model = LocalCheckpointGenerator(ckpt, tok_file)
    cases = read_examples(dataset_path / "test.jsonl")
    # Round-robin sampling covers every held-out task family even for small budgets.
    grouped = {family: [row for row in cases if row["family"] == family] for family in TEST_FAMILIES}
    selected = [row for i in range(max(len(v) for v in grouped.values()))
                for family in TEST_FAMILIES for row in grouped[family][i:i+1]]
    compiled = 0
    exact_matches = 0
    records = []
    for row in selected[:count]:
        task_id = row["id"]
        exact = False
        # Exact SFT prefix, not the response or held-out reference solution.
        prompt = "User: " + row["messages"][0]["content"] + "\nAssistant: "
        try:
            generated = model(prompt, max_new_tokens=170)
            code = extract_java(generated, "Main")
            exact = code.strip() == row["messages"][1]["content"].strip()
            with tempfile.TemporaryDirectory(prefix="sagallm-java-compile-") as temp:
                source = Path(temp) / "Main.java"
                source.write_text(code, encoding="utf-8")
                proc = subprocess.run(["javac", "-proc:none", "-d", temp, str(source)],
                                      cwd=temp, capture_output=True, text=True, timeout=15, check=False)
                success = proc.returncode == 0
        except (ValueError, TimeoutError, subprocess.TimeoutExpired) as err:
            success = False
        compiled += int(success)
        exact_matches += int(exact)
        records.append({"id": task_id, "family": row["family"], "compiled": success,
                        "exact_reference_match": exact})
    return {"status": "PASS_RUNTIME", "attempted": len(records), "compiled": compiled,
            "exact_reference_matches": exact_matches, "items": records,
            "program_execution": "NOT_RUN", "semantic_correctness": "NOT_ESTABLISHED"}


def train(dataset_path: Path, output: Path, *, steps: int = 400, max_seconds: float = 180.0,
          seed: int = 20261122, lr: float = 0.0012, hidden_size: int = 160, layers: int = 3,
          progress_every: int = 50, compile_examples: int = 15) -> dict:
    if (not 1 <= steps <= 3000 or not 15 <= max_seconds <= 1800 or lr <= 0 or
        hidden_size not in (96,128,160,192) or layers not in (2,3,4) or
        not 0 <= compile_examples <= 15):
        raise ValueError("Unsafe or unsupported training budget")
    if output.exists():
        raise FileExistsError("Training output directory already exists")
    audit = verify_java_dataset(dataset_path)
    torch.set_num_threads(min(2, max(1, os.cpu_count() or 1)))
    torch.manual_seed(seed)
    random.seed(seed)
    data = {split: read_examples(dataset_path / f"{split}.jsonl") for split in ("train", "validation", "test")}
    tok = ByteBPETokenizer.train((m["content"] for row in data["train"] for m in row["messages"]),
                                 vocab_size=512, min_frequency=2)
    config = ModelConfig(vocab_size=512, hidden_size=hidden_size, num_layers=layers,
                         num_heads=4, num_kv_heads=2, intermediate_size=hidden_size*2,
                         max_position_embeddings=384)
    if config.parameter_count() > 2_000_000:
        raise ValueError("Experimental model would exceed 2 million parameters")
    pairs = {name: [format_sft_example(row["messages"], tok, config.max_position_embeddings)
                    for row in rows] for name, rows in data.items()}
    output.mkdir(parents=True)
    try:
        tok_file = output / "tokenizer.json"
        tok.save(tok_file)
        model = SagaTransformer(config).train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        meta = {
            "stage": "original_java_sft_random_initialization", "seed": seed,
            "tokenizer_sha256": sha256(tok_file),
            "training_dataset_sha256": sha256(dataset_path / "train.jsonl"),
            "validation_dataset_sha256": sha256(dataset_path / "validation.jsonl"),
            "test_dataset_sha256": sha256(dataset_path / "test.jsonl"),
            "dataset_manifest_sha256": sha256(dataset_path / "manifest.json"),
            "optimizer": "AdamW", "learning_rate": lr,
        }
        initial_checkpoint = output / "random-init.pt"
        checkpoint_save(initial_checkpoint, model, opt, 0, config, meta)
        metrics = {"step": [0], "train_loss": [avg_loss(model, pairs["train"][:24])],
                   "validation_loss": [avg_loss(model, pairs["validation"])],
                   "test_loss": [avg_loss(model, pairs["test"])]}
        # For reproducible training all batches are sampled from TRAIN ONLY.
        order = list(range(len(pairs["train"])))
        rng = random.Random(seed)
        rng.shuffle(order)
        index = 0
        completed = 0
        loss_latest = None
        start = time.monotonic()
        for step in range(1, steps+1):
            if time.monotonic()-start >= max_seconds:
                break
            if index >= len(order):
                rng.shuffle(order)
                index = 0
            x, y = pairs["train"][order[index]]
            index += 1
            opt.zero_grad(set_to_none=True)
            output_logits = model(x[None,:], labels=y[None,:])
            loss = output_logits.loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            completed = step
            loss_latest = float(loss.detach())
            if step % progress_every == 0 or step == steps:
                val = avg_loss(model, pairs["validation"])
                print(f"step={step} last_train_loss={loss_latest:.4f} validation_loss={val:.4f} elapsed={time.monotonic()-start:.1f}s", flush=True)
                metrics["step"].append(step)
                metrics["train_loss"].append(avg_loss(model, pairs["train"][:24]))
                metrics["validation_loss"].append(val)
                metrics["test_loss"].append(None)  # test is held out until final training
        final_checkpoint = output / "java-trained.pt"
        checkpoint_save(final_checkpoint, model, opt, completed, config, {
            **meta, "observed_steps": completed, "train_example_count": len(pairs["train"]),
        })
        final_train = avg_loss(model, pairs["train"][:24])
        final_validation = avg_loss(model, pairs["validation"])
        final_test = avg_loss(model, pairs["test"])
        metrics["step"].append(completed)
        metrics["train_loss"].append(final_train)
        metrics["validation_loss"].append(final_validation)
        metrics["test_loss"].append(final_test)
        del model, opt
        evaluation = (compile_check(dataset_path, final_checkpoint, tok_file, count=compile_examples)
                      if compile_examples else {"status": "NOT_RUN", "reason": "compile_examples=0"})
        report = {
            "status": "TRAINED_CPU", "base_weights": "RANDOM_INIT_ONLY",
            "pretrained_weight_downloads": False, "dataset_origin": "original_deterministic_java_examples",
            "train_examples": len(pairs["train"]), "validation_examples": len(pairs["validation"]),
            "test_examples": len(pairs["test"]), "disjoint_task_families": audit["family_disjoint"],
            "parameter_count": config.parameter_count(), "vocab_size": tok.vocab_size,
            "steps_requested": steps, "steps_completed": completed, "training_seconds": round(time.monotonic()-start,2),
            "train_loss_before": metrics["train_loss"][0], "train_loss_after": final_train,
            "validation_loss_before": metrics["validation_loss"][0], "validation_loss_after": final_validation,
            "test_loss_before": metrics["test_loss"][0], "test_loss_after": final_test,
            "history": metrics, "compile_only_heldout": evaluation,
            "optimizer": "AdamW", "learning_rate": lr, "seed": seed,
            "model_config": config.to_dict(), "checkpoint": "java-trained.pt",
            "checkpoint_sha256": sha256(final_checkpoint), "tokenizer": "tokenizer.json",
            "tokenizer_sha256": sha256(tok_file), "dataset_manifest_sha256": meta["dataset_manifest_sha256"],
            "java_runtime_tests": "NOT_RUN", "independent_real_world_java_ability": "NOT_ESTABLISHED",
            "license_statement": "Original generated code; no third-party Java corpus. No legal attestation.",
        }
        (output / "train-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        return report
    except BaseException:
        # Preserve explicit failures rather than silently claiming success.
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--hidden-size", type=int, default=160)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--compile-examples", type=int, default=15)
    options = parser.parse_args()
    result = train(options.dataset, options.output, steps=options.steps,
                   max_seconds=options.max_seconds, hidden_size=options.hidden_size,
                   layers=options.layers, progress_every=options.progress_every,
                   compile_examples=options.compile_examples)
    print(json.dumps({k: v for k,v in result.items() if k not in ("model_config","history")}, ensure_ascii=False, indent=2))