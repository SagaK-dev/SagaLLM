"""Small deterministic CPU proof of SFT mechanics on ORIGINAL Java fixtures.

This experiment trains a tiny randomly initialized model for only two steps.
It is not a code-capable model, and neither generated code nor test solutions
are executed. No network, GPUs, paid API or external dataset is used.
"""
from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from sagallm.config import ModelConfig
from sagallm.java_coding_eval import build_coding_fixture, verify_coding_fixture, evaluate_java_compilation
from sagallm.java_ai import LocalCheckpointGenerator
from sagallm.model import SagaTransformer
from sagallm.post_training import sft_train, format_sft_example
from sagallm.tokenization import ByteBPETokenizer
from sagallm.training import checkpoint_save, checkpoint_load


def _heldout_loss(path: Path, tok: ByteBPETokenizer, config: ModelConfig, checkpoint: Path) -> float:
    model = SagaTransformer(config).eval()
    checkpoint_load(checkpoint, model)
    losses = []
    with torch.inference_mode():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            x, y = format_sft_example(row["messages"], tok, config.max_position_embeddings)
            losses.append(float(model(x[None, :], labels=y[None, :]).loss))
    return sum(losses) / len(losses)


def run(output: str | Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="sagallm-java-sft-smoke-") as temporary:
        root = Path(temporary)
        dataset = root / "dataset"
        build_coding_fixture(dataset)
        assert verify_coding_fixture(dataset)["status"] == "PASS"
        # Learn tokenizer ONLY on train, not validation/test answers.
        train = [json.loads(line) for line in (dataset / "train.jsonl").read_text().splitlines()]
        tok = ByteBPETokenizer.train([m["content"] for item in train for m in item["messages"]],
                                     vocab_size=384)
        tok_file = root / "tokenizer.json"
        tok.save(tok_file)
        cfg = ModelConfig(vocab_size=384, hidden_size=64, num_layers=2, num_heads=4,
                          num_kv_heads=2, intermediate_size=128, max_position_embeddings=1024)
        torch.manual_seed(23)
        model = SagaTransformer(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        base = root / "random-base.pt"
        after = root / "tiny-java-sft.pt"
        checkpoint_save(base, model, optimizer, 0, cfg, {
            "stage": "random_initialization_only",
            "tokenizer_sha256": hashlib.sha256(tok_file.read_bytes()).hexdigest(),
        })
        before_loss = _heldout_loss(dataset / "test.jsonl", tok, cfg, base)
        train_result = sft_train(base, tok_file, dataset / "train.jsonl", after,
                                 steps=2, device_name="cpu", seed=23)
        after_loss = _heldout_loss(dataset / "test.jsonl", tok, cfg, after)
        generator = LocalCheckpointGenerator(after, tok_file)
        compile_result = evaluate_java_compilation(dataset, generator=generator,
                                                  split="test", max_examples=2,
                                                  max_new_tokens=80, attempts=1)
        report = {
            "status": "PASS_RUNTIME",
            "experiment": "tiny-random-base-java-sft-smoke",
            "test_family_split": ["calculator", "factorial"],
            "train_examples": 10,
            "sft_optimizer_steps": 2,
            "heldout_test_loss_before": before_loss,
            "heldout_test_loss_after": after_loss,
            "sft_final_training_loss": train_result["final_loss"],
            "model_parameters": cfg.parameter_count(),
            "eval_compile_passes": compile_result["compile_passes"],
            "eval_total": compile_result["examples"],
            "runtime_behavior": "NOT_RUN",
            "general_java_capability": "NOT_ESTABLISHED",
            "training_data": "local original synthetic fixtures only",
            "note": "Tiny 2-step random-base SFT is not proof of Java coding ability",
        }
        if output is not None:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="New JSON report; never overwrite")
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))