"""Re-evaluate pinned v0.18 base and best checkpoints on identical Java prompts.

Compile only; never execute generated Java. Does not retrain or select weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sagallm.java_diverse_corpus import verify_java_diverse_corpus
from sagallm.java_instruction_sft import (
    V017_SFT_BEST_SHA256, V016_TOKENIZER_SHA256, _validate_base,
    extend_rope_context, heldout_compilation, compile_heldout_references, read_split,
    restore_verified_base_weights, sha_file,
)
from sagallm.model import SagaTransformer
from sagallm.tokenization import load_tokenizer
from sagallm.training import checkpoint_load


def evaluate(base: Path, best: Path, tokenizer_file: Path, dataset: Path,
             *, cases: int = 8) -> dict:
    if not 1 <= cases <= 12:
        raise ValueError("Cases must be between 1 and 12")
    verify_java_diverse_corpus(dataset)
    state, original, base_sha, tokenizer_sha = _validate_base(
        base, tokenizer_file, require_pinned=True,
        pinned_checkpoint_sha256=V017_SFT_BEST_SHA256)
    if tokenizer_sha != V016_TOKENIZER_SHA256:
        raise ValueError("Unexpected tokenizer")
    if not best.is_file() or best.is_symlink() or best.stat().st_size > 90 * 1024 * 1024:
        raise ValueError("Missing or unsafe best checkpoint")
    tokenizer = load_tokenizer(tokenizer_file)
    config = extend_rope_context(original, 384)
    rows, counts = read_split(dataset, "test", tokenizer, 384,
                              original_context=original.max_position_embeddings)
    # Validate benchmark reference sources independently of generated output.
    # Do not use test references for checkpoint or decoding hyperparameter selection.
    references = compile_heldout_references(rows, max_cases=cases)
    model = SagaTransformer(config)
    model.eval()
    restore_verified_base_weights(model, state, original)
    baseline = heldout_compilation(model, tokenizer, rows, max_cases=cases)
    # checkpoint_load enforces config compatibility; reject unexpected weights.
    checkpoint_load(best, model)
    if model.config != config:
        raise ValueError("Best checkpoint has unexpected context")
    model.eval()
    selected = heldout_compilation(model, tokenizer, rows, max_cases=cases)
    return {
        "status": "PAIRED_COMPILE_ONLY",
        "baseline_sha256": base_sha,
        "best_sha256": sha_file(best),
        "tokenizer_sha256": tokenizer_sha,
        "test_split": counts,
        "reference_compilation": references,
        "base": baseline,
        "best": selected,
        "java_executed": False,
        "root_cause": "NOT_ESTABLISHED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--best", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse to overwrite evaluation evidence")
    result = evaluate(args.base, args.best, args.tokenizer, args.dataset, cases=args.cases)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"],
                      "base_compiled": result["base"]["compiled"],
                      "best_compiled": result["best"]["compiled"],
                      "root_cause": result["root_cause"]}))


if __name__ == "__main__":
    main()
