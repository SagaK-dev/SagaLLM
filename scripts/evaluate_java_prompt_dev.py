"""Validation-first, compile-only Java prompt experiment for pinned v0.17/v0.18.

Pre-register exactly two decoding prompt styles, select only on the validation
split, then evaluate once on test case positions 8..15 (not the previously
repeated positions 0..7). Test families are still synthetic and previously
known to the project: this is NOT an independent real-world benchmark.
Never execute generated Java or change model weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sagallm.java_diverse_corpus import verify_java_diverse_corpus
from sagallm.java_instruction_sft import (
    JAVA_EVAL_PROMPT_STYLES, V016_TOKENIZER_SHA256, V017_SFT_BEST_SHA256,
    _validate_base, compile_heldout_references, extend_rope_context,
    heldout_compilation, read_split, restore_verified_base_weights, sha_file,
    select_java_eval_style,
)
from sagallm.model import SagaTransformer
from sagallm.tokenization import load_tokenizer
from sagallm.training import checkpoint_load

PINNED_V018_BEST_SHA256 = (
    "9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc"
)
FRESH_TEST_OFFSET = 8


def evaluate(base: Path, best: Path, tokenizer_file: Path, dataset: Path,
             *, dev_cases: int = 8, test_cases: int = 8) -> dict:
    if not 1 <= dev_cases <= 12 or not 1 <= test_cases <= 12:
        raise ValueError("Case count outside 1..12")
    if (not best.is_file() or best.is_symlink() or
            best.stat().st_size > 90 * 1024 * 1024 or
            sha_file(best) != PINNED_V018_BEST_SHA256):
        raise ValueError("v0.18 checkpoint differs from the pinned original")
    verify_java_diverse_corpus(dataset)
    state, original, base_sha, tok_sha = _validate_base(
        base, tokenizer_file, require_pinned=True,
        pinned_checkpoint_sha256=V017_SFT_BEST_SHA256)
    if tok_sha != V016_TOKENIZER_SHA256:
        raise ValueError("Unexpected tokenizer")

    tokenizer = load_tokenizer(tokenizer_file)
    config = extend_rope_context(original, 384)
    # Only validation rows are loaded/scored during the selection step.
    validation_rows, validation_counts = read_split(
        dataset, "validation", tokenizer, 384,
        original_context=original.max_position_embeddings)
    torch.set_num_threads(2)
    model = SagaTransformer(config)
    checkpoint_load(best, model)
    if model.config != config:
        raise ValueError("Unexpected v0.18 model context")
    model.eval()

    dev_results = {}
    # Order is fixed in code. On ties, canonical is retained.
    for style in JAVA_EVAL_PROMPT_STYLES:
        dev_results[style] = heldout_compilation(
            model, tokenizer, validation_rows, max_cases=dev_cases,
            prompt_style=style)
    selected_style = select_java_eval_style(
        {style: result["compiled"] for style, result in dev_results.items()})
    selected_on = "VALIDATION_COMPILE_ONLY"
    # Freeze selected style before reading or scoring any test examples.
    frozen_style = selected_style

    test_rows, test_counts = read_split(
        dataset, "test", tokenizer, 384,
        original_context=original.max_position_embeddings)
    if len(validation_rows) < dev_cases or len(test_rows) < FRESH_TEST_OFFSET + test_cases:
        raise ValueError("Too few accepted examples to compare full case windows")
    # Use the second seed-67 block, not the first 8 previously inspected cases.
    references = compile_heldout_references(
        test_rows, max_cases=test_cases, case_offset=FRESH_TEST_OFFSET)
    best_canonical = heldout_compilation(
        model, tokenizer, test_rows, max_cases=test_cases,
        prompt_style="canonical", case_offset=FRESH_TEST_OFFSET)
    if frozen_style == "canonical":
        best_selected = best_canonical
    else:
        best_selected = heldout_compilation(
            model, tokenizer, test_rows, max_cases=test_cases,
            prompt_style=frozen_style, case_offset=FRESH_TEST_OFFSET)
    restore_verified_base_weights(model, state, original)
    model.eval()
    base_canonical = heldout_compilation(
        model, tokenizer, test_rows, max_cases=test_cases,
        prompt_style="canonical", case_offset=FRESH_TEST_OFFSET)

    return {
        "status": "VALIDATION_SELECTED_COMPILE_ONLY",
        "base_checkpoint_sha256": base_sha,
        "best_checkpoint_sha256": PINNED_V018_BEST_SHA256,
        "tokenizer_sha256": tok_sha,
        "validation_counts": validation_counts,
        "validation_candidates": dev_results,
        "selection_metric": "validation_compiled_count",
        "tie_break": "canonical_first",
        "selection_used_only_validation": True,
        "selected_style": frozen_style,
        "selected_on": selected_on,
        "test_counts": test_counts,
        "test_offset": FRESH_TEST_OFFSET,
        "test_novelty_caveat": (
            "Different test examples from prior 8, but synthetic and from "
            "already known heldout families; NOT a real-world independent benchmark"
        ),
        "test_reference_compilation": references,
        "test_base_canonical": base_canonical,
        "test_best_canonical": best_canonical,
        "test_best_selected": best_selected,
        "java_executed": False,
        "model_retrained": False,
        "behavioral_correctness": "NOT_RUN",
        "improvement_established": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--best", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dev-cases", type=int, default=8)
    parser.add_argument("--test-cases", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse to overwrite existing evaluation evidence")
    report = evaluate(
        args.base, args.best, args.tokenizer, args.dataset,
        dev_cases=args.dev_cases, test_cases=args.test_cases)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    # Keep useful aggregate evidence in Actions logs without printing source.
    print(json.dumps({
        "status": report["status"],
        "validation_compiled": {
            style: item["compiled"]
            for style, item in report["validation_candidates"].items()
        },
        "selected_style": report["selected_style"],
        "test_references_compiled": report["test_reference_compilation"]["compiled"],
        "test_base_compiled": report["test_base_canonical"]["compiled"],
        "test_best_canonical_compiled": report["test_best_canonical"]["compiled"],
        "test_best_selected_compiled": report["test_best_selected"]["compiled"],
        "test_offset": report["test_offset"],
        "java_executed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
