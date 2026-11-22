"""Controlled train/validation-only A/B fine-tune for Java identifier variants.

Both arms start from the exact pinned v0.18 weights. Same training prompts,
example IDs, order, seed, batch size, optimizer, LR, steps and selection logic;
only the Java answer identifier spelling differs. Do not load, score or tune
on the test split. Do not execute generated Java. All examples are synthetic.

This is a small development experiment, NOT evidence of general Java ability.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import time

import torch

from sagallm.config import ModelConfig
from sagallm.java_diverse_corpus import NEW_TRAIN, new_case, verify_java_diverse_corpus
from sagallm.java_identifier_curriculum import (
    VARIABLES, verify_identifier_train_candidates,
)
from sagallm.java_instruction_sft import (
    V016_TOKENIZER_SHA256, encode_row, eval_sample, heldout_compilation,
    interleaved_order, pack, read_split, response_loss, sha_file, source_task,
)
from sagallm.java_large_corpus import TRAIN_FAMILIES, _case as old_case
from sagallm.model import SagaTransformer
from sagallm.tokenization import load_tokenizer
from sagallm.training import checkpoint_load, verify_checkpoint_tokenizer

PINNED_V018_SHA256 = "9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc"
PINNED_CANDIDATE_SHA256 = "ad18e0a328f00db2bca204c3911afd66307099e3d41ba0f0d97c964463d3c2d9"
SEED = 733
MAX_ARM_SECONDS = 240
STEPS = 60
BATCH_SIZE = 4
LEARNING_RATE = 4e-5
EVAL_AT = (30, 60)
VALIDATION_LOSS_EXAMPLES = 24
VALIDATION_COMPILE_EXAMPLES = 12
ARMS = ("canonical", "identifier_augmented")
CASE_ID = re.compile(r"-([0-9]{4})-v([0-9]+)$")


def baseline_row(augmented: dict) -> dict:
    """Recover original TRAIN-family template with the same prompt and ID."""
    family = augmented["family"]
    if family not in VARIABLES:
        raise ValueError("A heldout or unknown family was supplied")
    match = CASE_ID.search(augmented["id"])
    if match is None:
        raise ValueError("Invalid training candidate identifier")
    index = int(match.group(1))
    if not 0 <= index < 6:
        raise ValueError("Unexpected training candidate index")
    make = old_case if family in TRAIN_FAMILIES else new_case
    original_task_ja, original_task_en, original = make(family, index)
    task, candidate = source_task(augmented)
    expected_task = original_task_ja if index % 2 == 0 else original_task_en
    if task != expected_task or candidate == original:
        raise ValueError("Candidate prompt/source identity mismatch")
    result = copy.deepcopy(augmented)
    result["messages"][1]["content"] = original
    return result


def paired_rows(candidate_jsonl: Path, tokenizer, context: int) -> tuple[list[tuple[dict, dict]], dict]:
    if context != 384:
        raise ValueError("This experiment requires the pinned 384-token model")
    entries = [json.loads(line) for line in candidate_jsonl.read_text(encoding="utf8").splitlines()]
    canonical, augmented = [], []
    excluded = 0
    for row in entries:
        original = baseline_row(row)
        control = encode_row(original, tokenizer, context)
        alternative = encode_row(row, tokenizer, context)
        if control is None or alternative is None:
            excluded += 1
            continue
        if (control["prefix_ids"] != alternative["prefix_ids"] or
                control["id"] != alternative["id"] or
                control["family"] != alternative["family"]):
            raise ValueError("The A/B training prompts are not paired")
        canonical.append(control)
        augmented.append(alternative)
    pairs = list(zip(canonical, augmented))
    families = {left["family"] for left, _ in pairs}
    if len(pairs) < 24 or len(families) < 5:
        raise ValueError("Too few paired complete training examples")
    return pairs, {
        "candidates_total": len(entries),
        "pairs_admitted": len(pairs),
        "pairs_excluded_context": excluded,
        "train_families_admitted": len(families),
        "same_training_prompts_and_ids": True,
    }


def validation_gate(control: dict, augmented: dict) -> dict:
    """Pre-registered conservative gate on validation only; no test lookup."""
    for data in (control, augmented):
        if (data.get("validation_compiled_total") != VALIDATION_COMPILE_EXAMPLES or
                type(data.get("validation_compiled")) is not int or
                not 0 <= data["validation_compiled"] <= VALIDATION_COMPILE_EXAMPLES or
                not isinstance(data.get("best_validation_loss"), (float, int)) or
                not 0 < data["best_validation_loss"] < 100):
            raise ValueError("Missing or invalid validation metrics")
    compile_delta = augmented["validation_compiled"] - control["validation_compiled"]
    loss_ratio = augmented["best_validation_loss"] / control["best_validation_loss"]
    qualifies = compile_delta >= 2 and loss_ratio <= 1.02
    return {
        "status": ("CANDIDATE_WARRANTS_INDEPENDENT_EVALUATION"
                   if qualifies else "NO_VALIDATED_COMPILE_GAIN"),
        "compile_delta": compile_delta,
        "loss_ratio": round(loss_ratio, 6),
        "min_compile_gain_required": 2,
        "max_loss_ratio_allowed": 1.02,
        "production_adoption": False,
        "uses_test_data": False,
    }


def train_one_arm(arm: str, pairs: list[tuple[dict, dict]],
                  base: Path, config: ModelConfig, valid_rows: list[dict],
                  tokenizer, output: Path) -> dict:
    if arm not in ARMS:
        raise ValueError("Unregistered training arm")
    torch.manual_seed(SEED)
    model = SagaTransformer(config)
    state = checkpoint_load(base, model)
    model.eval()
    if state["config"] != config.to_dict():
        raise ValueError("Pinned checkpoint architecture mismatch")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    heldout_loss_rows = eval_sample(valid_rows, VALIDATION_LOSS_EXAMPLES, seed=47)
    first_loss = response_loss(model, heldout_loss_rows)
    best_loss = first_loss
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    side = ARMS.index(arm)
    training_rows = [pair[side] for pair in pairs]
    order = interleaved_order(training_rows, SEED)
    history = []
    seen = set()
    started = time.monotonic()
    for step in range(1, STEPS + 1):
        if time.monotonic() - started > MAX_ARM_SECONDS:
            raise TimeoutError(f"AB training arm exceeded fixed budget: {arm}")
        selected = []
        for j in range(BATCH_SIZE):
            index = order[((step - 1) * BATCH_SIZE + j) % len(order)]
            selected.append(training_rows[index])
            seen.add(index)
        x, y = pack(selected)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, labels=y).loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in EVAL_AT:
            current = response_loss(model, heldout_loss_rows)
            history.append({
                "step": step,
                "validation_response_loss": round(current, 8),
                "train_examples_seen": len(seen),
            })
            if current < best_loss - 0.001:
                best_loss = current
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    compile_result = heldout_compilation(
        model, tokenizer, valid_rows, max_cases=VALIDATION_COMPILE_EXAMPLES,
        prompt_style="canonical")
    if (compile_result.get("evaluated") != VALIDATION_COMPILE_EXAMPLES or
            compile_result.get("behavior") != "NOT_RUN"):
        raise RuntimeError("Incomplete or unsafe validation evaluation")
    checkpoint_file = output / f"{arm}-validation-best.pt"
    torch.save({
        "model": best_state, "config": config.to_dict(), "arm": arm,
        "selected_step": best_step, "validation_loss": best_loss,
        "base_checkpoint_sha256": PINNED_V018_SHA256,
        "training_seed": SEED,
        "training_steps_requested": STEPS,
    }, checkpoint_file)
    return {
        "arm": arm, "optimizer_steps_completed": STEPS,
        "examples_seen": len(seen), "initial_validation_loss": round(first_loss, 8),
        "best_validation_loss": round(best_loss, 8),
        "selected_step": best_step,
        "validation_compiled": compile_result["compiled"],
        "validation_compiled_total": compile_result["evaluated"],
        "validation_compile_only": compile_result,
        "history": history,
        "checkpoint_sha256": sha_file(checkpoint_file),
        "training_seconds": round(time.monotonic() - started, 2),
        "java_executed": False,
    }


def experiment(base: Path, tokenizer_file: Path, canonical_dir: Path,
               candidate_dir: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refuse to overwrite A/B evaluation evidence")
    for path, pinned in ((base, PINNED_V018_SHA256),
                         (tokenizer_file, V016_TOKENIZER_SHA256)):
        if (not path.is_file() or path.is_symlink() or
                path.stat().st_size > 90 * 1024 * 1024 or
                sha_file(path) != pinned):
            raise ValueError("Pinned model or tokenizer does not match")
    # Canonical regeneration checks split identities but does not use test rows
    # for model selection; no test split is decoded, scored, or loaded for SFT.
    verify_java_diverse_corpus(canonical_dir)
    candidate_meta = verify_identifier_train_candidates(candidate_dir)
    if candidate_meta["candidate_sha256"] != PINNED_CANDIDATE_SHA256:
        raise ValueError("Unexpected candidate data identity")
    checkpoint_meta = torch.load(base, map_location="cpu", weights_only=True)
    verify_checkpoint_tokenizer(checkpoint_meta, tokenizer_file)
    config = ModelConfig(**checkpoint_meta["config"])
    if config.max_position_embeddings != 384 or config.parameter_count() > 3_000_000:
        raise ValueError("Unexpected model context or parameter count")
    tokenizer = load_tokenizer(tokenizer_file)
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError("Tokenizer exceeds model vocabulary")
    train_pairs, pairing = paired_rows(
        candidate_dir / candidate_meta["candidate_file"],
        tokenizer, config.max_position_embeddings)
    valid_rows, valid_counts = read_split(
        canonical_dir, "validation", tokenizer, config.max_position_embeddings)
    if len(valid_rows) < VALIDATION_LOSS_EXAMPLES or len(valid_rows) < VALIDATION_COMPILE_EXAMPLES:
        raise ValueError("Insufficient validation examples")
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    results = {}
    for arm in ARMS:
        results[arm] = train_one_arm(
            arm, train_pairs, base, config, valid_rows, tokenizer, output)
    decision = validation_gate(results["canonical"], results["identifier_augmented"])
    report = {
        "status": "TRAIN_VALIDATION_AB_COMPLETE",
        "model_origin": "PINNED_V018_BEST",
        "base_checkpoint_sha256": PINNED_V018_SHA256,
        "tokenizer_sha256": V016_TOKENIZER_SHA256,
        "candidate_sha256": PINNED_CANDIDATE_SHA256,
        "canonical_manifest_sha256": sha_file(canonical_dir / "manifest.json"),
        "dataset_source": "project-authored synthetic train families only",
        "arms": list(ARMS),
        "seed": SEED, "steps_per_arm": STEPS, "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE, "weight_decay": 0.01,
        "selection_checkpoints": [0, *EVAL_AT],
        "selection_uses_only_validation": True,
        "training_pairs": pairing,
        "validation": valid_counts,
        "results": results,
        "decision": decision,
        "test_split_evaluated": False,
        "test_used_for_selection": False,
        "independent_real_world_benchmark": False,
        "model_deployed": False,
        "generated_java_executed": False,
        "behavioral_correctness": "NOT_RUN",
        "generalization_improvement_established": False,
    }
    with (output / "ab-report.json").open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = experiment(args.base, args.tokenizer, args.canonical,
                        args.candidates, args.output)
    print(json.dumps({
        "status": result["status"],
        "training_pairs": result["training_pairs"],
        "validation_compiled": {
            arm: result["results"][arm]["validation_compiled"]
            for arm in ARMS
        },
        "selected_step": {
            arm: result["results"][arm]["selected_step"]
            for arm in ARMS
        },
        "decision": result["decision"],
        "test_split_evaluated": False,
        "generated_java_executed": False,
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
