"""Pre-registered train/validation-only Java broad-mixture, low-LR A/B.

Same pinned v0.18 weights, 80 optimizer steps, 1e-5 LR and exact batches.
Each batch has 3 canonical examples from the full 90-family TRAIN corpus
plus one paired example. The ONLY difference is the paired answer spelling:
canonical identifiers (control) vs verified renamed identifiers (treatment).

The test split is not tokenized, scored or used for model selection.
All Java generation diagnostics compile only; never execute Java.
All sources are project-authored synthetic exercises, not independent tasks.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from sagallm.config import ModelConfig
from sagallm.java_diverse_corpus import NEW_TRAIN, verify_java_diverse_corpus
from sagallm.java_identifier_curriculum import verify_identifier_train_candidates
from sagallm.java_instruction_sft import (
    V016_TOKENIZER_SHA256, eval_sample, heldout_compilation, interleaved_order,
    pack, read_split, response_loss, sha_file,
)
from sagallm.java_large_corpus import TRAIN_FAMILIES
from sagallm.model import SagaTransformer
from sagallm.tokenization import load_tokenizer
from sagallm.training import checkpoint_load, verify_checkpoint_tokenizer
from scripts.train_java_identifier_ab import (
    ARMS, PINNED_V018_SHA256, PINNED_CANDIDATE_SHA256,
    VALIDATION_COMPILE_EXAMPLES, VALIDATION_LOSS_EXAMPLES,
    paired_rows, validation_gate,
)

EXPERIMENT = "JAVA_BROAD_MIXTURE_LOW_LR_AB_V1"
SEED = 812
STEPS = 80
BATCH_SIZE = 4
BROAD_PER_BATCH = 3
PAIRED_PER_BATCH = 1
LR = 1e-5
EVAL_AT = (40, 80)
MAX_ARM_SECONDS = 240
REQUIRED_CANONICAL_FAMILIES = 80
REQUIRED_CANONICAL_ROWS = 1000


def fixed_training_schedule(broad_rows: list[dict],
                            pairs: list[tuple[dict, dict]],
                            *, steps: int = STEPS) -> list[tuple[tuple[str, int], ...]]:
    """Equal, pre-registered schedule; no validation/test data can enter."""
    if (type(steps) is not int or not 1 <= steps <= STEPS or
            not pairs or len(broad_rows) < REQUIRED_CANONICAL_ROWS or
            len({r["family"] for r in broad_rows}) < REQUIRED_CANONICAL_FAMILIES):
        raise ValueError("Insufficient broad TRAIN-family evidence")
    train_families = set(TRAIN_FAMILIES) | set(NEW_TRAIN)
    if any(row["family"] not in train_families for row in broad_rows):
        raise ValueError("Non-TRAIN task family in broad training schedule")
    if len(pairs) < 100 or any(
            left["id"] != right["id"] or
            left["prefix_ids"] != right["prefix_ids"] or
            left["family"] != right["family"] or
            left["family"] not in train_families
            for left, right in pairs):
        raise ValueError("Unpaired or undersized identifier TRAIN samples")
    broad_order = interleaved_order(broad_rows, SEED)
    paired_order = interleaved_order([a for a, _ in pairs], SEED + 1)
    needed_broad = steps * BROAD_PER_BATCH
    needed_pairs = steps * PAIRED_PER_BATCH
    if needed_broad > len(broad_order) or needed_pairs > len(paired_order):
        raise ValueError("The fixed schedule would repeat examples")
    schedule = []
    for step in range(steps):
        candidates = tuple(
            ("broad", index)
            for index in broad_order[
                step * BROAD_PER_BATCH:(step + 1) * BROAD_PER_BATCH]
        )
        intervention = ("paired", paired_order[step])
        schedule.append(candidates + (intervention,))
    return schedule


def schedule_summary(schedule: list[tuple[tuple[str, int], ...]],
                     broad_rows: list[dict],
                     paired_rows_data: list[tuple[dict, dict]]) -> dict:
    broad = [index for step in schedule for kind, index in step if kind == "broad"]
    paired = [index for step in schedule for kind, index in step if kind == "paired"]
    if (len(set(broad)) != len(broad) or len(set(paired)) != len(paired) or
            len(broad) != len(schedule) * BROAD_PER_BATCH or
            len(paired) != len(schedule) * PAIRED_PER_BATCH):
        raise ValueError("Schedule repeats examples or violates mixture ratio")
    families = {broad_rows[i]["family"] for i in broad}
    families.update(paired_rows_data[i][0]["family"] for i in paired)
    return {
        "total_train_exposures": len(broad) + len(paired),
        "canonical_broad_exposures": len(broad),
        "paired_exposures": len(paired),
        "broad_families_seen": len({broad_rows[i]["family"] for i in broad}),
        "overall_train_families_seen": len(families),
        "identical_example_order_and_prompts": True,
        "no_training_example_repeated_within_source": True,
        "paired_ratio": "1/4",
        "test_examples_in_training": False,
    }


def fit_arm(arm: str, base: Path, config: ModelConfig,
            broad: list[dict], pairs: list[tuple[dict, dict]],
            schedule: list[tuple[tuple[str, int], ...]],
            valid: list[dict], tokenizer, output: Path) -> dict:
    if arm not in ARMS:
        raise ValueError("Unregistered A/B arm")
    torch.manual_seed(SEED)
    model = SagaTransformer(config)
    checkpoint_load(base, model)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    validation_loss_rows = eval_sample(valid, VALIDATION_LOSS_EXAMPLES, seed=47)
    initial_loss = response_loss(model, validation_loss_rows)
    best_loss, selected_step = initial_loss, 0
    best_weights = copy.deepcopy(model.state_dict())
    side = ARMS.index(arm)
    history = [{"step": 0, "validation_response_loss": round(initial_loss, 8)}]
    training_started = time.monotonic()
    for step, indices in enumerate(schedule, start=1):
        if time.monotonic() - training_started > MAX_ARM_SECONDS:
            raise TimeoutError("Training exceeded fixed CPU budget")
        items = [broad[index] if kind == "broad" else pairs[index][side]
                 for kind, index in indices]
        if len(items) != BATCH_SIZE:
            raise ValueError("Invalid fixed batch")
        x, y = pack(items)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, labels=y).loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in EVAL_AT:
            current = response_loss(model, validation_loss_rows)
            history.append({"step": step,
                            "validation_response_loss": round(current, 8)})
            # Same loss-based checkpoint selection as prior experiment.
            if current < best_loss - 0.001:
                best_loss = current
                selected_step = step
                best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    model.eval()
    compile_report = heldout_compilation(
        model, tokenizer, valid, max_cases=VALIDATION_COMPILE_EXAMPLES,
        prompt_style="canonical")
    if (compile_report.get("evaluated") != VALIDATION_COMPILE_EXAMPLES or
            compile_report.get("behavior") != "NOT_RUN"):
        raise ValueError("Unexpected compile-only validation coverage")
    checkpoint = output / f"{arm}-validation-best.pt"
    torch.save({
        "model": best_weights, "config": config.to_dict(), "arm": arm,
        "selected_step": selected_step,
        "validation_response_loss": best_loss,
        "origin_sha256": PINNED_V018_SHA256, "seed": SEED,
        "lr": LR, "steps_requested": STEPS,
        "experiment": EXPERIMENT,
    }, checkpoint)
    return {
        "arm": arm, "optimizer_steps_completed": STEPS,
        "initial_validation_loss": round(initial_loss, 8),
        "best_validation_loss": round(best_loss, 8),
        "selected_step": selected_step,
        "validation_compiled": compile_report["compiled"],
        "validation_compiled_total": compile_report["evaluated"],
        "validation_compile_only": compile_report,
        "history": history,
        "checkpoint_sha256": sha_file(checkpoint),
        "training_seconds": round(time.monotonic()-training_started, 2),
        "java_executed": False,
    }


def experiment(base: Path, tokenizer_path: Path, canonical: Path,
               candidates: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refuse to replace experimental evidence")
    for path, required in (
            (base, PINNED_V018_SHA256),
            (tokenizer_path, V016_TOKENIZER_SHA256)):
        if (not path.is_file() or path.is_symlink() or
                path.stat().st_size > 90 * 1024 * 1024 or
                sha_file(path) != required):
            raise ValueError("Unpinned or unsafe input model/tokenizer")
    verify_java_diverse_corpus(canonical)
    candidate_manifest = verify_identifier_train_candidates(candidates)
    if candidate_manifest["candidate_sha256"] != PINNED_CANDIDATE_SHA256:
        raise ValueError("TRAIN identifier candidate identity mismatch")
    state = torch.load(base, map_location="cpu", weights_only=True)
    verify_checkpoint_tokenizer(state, tokenizer_path)
    config = ModelConfig(**state["config"])
    if config.max_position_embeddings != 384 or config.parameter_count() > 3_000_000:
        raise ValueError("Unexpected model configuration")
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError("Token vocabulary mismatch")
    # No call to read_split(..., 'test') anywhere in this experiment.
    broad, broad_counts = read_split(
        canonical, "train", tokenizer, config.max_position_embeddings)
    valid, valid_counts = read_split(
        canonical, "validation", tokenizer, config.max_position_embeddings)
    pairs, pair_counts = paired_rows(
        candidates / candidate_manifest["candidate_file"],
        tokenizer, config.max_position_embeddings)
    if (len(valid) < VALIDATION_LOSS_EXAMPLES or
            len(valid) < VALIDATION_COMPILE_EXAMPLES or
            len(broad) < REQUIRED_CANONICAL_ROWS or
            len({r["family"] for r in broad}) < REQUIRED_CANONICAL_FAMILIES):
        raise ValueError("Insufficient full-family data after context filtering")
    train_families = {r["family"] for r in broad}
    validation_families = {r["family"] for r in valid}
    if (train_families & validation_families or
            any(pair[0]["family"] in validation_families for pair in pairs)):
        raise ValueError("Validation family leaked into training")
    schedule = fixed_training_schedule(broad, pairs)
    schedule_info = schedule_summary(schedule, broad, pairs)
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    results = {}
    for arm in ARMS:
        results[arm] = fit_arm(
            arm, base, config, broad, pairs, schedule, valid, tokenizer, output)
    decision = validation_gate(results["canonical"], results["identifier_augmented"])
    report = {
        "status": EXPERIMENT,
        "origin_model_sha256": PINNED_V018_SHA256,
        "tokenizer_sha256": V016_TOKENIZER_SHA256,
        "canonical_manifest_sha256": sha_file(canonical/"manifest.json"),
        "candidate_corpus_sha256": PINNED_CANDIDATE_SHA256,
        "original_training_counts": broad_counts,
        "paired_training_counts": pair_counts,
        "validation_counts": valid_counts,
        "schedule": schedule_info,
        "seed": SEED,
        "steps_per_arm": STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "weight_decay": 0.01,
        "loss_checkpoint_steps": [0, *EVAL_AT],
        "selection_uses_validation_only": True,
        "results": results,
        "decision": decision,
        "test_split_read_or_scored": False,
        "human_authored_independent_tasks": False,
        "Java_programs_executed": False,
        "behavioral_correctness": "NOT_RUN",
        "production_model_unchanged": True,
        "performance_improvement_established": False,
    }
    (output/"ab-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = experiment(
        args.base, args.tokenizer, args.canonical,
        args.candidates, args.output)
    print(json.dumps({
        "status": report["status"],
        "schedule": report["schedule"],
        "selected_checkpoint_steps": {
            arm: report["results"][arm]["selected_step"] for arm in ARMS
        },
        "validation_compile_success": {
            arm: report["results"][arm]["validation_compiled"] for arm in ARMS
        },
        "best_validation_response_loss": {
            arm: report["results"][arm]["best_validation_loss"] for arm in ARMS
        },
        "decision": report["decision"],
        "test_split_read_or_scored": False,
        "production_model_unchanged": True,
    }, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
