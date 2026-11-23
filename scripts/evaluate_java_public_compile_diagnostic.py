"""Public Java intake: exploratory compile-only measurement, NO human review.

This separate experiment does not bypass independent benchmark review gates.
Six public Exercism-derived task prompts are shortened and pre-registered
below to fit the tiny pinned SagaLLM model. This changes the original wording:
results are exploratory prompt/compile diagnostics, NOT verified task success.

Never run generated Java or select/promote model weights from these results.
Generated source and raw compiler diagnostics are ephemeral and NOT persisted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch

from sagallm.java_independent_benchmark import read_suite
from sagallm.java_instruction_sft import (
    V016_TOKENIZER_SHA256, V017_SFT_BEST_SHA256, _validate_base,
    classify_javac_diagnostic, classify_javac_symbol_kind, compact_prompt,
    extend_rope_context, restore_verified_base_weights, sha_file,
)
from sagallm.java_vibe import _compile
from sagallm.model import SagaTransformer
from sagallm.tokenization import load_tokenizer
from sagallm.training import checkpoint_load

PINNED_SUITE_SHA256 = "d1af64a934c304f04b2cfa822280c73cb7da7b64d09e83f62b071b945a244c74"
PINNED_V018_SHA256 = "9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc"
PINNED_CONTEXT = 384
MAX_OUTPUT_TOKENS = 170
GENERATION_SECONDS = 25

# Fixed before running model inference. These are assistant-authored abridged
# instructions derived from externally published tasks; NOT original prompts.
PROMPTS = {
    "ex_hamming": (
        "Read two equal-length DNA lines. Print the number of character positions "
        "that differ, followed by a newline."
    ),
    "ex_rna_transcription": (
        "Read one DNA line, possibly empty. Print its RNA complement using "
        "A->U, C->G, G->C, T->A, followed by a newline."
    ),
    "ex_luhn": (
        "Read a string of digits and spaces. Ignore spaces. Apply Luhn checksum "
        "and print true or false. Under two digits or other characters are invalid."
    ),
    "ex_roman_numerals": (
        "Read an integer from 1 to 3999. Print uppercase Roman numerals, "
        "including the subtractive forms IV, IX, XL, XC, CD, CM."
    ),
    "ex_allergies": (
        "Read an integer and an allergen name. Bits: eggs=1, peanuts=2, "
        "shellfish=4, strawberries=8, tomatoes=16, chocolate=32, pollen=64, "
        "cats=128. Print true if that bit is set, otherwise false."
    ),
    "ex_atbash_cipher": (
        "Read a line. Apply Atbash (a to z, b to y), lowercase the result, "
        "keep ASCII digits, discard punctuation and spaces, group output "
        "in blocks of five."
    ),
}


def fixed_public_tasks(suite_file: Path) -> tuple[list[str], str, int]:
    suite, source_sha, metadata = read_suite(suite_file)
    if source_sha != PINNED_SUITE_SHA256:
        raise ValueError("Public intake changed after experiment pre-registration")
    if (suite["benchmark_id"] != "exercism_io_intake_2026"
            or suite["origin"] != "EXTERNAL_SOURCE_INTAKE_UNREVIEWED"
            or suite["review_status"] != "PENDING_HUMAN_REVIEW"
            or suite["license_id"] != "NOT_REVIEWED"
            or suite["reviewers"] != []
            or metadata["review_gate"] != "REVIEW_REQUIRED"):
        raise ValueError("This is only for unreviewed public intake; cannot claim review")
    ids = [t["id"] for t in suite["tasks"]]
    if ids != list(PROMPTS) or len(ids) != 6 or metadata["io_cases"] != 27:
        raise ValueError("Unexpected task order or external intake coverage")
    if any(len(prompt) > 240 or "\n" in prompt
           for prompt in PROMPTS.values()):
        raise ValueError("Unregistered or invalid instruction framing")
    return ids, source_sha, metadata["io_cases"]


def score_task(model, tokenizer, task_id: str) -> dict:
    """Compile one model-generated Main.java in a temporary directory, never execute."""
    if task_id not in PROMPTS or model.config.max_position_embeddings != PINNED_CONTEXT:
        raise ValueError("Unregistered prompt or model context")
    prefix = [tokenizer.SPECIAL["<bos>"]] + tokenizer.encode(compact_prompt(PROMPTS[task_id]))
    remaining = PINNED_CONTEXT - len(prefix)
    record = {
        "id": task_id,
        "prefix_tokens": len(prefix),
        "prompt_sha256": hashlib.sha256(PROMPTS[task_id].encode("utf8")).hexdigest(),
        "status": "NOT_RUN",
        "reason": "NOT_STARTED",
        "source_saved": False,
        "java_executed": False,
    }
    if remaining < 2:
        record.update(reason="INSUFFICIENT_CONTEXT", status="SKIPPED")
        return record
    with torch.inference_mode():
        sequence = model.generate(
            torch.tensor([prefix], dtype=torch.long),
            max_new_tokens=min(remaining, MAX_OUTPUT_TOKENS),
            temperature=0.0,
            eos_token_id=tokenizer.SPECIAL["<eos>"],
            vocab_limit=tokenizer.vocab_size,
            deadline_monotonic=time.monotonic() + GENERATION_SECONDS,
        )
    generated_ids = sequence[0, len(prefix):].tolist()
    code = tokenizer.decode(generated_ids)
    record.update(
        output_sha256=hashlib.sha256(code.encode("utf8")).hexdigest(),
        generated_tokens=len(generated_ids),
        eos_seen=bool(generated_ids and generated_ids[-1] == tokenizer.SPECIAL["<eos>"]),
    )
    if (not code or len(code.encode("utf8")) > 4000 or "\x00" in code
            or "public class Main" not in code):
        record.update(status="FAIL", reason="INVALID_GENERATED_MAIN")
        return record
    with tempfile.TemporaryDirectory(prefix="sagallm-public-java-javac-") as location:
        folder = Path(location)
        main_file = folder / "Main.java"
        main_file.write_text(code, encoding="utf8")
        compilation = _compile(main_file, folder, timeout=12)
    status = compilation["status"]
    if status not in ("PASS", "FAIL", "NOT_RUN"):
        raise ValueError("Unexpected javac status")
    stderr = compilation.get("diagnostics", "")
    record.update(
        status=status,
        reason=("JAVAC_PASS" if status == "PASS" else
                "COMPILER_UNAVAILABLE" if status == "NOT_RUN" else "JAVAC_FAIL"),
        error_category=(classify_javac_diagnostic(stderr) if status == "FAIL" else "NONE"),
        symbol_kind=(classify_javac_symbol_kind(stderr) if status == "FAIL" else "NONE"),
    )
    return record


def run_experiment(v017: Path, v018: Path, tokenizer_file: Path, suite: Path) -> dict:
    ids, suite_sha, io_cases = fixed_public_tasks(suite)
    if (not v018.is_file() or v018.is_symlink()
            or v018.stat().st_size > 90 * 1024 * 1024
            or sha_file(v018) != PINNED_V018_SHA256):
        raise ValueError("Pinned v0.18 checkpoint missing or mismatched")
    base, original_cfg, base_sha, tok_sha = _validate_base(
        v017, tokenizer_file, require_pinned=True,
        pinned_checkpoint_sha256=V017_SFT_BEST_SHA256,
    )
    if tok_sha != V016_TOKENIZER_SHA256:
        raise ValueError("Pinned tokenizer identity mismatch")
    config = extend_rope_context(original_cfg, PINNED_CONTEXT)
    tokenizer = load_tokenizer(tokenizer_file)
    torch.set_num_threads(2)
    model = SagaTransformer(config)
    model.eval()
    restore_verified_base_weights(model, base, original_cfg)
    v017_report = [score_task(model, tokenizer, task_id) for task_id in ids]

    checkpoint_load(v018, model)
    if model.config != config:
        raise ValueError("v0.18 model context differs from pinned architecture")
    model.eval()
    v018_report = [score_task(model, tokenizer, task_id) for task_id in ids]

    def counts(records: list[dict]) -> dict:
        return {
            "evaluated": len(records),
            "compiled": sum(x["status"] == "PASS" for x in records),
            "failed": sum(x["status"] == "FAIL" for x in records),
            "unavailable_or_skipped": sum(
                x["status"] in ("NOT_RUN", "SKIPPED") for x in records),
            "cases": records,
        }
    return {
        "status": "EXPLORATORY_PUBLIC_COMPILE_ONLY_UNREVIEWED",
        "suite_sha256": suite_sha,
        "task_count": len(ids),
        "original_io_cases": io_cases,
        "io_cases_used_for_scoring": 0,
        "original_problem_prompts_used_verbatim": False,
        "prompt_framing": "PRE_REGISTERED_ABRIDGED_COMPACT_JAVA_SFT",
        "v017_checkpoint_sha256": base_sha,
        "v018_checkpoint_sha256": PINNED_V018_SHA256,
        "tokenizer_sha256": tok_sha,
        "model_context": config.max_position_embeddings,
        "generation": "GREEDY_TEMPERATURE_0_MAX_170",
        "v017": counts(v017_report),
        "v018": counts(v018_report),
        "human_reviews": 0,
        "source_license_review": "PENDING",
        "benchmark_independence_proven": False,
        "public_tasks_possible_training_exposure": True,
        "behavioral_status": "NOT_RUN",
        "java_executed": False,
        "compiler_annotation_processing_disabled": True,
        "checkpoint_selected_or_promoted": False,
        "general_java_capability_established": False,
        "model_selection_permitted": False,
        "raw_java_source_or_diagnostics_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v017", required=True, type=Path)
    parser.add_argument("--v018", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("Refuse to overwrite previously recorded measurements")
    report = run_experiment(args.v017, args.v018, args.tokenizer, args.suite)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf8")
    print(json.dumps({
        "status": report["status"],
        "v017_compiled": report["v017"]["compiled"],
        "v018_compiled": report["v018"]["compiled"],
        "tasks": report["task_count"],
        "behavioral_status": report["behavioral_status"],
        "model_selection_permitted": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
