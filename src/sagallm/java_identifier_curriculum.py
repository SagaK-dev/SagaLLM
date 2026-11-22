"""Deterministic candidate identifier-diversity augmentation for TRAIN families only.

These are project-authored synthetic training candidates, not curated human
code, not independent tasks, and not evidence of better model performance.
The canonical v0.15/v0.17/v0.18 datasets and pinned checkpoints stay unchanged.
No generated Java model output is used or executed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .java_diverse_corpus import NEW_TRAIN, NEW_TEST, NEW_VALIDATION, new_case
from .java_large_corpus import (
    TRAIN_FAMILIES, TEST_FAMILIES, VALIDATION_FAMILIES, _case as old_case,
)
from .java_ai import _model_user_message
from .java_coding_eval import BASE_SOURCE

SCHEMA = "sagallm-identifier-train-candidates-v1"
SEED = 20261122
MAX_CASES_PER_FAMILY = 16
MAX_VARIANTS = 3
MAX_FILE_BYTES = 1_000_000
# This is an explicit train-only allowlist. No heldout family may appear.
VARIABLES = {
    "add": ("x", "y"),
    "subtract": ("x", "y"),
    "multiply": ("x", "y"),
    "count_divisors": ("v", "count", "k"),
    "sum_squares": ("total", "k"),
    "triangular_number": ("v",),
    "reverse_digits": ("v", "out"),
    "collatz_steps": ("v", "steps"),
    "array_count_greater": ("a", "count", "x"),
    "array_running_max": ("a", "best", "x"),
}
# Readable local identifiers; fixed, non-keyword vocabulary with varied lengths.
NAMES = (
    "firstOperand", "secondOperand", "inputNumber", "nextNumber",
    "runningTotal", "sumValue", "scanIndex", "loopIndex",
    "currentValue", "candidateValue", "outputValue", "resultNumber",
    "itemCount", "matchCount", "stepCounter", "remainingSteps",
    "arrayValues", "sourceNumbers", "runningMaximum", "largestValue",
    "workingValue", "digitValue", "numberValue", "temporaryValue",
    "accumulator", "positionIndex", "trackedTotal", "chosenValue",
)
# Match comments/literals before identifiers, so names inside them are not changed.
JAVA_TOKENS = re.compile(
    r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|'
    r"'(?:\\.|[^'\\])*'|[A-Za-z_$][A-Za-z0-9_$]*|.",
    re.DOTALL,
)
IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*\Z")


def _raw_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def check_train_family_boundary() -> None:
    known_train = set(TRAIN_FAMILIES) | set(NEW_TRAIN)
    heldout = (set(VALIDATION_FAMILIES) | set(TEST_FAMILIES) |
               set(NEW_VALIDATION) | set(NEW_TEST))
    if not VARIABLES or not set(VARIABLES) <= known_train:
        raise ValueError("Identifier augmentation includes a non-train family")
    if set(VARIABLES) & heldout:
        raise ValueError("Identifier augmentation crosses the heldout boundary")


def _new_name(family: str, index: int, variant: int,
              identifier: str, used: set[str]) -> str:
    key = f"{SEED}/{family}/{index}/{variant}/{identifier}".encode("utf8")
    start = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % len(NAMES)
    for offset in range(len(NAMES)):
        name = NAMES[(start + offset) % len(NAMES)]
        if name not in used:
            used.add(name)
            return name
    raise ValueError("Insufficient unique variable names")


def rename_identifiers(source: str, mapping: dict[str, str]) -> str:
    """Replace only Java identifier tokens, excluding comments and literals.

    This narrow lexical operation does NOT prove semantic equivalence.
    Callers must use known train-only templates and compile-check results.
    """
    if not isinstance(source, str) or len(source) > 4000:
        raise ValueError("Unexpected Java source length")
    if not mapping or any(not IDENTIFIER.fullmatch(k) or
                          not IDENTIFIER.fullmatch(v)
                          for k, v in mapping.items()):
        raise ValueError("Invalid identifier replacement")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Replacement identifiers are not distinct")
    if '"""' in source:
        raise ValueError("Unsupported Java text block")
    # Refuse to replace with any already existing identifier.
    existing = {token for token in JAVA_TOKENS.findall(source)
                if IDENTIFIER.fullmatch(token)}
    if set(mapping.values()) & existing:
        raise ValueError("Replacement identifier already exists in source")
    counts = Counter()
    parts = []
    for token in JAVA_TOKENS.findall(source):
        if token in mapping:
            counts[token] += 1
            parts.append(mapping[token])
        else:
            parts.append(token)
    if any(counts[key] < 2 for key in mapping):
        raise ValueError("Expected both declaration and use of every identifier")
    return "".join(parts)


def expected_candidate_rows(*, cases_per_family: int = 6,
                            variants: int = 2) -> list[dict]:
    check_train_family_boundary()
    if (type(cases_per_family) is not int or
        not 1 <= cases_per_family <= MAX_CASES_PER_FAMILY or
        type(variants) is not int or not 1 <= variants <= MAX_VARIANTS):
        raise ValueError("Unsafe candidate corpus size")
    rows = []
    for family, identifiers in sorted(VARIABLES.items()):
        make_case = old_case if family in TRAIN_FAMILIES else new_case
        for index in range(cases_per_family):
            jp, en, original = make_case(family, index)
            original_tokens = {part for part in JAVA_TOKENS.findall(original)
                               if IDENTIFIER.fullmatch(part)}
            if any(token not in original_tokens for token in identifiers):
                raise ValueError("Training family source template changed")
            for variant in range(variants):
                occupied = set(original_tokens)
                mapping = {name: _new_name(family, index, variant, name, occupied)
                           for name in identifiers}
                rewritten = rename_identifiers(original, mapping)
                if rewritten == original or "public class Main" not in rewritten:
                    raise ValueError("Identifier rename was not applied")
                task = jp if index % 2 == 0 else en
                rows.append({
                    "id": f"train-id-{family}-{index:04d}-v{variant}",
                    "family": family,
                    "messages": [
                        {"role": "user",
                         "content": _model_user_message(task, BASE_SOURCE, "Main")},
                        {"role": "assistant", "content": rewritten},
                    ],
                })
    return rows


def build_identifier_train_candidates(
    folder: str | Path, *, cases_per_family: int = 6,
    variants: int = 2,
) -> dict:
    """Create a separate candidate TRAIN-only dataset; never edit canonical data."""
    root = Path(folder)
    if root.exists() or root.is_symlink():
        raise FileExistsError("Refuse to overwrite an existing candidate corpus")
    rows = expected_candidate_rows(
        cases_per_family=cases_per_family, variants=variants)
    raw = ("\n".join(_raw_json(row) for row in rows) + "\n").encode("utf8")
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("Candidate dataset too large")
    identifiers = set()
    for family, names in VARIABLES.items():
        identifiers.update(names)
    info = {
        "schema": SCHEMA,
        "seed": SEED,
        "cases_per_family": cases_per_family,
        "variants": variants,
        "examples": len(rows),
        "families": sorted(VARIABLES),
        "train_only": True,
        "canonical_data_modified": False,
        "model_training_performed": False,
        "java_executed": False,
        "human_authored_independent_tasks": False,
        "derived_from": "project-authored synthetic train-family templates",
        "candidate_file": "train-identifier-variants.jsonl",
        "candidate_sha256": _sha256(raw),
    }
    root.mkdir(parents=True)
    try:
        (root / info["candidate_file"]).write_bytes(raw)
        (root / "manifest.json").write_text(
            _raw_json(info) + "\n", encoding="utf8")
    except BaseException:
        import shutil
        shutil.rmtree(root)
        raise
    return info


def verify_identifier_train_candidates(folder: str | Path) -> dict:
    root = Path(folder)
    manifest = root / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink() or manifest.stat().st_size > 8192:
        raise ValueError("Missing or unsafe identifier candidate manifest")
    meta = json.loads(manifest.read_text(encoding="utf8"))
    if not isinstance(meta, dict) or meta.get("schema") != SCHEMA:
        raise ValueError("Incorrect candidate corpus schema")
    count = meta.get("cases_per_family")
    variants = meta.get("variants")
    expected = expected_candidate_rows(cases_per_family=count, variants=variants)
    raw = ("\n".join(_raw_json(row) for row in expected) + "\n").encode("utf8")
    file = root / "train-identifier-variants.jsonl"
    if (not file.is_file() or file.is_symlink() or
            file.stat().st_size > MAX_FILE_BYTES or file.read_bytes() != raw):
        raise ValueError("Candidate source does not match canonical regeneration")
    reference = {
        "schema": SCHEMA, "seed": SEED,
        "cases_per_family": count, "variants": variants,
        "examples": len(expected), "families": sorted(VARIABLES),
        "train_only": True, "canonical_data_modified": False,
        "model_training_performed": False, "java_executed": False,
        "human_authored_independent_tasks": False,
        "derived_from": "project-authored synthetic train-family templates",
        "candidate_file": "train-identifier-variants.jsonl",
        "candidate_sha256": _sha256(raw),
    }
    if meta != reference:
        raise ValueError("Candidate manifest does not match canonical regeneration")
    return {"status": "PASS", **reference}
