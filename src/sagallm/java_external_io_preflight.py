"""Non-executing preflight for external Java benchmark candidate review.

This is NOT human review, a proof of independence, or a Java model test.
Six small standalone Python specifications cross-check the 27 adapted
Exercism I/O vectors, then flag plausible curricular topic overlap and
known coverage gaps for actual human reviewers. No Java or model is run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .java_diverse_corpus import NEW_TRAIN, NEW_TEST, NEW_VALIDATION
from .java_external_io_intake import verify_intake
from .java_large_corpus import TRAIN_FAMILIES, TEST_FAMILIES, VALIDATION_FAMILIES

# Curated *review leads*, not automatic proof of semantic intersection.
# Keep every lead tied to a genuinely existing SagaLLM family.
TOPIC_REVIEW_LEADS = {
    "ex_hamming": ("char_count", "array_count_greater"),
    "ex_rna_transcription": ("replace", "lower"),
    "ex_luhn": ("digit_sum", "numeric_string", "divisible"),
    "ex_roman_numerals": ("hex_format", "binary_format"),
    "ex_allergies": ("bit_count", "power_two", "set_intersection"),
    "ex_atbash_cipher": ("replace", "lower", "reverse_string"),
}
# Known weaknesses of the *selected cases*, not claims about all Exercism cases.
COVERAGE_WARNINGS = {
    "ex_hamming": (
        "Only same-length strands selected; unequal-length error handling excluded.",
    ),
    "ex_rna_transcription": (
        "Selected cases lack separate one-character A and T checks.",
    ),
    "ex_luhn": (
        "Selected cases omit punctuation, tab and empty/space-only inputs.",
    ),
    "ex_roman_numerals": (
        "Only 1, 4, 9 and 3999; 40, 90, 400, 900 not individually checked.",
    ),
    "ex_allergies": (
        "All five selected cases check only eggs; other allergen bits untested.",
    ),
    "ex_atbash_cipher": (
        "Only encoding selected; decoding and Unicode outside this adapter.",
    ),
}
EXPECTED_TASKS = tuple(TOPIC_REVIEW_LEADS)


def _line(stdin: str) -> str:
    if not isinstance(stdin, str) or not stdin.endswith("\n"):
        raise ValueError("Expected line-terminated standard input")
    value = stdin[:-1]
    if "\n" in value or "\r" in value:
        raise ValueError("Only one logical line supported")
    return value


def _hamming(stdin: str) -> str:
    if not stdin.endswith("\n"):
        raise ValueError("Expected two lines")
    parts = stdin[:-1].split("\n")
    if len(parts) != 2 or len(parts[0]) != len(parts[1]):
        raise ValueError("Requires two equal-length strands")
    if any(c not in "ACGT" for part in parts for c in part):
        raise ValueError("Unexpected DNA character")
    return str(sum(a != b for a, b in zip(*parts))) + "\n"


def _rna(stdin: str) -> str:
    dna = _line(stdin)
    if any(c not in "ACGT" for c in dna):
        raise ValueError("Unexpected DNA character")
    return dna.translate(str.maketrans("ACGT", "UGCA")) + "\n"


def _luhn(stdin: str) -> str:
    number = _line(stdin)
    if any(c not in "0123456789 " for c in number):
        return "false\n"
    digits = [ord(c) - 48 for c in number if c != " "]
    if len(digits) < 2:
        return "false\n"
    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        value = digit * 2 if index % 2 else digit
        checksum += value - 9 if value > 9 else value
    return ("true" if checksum % 10 == 0 else "false") + "\n"


def _roman(stdin: str) -> str:
    text = _line(stdin)
    if not text.isascii() or not text.isdecimal():
        raise ValueError("Expected decimal integer")
    value = int(text)
    if not 1 <= value <= 3999:
        raise ValueError("Outside Roman numeral exercise range")
    out = []
    for magnitude, glyph in (
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ):
        repeats, value = divmod(value, magnitude)
        out.append(glyph * repeats)
    return "".join(out) + "\n"


ALLERGEN_BITS = {
    "eggs": 1, "peanuts": 2, "shellfish": 4, "strawberries": 8,
    "tomatoes": 16, "chocolate": 32, "pollen": 64, "cats": 128,
}


def _allergies(stdin: str) -> str:
    parts = _line(stdin).split()
    if len(parts) != 2 or parts[1] not in ALLERGEN_BITS:
        raise ValueError("Expected a score and known allergen")
    try:
        score = int(parts[0])
    except ValueError as exc:
        raise ValueError("Invalid allergy score") from exc
    return ("true" if score & ALLERGEN_BITS[parts[1]] else "false") + "\n"


def _atbash(stdin: str) -> str:
    text = _line(stdin)
    chars = []
    for character in text.lower():
        if "a" <= character <= "z":
            chars.append(chr(ord("z") - (ord(character) - ord("a"))))
        elif "0" <= character <= "9":
            chars.append(character)
    raw = "".join(chars)
    return " ".join(raw[pos:pos + 5] for pos in range(0, len(raw), 5)) + "\n"


ORACLES = {
    "ex_hamming": _hamming,
    "ex_rna_transcription": _rna,
    "ex_luhn": _luhn,
    "ex_roman_numerals": _roman,
    "ex_allergies": _allergies,
    "ex_atbash_cipher": _atbash,
}


def _split(family: str) -> str:
    if family in set(TRAIN_FAMILIES) | set(NEW_TRAIN):
        return "train"
    if family in set(VALIDATION_FAMILIES) | set(NEW_VALIDATION):
        return "validation"
    if family in set(TEST_FAMILIES) | set(NEW_TEST):
        return "test"
    raise ValueError("Review hint references an unknown synthetic family")


def run_preflight(root: Path) -> dict:
    """Check all 27 fixtures; emit no raw prompts, test inputs or Java."""
    root = Path(root)
    integrity = verify_intake(root)
    if integrity["review_gate"] != "REVIEW_REQUIRED":
        raise ValueError("Intake unexpectedly approved for evaluation")
    suite_bytes = (root / "examples/java_external_io_INTAKE.json").read_bytes()
    suite = json.loads(suite_bytes)
    provenance = json.loads(
        (root / "examples/java_external_io_sources.json").read_text(encoding="utf8"))
    by_task = {item["task_id"]: item for item in provenance["sources"]}
    if (tuple(task["id"] for task in suite["tasks"]) != EXPECTED_TASKS or
            set(by_task) != set(EXPECTED_TASKS)):
        raise ValueError("Unexpected suite tasks or missing provenance")
    verified = 0
    results = []
    for task in suite["tasks"]:
        task_id = task["id"]
        for index, case in enumerate(task["cases"]):
            try:
                actual = ORACLES[task_id](case["stdin"])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(
                    "Independent specification rejected case: " +
                    task_id + "/" + str(index)) from exc
            if actual != case["stdout"]:
                raise ValueError(
                    "Independent specification mismatch: " +
                    task_id + "/" + str(index))
            verified += 1
        leads = [
            {"family": family, "original_split": _split(family),
             "kind": "CONCEPTUAL_REVIEW_LEAD_NOT_PROVEN_OVERLAP"}
            for family in TOPIC_REVIEW_LEADS[task_id]
        ]
        results.append({
            "task_id": task_id,
            "source_exercise": by_task[task_id]["exercise"],
            "io_cases_checked": len(task["cases"]),
            "standalone_python_specification": "PASS",
            "conceptual_overlap_review_leads": leads,
            "case_coverage_warnings": list(COVERAGE_WARNINGS[task_id]),
            "original_source_rights": "HUMAN_REVIEW_PENDING",
            "console_adapter_signoff": "HUMAN_REVIEW_PENDING",
            "semantic_contamination_review": "HUMAN_REVIEW_PENDING",
        })
    if verified != 27 or integrity["io_cases"] != verified:
        raise ValueError("Expected all original 27 cases to be present")
    return {
        "status": "PRE_REVIEW_ORACLE_CHECKS_PASS",
        "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "source_revision": integrity["source_revision"],
        "tasks": len(results),
        "io_cases": verified,
        "specification_checks_passed": verified,
        "actual_independent_human_reviews": 0,
        "exact_family_label_collisions": integrity["exact_family_label_overlap_count"],
        "conceptual_overlap_resolved": False,
        "review_gate": "REVIEW_REQUIRED",
        "behavior": "NOT_RUN",
        "generated_java_executed": False,
        "model_outputs_evaluated": False,
        "model_selection_permitted": False,
        "independence_proven": False,
        "candidate_assessment_only": True,
        "reviewer_notes": results,
    }
