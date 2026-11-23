"""Pre-review QA for Exercism-derived Java I/O candidates (never runs Java)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from sagallm.java_external_io_preflight import (
    COVERAGE_WARNINGS, EXPECTED_TASKS, ORACLES, TOPIC_REVIEW_LEADS,
    _split, run_preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def _packet(dest: Path) -> Path:
    for rel in (
        "examples/java_external_io_INTAKE.json",
        "examples/java_external_io_sources.json",
        "third_party_notices/exercism-problem-specifications-LICENSE.txt",
    ):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    return dest


def test_entire_unreviewed_intake_matches_separate_algorithmic_specs():
    report = run_preflight(ROOT)
    assert report["status"] == "PRE_REVIEW_ORACLE_CHECKS_PASS"
    assert report["tasks"] == 6
    assert report["io_cases"] == 27
    assert report["specification_checks_passed"] == 27
    assert report["actual_independent_human_reviews"] == 0
    assert report["exact_family_label_collisions"] == 0
    assert report["conceptual_overlap_resolved"] is False
    assert report["review_gate"] == "REVIEW_REQUIRED"
    assert report["behavior"] == "NOT_RUN"
    assert report["generated_java_executed"] is False
    assert report["model_outputs_evaluated"] is False
    assert report["model_selection_permitted"] is False
    assert report["independence_proven"] is False
    assert set(x["task_id"] for x in report["reviewer_notes"]) == set(EXPECTED_TASKS)
    assert sum(x["io_cases_checked"] for x in report["reviewer_notes"]) == 27
    assert all(x["case_coverage_warnings"] for x in report["reviewer_notes"])
    assert all(x["conceptual_overlap_review_leads"]
               for x in report["reviewer_notes"])


def test_review_leads_reference_real_train_validation_or_test_families():
    assert len(TOPIC_REVIEW_LEADS) == len(COVERAGE_WARNINGS) == 6
    assert set(ORACLES) == set(TOPIC_REVIEW_LEADS)
    assert _split("char_count") == "train"
    assert _split("set_intersection") == "validation"
    assert _split("reverse_string") == "validation"
    assert _split("collatz_max") == "test"
    with pytest.raises(ValueError, match="unknown"):
        _split("fabricated_family_not_in_corpus")
    for hints in TOPIC_REVIEW_LEADS.values():
        assert all(_split(hint) in {"train", "validation", "test"}
                   for hint in hints)


def test_standalone_specifications_cover_important_behavior_edges():
    assert ORACLES["ex_hamming"]("G\nT\n") == "1\n"
    assert ORACLES["ex_hamming"]("\n\n") == "0\n"
    assert ORACLES["ex_rna_transcription"]("ACGT\n") == "UGCA\n"
    assert ORACLES["ex_luhn"]("059\n") == "true\n"
    assert ORACLES["ex_luhn"]("1\n") == "false\n"
    assert ORACLES["ex_luhn"]("59%59\n") == "false\n"
    assert ORACLES["ex_luhn"]("055 444 285\n") == "true\n"
    assert ORACLES["ex_roman_numerals"]("4\n") == "IV\n"
    assert ORACLES["ex_roman_numerals"]("40\n") == "XL\n"
    assert ORACLES["ex_roman_numerals"]("90\n") == "XC\n"
    assert ORACLES["ex_roman_numerals"]("400\n") == "CD\n"
    assert ORACLES["ex_roman_numerals"]("900\n") == "CM\n"
    assert ORACLES["ex_allergies"]("255 cats\n") == "true\n"
    assert ORACLES["ex_allergies"]("2 eggs\n") == "false\n"
    assert ORACLES["ex_atbash_cipher"]("yes\n") == "bvh\n"
    assert ORACLES["ex_atbash_cipher"]("OMG\n") == "lnt\n"
    with pytest.raises(ValueError):
        ORACLES["ex_hamming"]("A\nAA\n")
    with pytest.raises(ValueError):
        ORACLES["ex_rna_transcription"]("X\n")
    with pytest.raises(ValueError):
        ORACLES["ex_roman_numerals"]("0\n")


def test_mutated_upstream_expected_output_fails_independent_spec(tmp_path):
    packet = _packet(tmp_path / "packet")
    path = packet / "examples/java_external_io_INTAKE.json"
    suite = json.loads(path.read_text(encoding="utf8"))
    suite["tasks"][2]["cases"][1]["stdout"] = "false\n"
    path.write_text(json.dumps(suite), encoding="utf8")
    with pytest.raises(ValueError, match="Independent specification mismatch: ex_luhn/1"):
        run_preflight(packet)


def test_mutated_case_stdin_fails_even_when_expected_remains(tmp_path):
    packet = _packet(tmp_path / "packet")
    path = packet / "examples/java_external_io_INTAKE.json"
    suite = json.loads(path.read_text(encoding="utf8"))
    suite["tasks"][0]["cases"][1]["stdin"] = "G\nG\n"
    path.write_text(json.dumps(suite), encoding="utf8")
    with pytest.raises(ValueError, match="Independent specification mismatch: ex_hamming/1"):
        run_preflight(packet)


def test_cli_output_does_not_contain_cases_or_model_code(tmp_path):
    report = run_preflight(ROOT)
    encoded = json.dumps(report)
    assert "055 444 285" not in encoded
    assert "public class Main" not in encoded
    assert "GGACGGATTCTG" not in encoded
    assert '"java_executed": true' not in encoded
    assert report["candidate_assessment_only"] is True
    assert report["human_reviewers_verified"] if "human_reviewers_verified" in report else True
