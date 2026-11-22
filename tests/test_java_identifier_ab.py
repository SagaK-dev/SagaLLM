"""A/B identifier experiment: paired inputs, isolation and conservative gate."""
from __future__ import annotations

import copy
import json

import pytest

from sagallm.java_identifier_curriculum import expected_candidate_rows
from scripts.train_java_identifier_ab import (
    ARMS, BATCH_SIZE, CASE_ID, EVAL_AT, LEARNING_RATE,
    PINNED_CANDIDATE_SHA256, SEED, STEPS, VALIDATION_COMPILE_EXAMPLES,
    baseline_row, paired_rows, validation_gate,
)


class ShortTokenizer:
    SPECIAL = {"<bos>": 1, "<eos>": 3}

    def encode(self, text: str) -> list[int]:
        # Toy token mapping; only prompt equality / bounded context matters.
        return [4] * (1 + len(text) // 12)


def test_pre_registered_bounded_experiment_settings():
    assert ARMS == ("canonical", "identifier_augmented")
    assert STEPS == 60
    assert BATCH_SIZE == 4
    assert EVAL_AT == (30, 60)
    assert SEED == 733
    assert LEARNING_RATE == 4e-5
    assert VALIDATION_COMPILE_EXAMPLES == 12
    assert PINNED_CANDIDATE_SHA256 == (
        "ad18e0a328f00db2bca204c3911afd66307099e3d41ba0f0d97c964463d3c2d9")


def test_original_train_row_preserves_prompt_and_identifier():
    row = expected_candidate_rows(cases_per_family=1, variants=1)[0]
    copy_before = copy.deepcopy(row)
    original = baseline_row(row)
    assert original["id"] == row["id"]
    assert original["family"] == row["family"]
    assert original["messages"][0] == row["messages"][0]
    assert original["messages"][1]["content"] != row["messages"][1]["content"]
    assert row == copy_before

    poisoned = copy.deepcopy(row)
    poisoned["family"] = "collatz_max"
    with pytest.raises(ValueError, match="heldout|unknown"):
        baseline_row(poisoned)
    poisoned = copy.deepcopy(row)
    poisoned["messages"][0]["content"] = "not a canonical task"
    with pytest.raises(ValueError):
        baseline_row(poisoned)
    poisoned = copy.deepcopy(row)
    poisoned["id"] = "invalid"
    with pytest.raises(ValueError):
        baseline_row(poisoned)


def test_paired_rows_hold_all_prompts_and_ids_constant(tmp_path):
    entries = expected_candidate_rows(cases_per_family=3, variants=1)
    filename = tmp_path / "train-only.jsonl"
    filename.write_text(
        "\n".join(json.dumps(row) for row in entries) + "\n", encoding="utf8")
    pairings, summary = paired_rows(filename, ShortTokenizer(), 384)
    assert summary["candidates_total"] == len(entries)
    assert summary["pairs_admitted"] + summary["pairs_excluded_context"] == len(entries)
    assert summary["pairs_admitted"] >= 24
    assert summary["train_families_admitted"] >= 5
    assert summary["same_training_prompts_and_ids"] is True
    for control, augmented in pairings:
        assert control["prefix_ids"] == augmented["prefix_ids"]
        assert control["family"] == augmented["family"]
        assert control["id"] == augmented["id"]
        assert control["task"] == augmented["task"]
        assert control["source"] != augmented["source"]
        assert control["x"].shape[0] <= 384
        assert augmented["x"].shape[0] <= 384
    with pytest.raises(ValueError):
        paired_rows(filename, ShortTokenizer(), 192)


def test_validation_gate_requires_compile_and_loss_without_test():
    control = {
        "validation_compiled_total": 12, "validation_compiled": 2,
        "best_validation_loss": 1.8,
    }
    amended = {
        "validation_compiled_total": 12, "validation_compiled": 4,
        "best_validation_loss": 1.82,
    }
    outcome = validation_gate(control, amended)
    assert outcome["status"] == "CANDIDATE_WARRANTS_INDEPENDENT_EVALUATION"
    assert outcome["production_adoption"] is False
    assert outcome["uses_test_data"] is False
    assert outcome["compile_delta"] == 2

    amended["validation_compiled"] = 3
    assert validation_gate(control, amended)["status"] == "NO_VALIDATED_COMPILE_GAIN"
    amended["validation_compiled"] = 4
    amended["best_validation_loss"] = 1.9
    assert validation_gate(control, amended)["status"] == "NO_VALIDATED_COMPILE_GAIN"
    amended["validation_compiled"] = True
    with pytest.raises(ValueError):
        validation_gate(control, amended)
    amended["validation_compiled"] = 4
    amended.pop("best_validation_loss")
    with pytest.raises(ValueError):
        validation_gate(control, amended)
