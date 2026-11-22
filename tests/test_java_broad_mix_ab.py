"""Fixed 90-family, low-LR Java SFT A/B experiment contract."""
from __future__ import annotations

import ast
import inspect

import pytest

from sagallm.java_diverse_corpus import NEW_TRAIN
from sagallm.java_identifier_curriculum import VARIABLES
from sagallm.java_large_corpus import TRAIN_FAMILIES
from scripts import train_java_broad_mix_ab as ab


def training_rows():
    families = sorted(set(TRAIN_FAMILIES) | set(NEW_TRAIN))
    assert len(families) == 90
    return [
        {"id": f"train-{family}-{i:03d}", "family": family,
         "prefix_ids": [1, 2, 3]}
        for family in families for i in range(12)
    ]


def candidate_pairs():
    data = []
    for family in sorted(VARIABLES):
        for i in range(12):
            first = {"id": f"candidate-{family}-{i:03d}",
                     "family": family, "prefix_ids": [1, 2]}
            alternate = {**first}
            data.append((first, alternate))
    assert len(data) == 120
    return data


def test_locked_protocol_parameters_and_train_only_data():
    assert ab.EXPERIMENT == "JAVA_BROAD_MIXTURE_LOW_LR_AB_V1"
    assert ab.STEPS == 80
    assert ab.SEED == 812
    assert ab.BATCH_SIZE == 4
    assert ab.BROAD_PER_BATCH == 3
    assert ab.PAIRED_PER_BATCH == 1
    assert ab.BROAD_PER_BATCH + ab.PAIRED_PER_BATCH == ab.BATCH_SIZE
    assert ab.LR == 1e-5
    assert ab.EVAL_AT == (40, 80)
    assert ab.VALIDATION_COMPILE_EXAMPLES == 12
    assert ab.REQUIRED_CANONICAL_FAMILIES == 80
    assert ab.REQUIRED_CANONICAL_ROWS == 1000
    assert set(VARIABLES).isdisjoint({"collatz_max", "longest_common_prefix"})
    # Inspect actual calls, not comments: read_split must target train or validation.
    tree = ast.parse(inspect.getsource(ab.experiment))
    accessed = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == "read_split"):
            accessed.append(node.args[1].value)
    assert accessed == ["train", "validation"]


def test_fixed_schedule_is_identical_and_family_broad():
    rows = training_rows()
    pairs = candidate_pairs()
    first = ab.fixed_training_schedule(rows, pairs)
    second = ab.fixed_training_schedule(rows, pairs)
    assert first == second
    assert len(first) == 80
    assert all(len(batch) == 4 for batch in first)
    assert all([kind for kind, _ in batch] ==
               ["broad", "broad", "broad", "paired"] for batch in first)
    summary = ab.schedule_summary(first, rows, pairs)
    assert summary["total_train_exposures"] == 320
    assert summary["canonical_broad_exposures"] == 240
    assert summary["paired_exposures"] == 80
    assert summary["broad_families_seen"] == 90
    assert summary["overall_train_families_seen"] == 90
    assert summary["identical_example_order_and_prompts"] is True
    assert summary["no_training_example_repeated_within_source"] is True
    assert summary["paired_ratio"] == "1/4"
    assert summary["test_examples_in_training"] is False

    trial = ab.fixed_training_schedule(rows, pairs, steps=8)
    assert len(trial) == 8
    assert trial == first[:8]


def test_fixed_schedule_rejects_insufficient_or_leaky_training():
    broad = training_rows()
    pairs = candidate_pairs()
    with pytest.raises(ValueError):
        ab.fixed_training_schedule(broad[:999], pairs)
    with pytest.raises(ValueError):
        ab.fixed_training_schedule(broad, pairs[:99])
    with pytest.raises(ValueError):
        ab.fixed_training_schedule(broad, pairs, steps=81)
    with pytest.raises(ValueError):
        ab.fixed_training_schedule(broad, pairs, steps=0)
    changed = [dict(row) for row in broad]
    changed[0]["family"] = "collatz_max"
    with pytest.raises(ValueError, match="Non-TRAIN"):
        ab.fixed_training_schedule(changed, pairs)
    invalid = list(pairs)
    lhs, rhs = invalid[0]
    invalid[0] = (lhs, {**rhs, "prefix_ids": [9, 9]})
    with pytest.raises(ValueError, match="Unpaired"):
        ab.fixed_training_schedule(broad, invalid)
    invalid = list(pairs)
    lhs, rhs = invalid[0]
    invalid[0] = ({**lhs, "family": "collatz_max"}, rhs)
    with pytest.raises(ValueError, match="Unpaired"):
        ab.fixed_training_schedule(broad, invalid)


def test_summary_rejects_malformed_schedule():
    broad = training_rows()
    pairs = candidate_pairs()
    schedule = ab.fixed_training_schedule(broad, pairs, steps=3)
    bad = list(schedule)
    bad[1] = bad[0]
    with pytest.raises(ValueError, match="repeats"):
        ab.schedule_summary(bad, broad, pairs)
