"""Matched algorithm-pairing SFT and prompt-collapse diagnostics, CPU-only tests."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from sagallm.java_stdio_curriculum import (
    TRAIN_FAMILIES, VALIDATION_FAMILIES, validation_oracle_rows,
)
from scripts import train_java_contrastive_stdio_ab as ab


def dummy_data():
    broad = [
        {"id": f"train-old-{i}", "family": f"old_train_{i % 90}"}
        for i in range(400)
    ]
    stdio = [
        {"id": f"train-{family}-{index:03d}", "family": family}
        for family in TRAIN_FAMILIES for index in range(32)
    ]
    return broad, stdio


def test_match_only_changes_new_stdio_order_never_train_membership():
    broad, stdio = dummy_data()
    control = ab.train_schedule(broad, stdio)
    treatment = ab.contrastive_schedule(broad, stdio, control)
    evidence = ab.schedule_evidence(control, treatment)
    assert evidence["same_training_examples_both_arms"] is True
    assert evidence["same_original_broad_row_positions"] is True
    assert evidence["unique_training_exposures"] is True
    assert evidence["broad_rows_per_arm"] == 320
    assert evidence["stdio_rows_per_arm"] == 320
    assert evidence["validation_or_test_in_optimizer"] is False
    assert treatment == ab.contrastive_schedule(broad, stdio, control)
    assert any(treatment[i] != control[i] for i in range(ab.STEPS))
    assert all([kind for kind, _ in batch] ==
               ["broad", "broad", "stdio", "stdio"] for batch in treatment)


def test_each_contrastive_batch_shares_variant_but_changes_algorithm():
    broad, stdio = dummy_data()
    control = ab.train_schedule(broad, stdio)
    paired = ab.contrastive_schedule(broad, stdio, control)
    occurrences = []
    for batch in paired:
        (a_kind, a), (b_kind, b) = batch[2:]
        assert a_kind == b_kind == "stdio"
        a_rec, b_rec = stdio[a], stdio[b]
        assert a_rec["id"][-3:] == b_rec["id"][-3:]
        assert a_rec["family"] != b_rec["family"]
        assert (a_rec["family"], b_rec["family"]) in ab.FAMILY_PAIRS
        occurrences.extend((a,b))
    assert len(occurrences) == len(set(occurrences)) == 320
    assert set(occurrences) == set(range(320))
    assert set(family for pair in ab.FAMILY_PAIRS for family in pair) == set(TRAIN_FAMILIES)


def test_wrong_id_and_hidden_family_fails_closed():
    broad, stdio = dummy_data()
    control = ab.train_schedule(broad, stdio)
    modified = list(stdio)
    modified[0] = {**modified[0], "id": "train-unknown-001"}
    with pytest.raises(ValueError, match="Unknown stdin TRAIN id"):
        ab.contrastive_schedule(broad, modified, control)
    modified = list(stdio)
    modified[0] = {**modified[0], "family": VALIDATION_FAMILIES[0]}
    with pytest.raises(ValueError, match="Non-TRAIN"):
        ab.contrastive_schedule(broad, modified, control)
    corrupted = [tuple(batch) for batch in control]
    corrupted[0] = corrupted[0][2:] + corrupted[0][:2]
    with pytest.raises(ValueError, match="two broad"):
        ab.contrastive_schedule(broad, stdio, corrupted)


def test_no_behavior_task_can_enter_loss_selection():
    valid = [
        {"id": record["id"], "family": record["family"]}
        for record in validation_oracle_rows()
    ]
    heldback, loss_examples = ab.fixed_behavior_only(valid)
    assert len(heldback) == 6
    assert len(loss_examples) == 24
    id_set = {item["id"] for item, _ in heldback}
    assert all(item["id"] not in id_set for item in loss_examples)
    assert len(id_set) == 6
    assert {item["family"] for item, _ in heldback} == set(VALIDATION_FAMILIES)
    assert all(item["id"].endswith(("-002", "-003")) for item, _ in heldback)
    with pytest.raises(ValueError, match="excluded"):
        ab.fixed_behavior_only(valid[:2])


def _cases(hashes: list[str], families: list[str], passed: int = 0,
           unavailable: bool = False) -> dict:
    return {
        "total_cases": 18, "cases_passed": passed,
        "tasks_all_pass": passed // 3,
        "oci_invocations": 18, "unavailable": unavailable,
        "cases": [
            {"id": f"task-{i}", "family": families[i], "source_sha256": hashes[i]}
            for i in range(6)
        ],
    }


def test_prompt_collision_counts_cross_family_and_within_family_without_source():
    families = [name for name in VALIDATION_FAMILIES for _ in range(2)]
    base = _cases(["A","A","A","B","C","C"], families)
    stats = ab.conditioning_metrics(base)
    assert stats["unique_generated_sources"] == 3
    assert stats["cross_family_identical_source_pairs"] == 2
    assert stats["within_family_identical_source_pairs"] == 2
    assert stats["behavioral_semantics_proven"] is False
    with pytest.raises(ValueError, match="coverage"):
        ab.conditioning_metrics({**base, "cases": base["cases"][:5]})


def test_gate_requires_real_case_gain_and_more_unique_sources():
    fam = [name for name in VALIDATION_FAMILIES for _ in range(2)]
    baseline = _cases(["A"] * 6, fam, passed=0)
    control = _cases(["A", "A", "B", "B", "C", "C"], fam, passed=0)
    contrast = _cases(["A", "B", "C", "D", "E", "F"], fam, passed=3)
    metrics = {name: ab.conditioning_metrics(item)
               for name, item in zip(
                   ("base_v018", *ab.ARMS),
                   (baseline, control, contrast))}
    good = ab.selection_gate(baseline, control, contrast, metrics)
    assert good["decision"] == "CONTRASTIVE_VALIDATION_SIGNAL_NOT_DEPLOYED"
    assert good["improvement_demonstrated"] is True
    assert good["production_adoption"] is False
    no_gain = ab.selection_gate(
        baseline, control, _cases(["A"] * 6, fam, passed=0),
        metrics)
    assert no_gain["improvement_demonstrated"] is False
    assert no_gain["decision"] == "NO_VALIDATED_CONTRASTIVE_GAIN"
    unavailable = ab.selection_gate(
        baseline, _cases(["A"] * 6, fam, unavailable=True), contrast, metrics)
    assert unavailable["decision"] == "INCONCLUSIVE_OCI_UNAVAILABLE"
    assert unavailable["production_adoption"] is False


def test_synthetic_protocol_never_reads_old_test_or_exercism():
    assert ab.EXPERIMENT == "JAVA_CONTRASTIVE_STDIO_MATCHED_AB_V1"
    assert ab.PINNED_V018 == "9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc"
    assert ab.STEPS == 160 and ab.EVAL_AT == (80, 160)
    assert ab.SEED == 4201 and ab.LR == 2e-5
    assert ab.BEHAVIOR_VARIANTS == (2, 3)
    assert len(ab.FAMILY_PAIRS) == 5
    assert ab.ARMS == ("control_interleaved", "contrastive_same_variant")
