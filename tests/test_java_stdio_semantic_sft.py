"""Stdin semantics SFT scheduling and OCI-only behavioral grading."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from sagallm.java_stdio_curriculum import TRAIN_FAMILIES, VALIDATION_FAMILIES, validation_oracle_rows
from scripts import train_java_stdio_semantic_sft as exp


def _broad(n=400):
    return [
        {"id": f"broad-{i}", "family": f"old_train_{i % 40}",
         "prefix_ids": [1, 2, 3]}
        for i in range(n)
    ]


def _stdio():
    return [
        {"id": f"stdin-{family}-{i}", "family": family,
         "prefix_ids": [1, 2, 3]}
        for family in TRAIN_FAMILIES for i in range(32)
    ]


def _validation():
    return [
        {"id": record["id"], "family": record["family"],
         "prefix_ids": [1, 2, 3]}
        for record in validation_oracle_rows()
    ]


def test_training_schedule_fixed_and_without_train_replacement():
    assert exp.EXPERIMENT == "JAVA_STDIO_SEMANTIC_SFT_V1"
    assert exp.STEPS == 160 and exp.BATCH == 4
    assert exp.BROAD_PER_BATCH == 2 and exp.STDIO_PER_BATCH == 2
    assert exp.EVAL_AT == (80, 160)
    assert exp.LR == 2e-5
    a = exp.train_schedule(_broad(), _stdio())
    b = exp.train_schedule(_broad(), _stdio())
    assert a == b
    assert len(a) == 160 and all(len(batch) == 4 for batch in a)
    assert all([kind for kind, _ in batch] ==
               ["broad", "broad", "stdio", "stdio"] for batch in a)
    broad = [idx for batch in a for kind, idx in batch if kind == "broad"]
    stdio = [idx for batch in a for kind, idx in batch if kind == "stdio"]
    assert len(broad) == len(stdio) == 320
    assert len(set(broad)) == len(set(stdio)) == 320
    assert a[:5] == exp.train_schedule(_broad(), _stdio(), steps=5)


def test_train_rejects_validation_family_and_short_datasets():
    broad = _broad()
    stdio = _stdio()
    with pytest.raises(ValueError):
        exp.train_schedule(broad, stdio[:319])
    with pytest.raises(ValueError, match="Missing/foreign"):
        exp.train_schedule(broad, [{**s, "family": "stdin_min_offset"}
                                         if i == 0 else s
                                   for i, s in enumerate(stdio)])
    contaminated = list(broad)
    contaminated[0] = {**contaminated[0], "family": VALIDATION_FAMILIES[0]}
    with pytest.raises(ValueError, match="Validation"):
        exp.train_schedule(contaminated, stdio)
    with pytest.raises(ValueError, match="unique"):
        exp.train_schedule(broad, stdio, steps=161)


def test_validation_samples_are_disjoint_and_exactly_six_tasks():
    pairs = exp.fixed_validation_samples(_validation(), validation_oracle_rows())
    assert len(pairs) == exp.VALIDATION_PROMPTS == 6
    assert sum(len(oracle["cases"]) for _, oracle in pairs) == 18
    assert set(row["family"] for row, _ in pairs) == set(VALIDATION_FAMILIES)
    assert all(row["id"] == oracle["id"] for row, oracle in pairs)
    with pytest.raises(ValueError):
        exp.fixed_validation_samples(_validation()[2:], validation_oracle_rows())


def test_behavioral_oci_evaluates_all_cases_and_keeps_no_raw_text(monkeypatch):
    samples = exp.fixed_validation_samples(_validation(), validation_oracle_rows())
    fake_model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=384))
    monkeypatch.setattr(exp, "_generate", lambda model, tokenizer, row:
                        ("public class Main { public static void main(String[] a){} }",
                         "GENERATED"))
    calls = []
    def fake_oci(source, expected, *, backend, image, allow_execution,
                 stdin_data, timeout):
        assert allow_execution is True and backend == "docker"
        assert timeout == 20 and image.startswith("sha256:")
        calls.append((stdin_data, expected))
        return {"status": "FAIL", "returncode": 0}
    monkeypatch.setattr(exp, "run_java_in_oci", fake_oci)
    result = exp.behavioral_validation(fake_model, None, samples, "sha256:" + "a" * 64)
    assert result["tasks"] == 6 and result["total_cases"] == 18
    assert result["oci_invocations"] == 18
    assert result["cases_passed"] == 0 and result["tasks_all_pass"] == 0
    assert result["unavailable"] is False
    assert result["source_persisted"] is False
    assert all(case["case_categories"] == ["OUTPUT_MISMATCH_ZERO_EXIT"] * 3
               for case in result["cases"])
    assert len(calls) == 18
    rendered = json.dumps(result)
    assert "public class Main" not in rendered
    assert "MiXeD Case" not in rendered


def test_invalid_source_never_enters_oci(monkeypatch):
    samples = exp.fixed_validation_samples(_validation(), validation_oracle_rows())
    monkeypatch.setattr(exp, "_generate", lambda *a: (None, "INVALID_GENERATED_JAVA"))
    monkeypatch.setattr(exp, "run_java_in_oci", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("Never execute invalid source")))
    result = exp.behavioral_validation(None, None, samples, "sha256:" + "b" * 64)
    assert result["oci_invocations"] == 0
    assert result["cases_passed"] == 0
    assert all("NOT_RUN_INVALID_GENERATION" in result["cases"][0]["case_categories"]
               for _ in range(3))


def test_runtime_unavailability_must_not_be_treated_as_zero_model_quality(monkeypatch):
    samples = exp.fixed_validation_samples(_validation(), validation_oracle_rows())
    monkeypatch.setattr(exp, "_generate", lambda *a: ("public class Main{}", "GENERATED"))
    monkeypatch.setattr(exp, "run_java_in_oci", lambda *a, **k:
                        {"status": "RUNTIME_ERROR", "returncode": 125})
    data = exp.behavioral_validation(None, None, samples, "sha256:" + "c" * 64)
    assert data["unavailable"] is True
    gate = exp.conservative_gate(data, data, selected_step=160,
                                 initial_loss=2.0, selected_loss=1.0)
    assert gate["decision"] == "INCONCLUSIVE_RUNTIME_UNAVAILABLE"
    assert gate["production_adoption"] is False


def _score(passed, *, unavailable=False, total=18):
    return {"total_cases": total, "cases_passed": passed,
            "tasks_all_pass": passed // 3,
            "oci_invocations": total, "unavailable": unavailable}


def test_gate_requires_both_loss_and_behavior_gain_and_never_deploys():
    better = exp.conservative_gate(
        _score(1), _score(6), selected_step=160,
        initial_loss=2.0, selected_loss=1.0)
    assert better["validated_gain"] is True
    assert better["production_adoption"] is False
    unchanged = exp.conservative_gate(
        _score(1), _score(1), selected_step=0,
        initial_loss=2.0, selected_loss=2.0)
    assert unchanged["validated_gain"] is False
    assert unchanged["decision"] == "NO_VALIDATED_BEHAVIOR_GAIN"
    with pytest.raises(ValueError):
        exp.conservative_gate(
            _score(2), _score(2, total=15), selected_step=160,
            initial_loss=2.0, selected_loss=1.0)
