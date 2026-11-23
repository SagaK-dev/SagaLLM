"""Public unreviewed Java compiler-only diagnostics: fail-closed contracts."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import evaluate_java_public_compile_diagnostic as probe


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    SPECIAL = {"<bos>": 1, "<eos>": 2}
    vocab_size = 500

    def encode(self, text):
        return [5] * (len(text) // 3)

    def decode(self, ids):
        return "public class Main { public static void main(String[] a) {} }"


class FakeModel:
    config = SimpleNamespace(max_position_embeddings=384)

    def generate(self, prefix, **kwargs):
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_new_tokens"] <= 170
        assert kwargs["vocab_limit"] == 500
        return torch.cat([prefix, torch.tensor([[10, 2]])], dim=1)


def test_frozen_unreviewed_six_task_intake_is_not_human_approved():
    ids, suite_sha, total = probe.fixed_public_tasks(
        ROOT / "examples/java_external_io_INTAKE.json")
    assert ids == list(probe.PROMPTS)
    assert len(ids) == 6
    assert total == 27
    assert suite_sha == probe.PINNED_SUITE_SHA256
    for item in probe.PROMPTS.values():
        assert 0 < len(item) <= 240
        assert "\n" not in item
    assert probe.PINNED_CONTEXT == 384
    assert probe.MAX_OUTPUT_TOKENS == 170


def test_public_intake_refuses_to_look_approved_or_changed(tmp_path):
    suite = json.loads(
        (ROOT / "examples/java_external_io_INTAKE.json").read_text(encoding="utf8"))
    path = tmp_path / "candidate.json"
    suite["review_status"] = "APPROVED_FOR_INDEPENDENT_EVALUATION"
    path.write_text(json.dumps(suite), encoding="utf8")
    with pytest.raises(ValueError, match="pre-registration"):
        probe.fixed_public_tasks(path)
    suite["review_status"] = "PENDING_HUMAN_REVIEW"
    suite["tasks"][0]["prompt"] += " new test wording"
    path.write_text(json.dumps(suite), encoding="utf8")
    with pytest.raises(ValueError, match="pre-registration"):
        probe.fixed_public_tasks(path)


def test_score_compiles_without_executing_or_persisting_source(tmp_path, monkeypatch):
    saved_sources = []
    def fake_javac(path, root, timeout):
        saved_sources.append(path.read_text(encoding="utf8"))
        assert timeout == 12
        assert path.name == "Main.java"
        return {"status": "PASS", "diagnostics": ""}
    monkeypatch.setattr(probe, "_compile", fake_javac)
    report = probe.score_task(FakeModel(), FakeTokenizer(), "ex_hamming")
    assert report["status"] == "PASS"
    assert report["reason"] == "JAVAC_PASS"
    assert report["source_saved"] is False
    assert report["java_executed"] is False
    assert report["eos_seen"] is True
    assert report["generated_tokens"] == 2
    assert saved_sources == [
        "public class Main { public static void main(String[] a) {} }"]
    serialized = json.dumps(report)
    assert "public class Main" not in serialized
    assert "diagnostics" not in serialized


def test_compiler_errors_are_only_bounded_categories(monkeypatch):
    monkeypatch.setattr(probe, "_compile", lambda *a, **k: {
        "status": "FAIL",
        "diagnostics": (
            "Main.java:1: error: cannot find symbol\n"
            "  symbol: variable secretIdentifier\n"
            "  location: class Main\n"
        ),
    })
    report = probe.score_task(FakeModel(), FakeTokenizer(), "ex_luhn")
    assert report["status"] == "FAIL"
    assert report["error_category"] == "MISSING_SYMBOL"
    assert report["symbol_kind"] == "VARIABLE"
    assert "secretIdentifier" not in json.dumps(report)
    assert report["java_executed"] is False


def test_missing_javac_is_not_counted_as_compilation_failure(monkeypatch):
    monkeypatch.setattr(probe, "_compile", lambda *a, **k: {
        "status": "NOT_RUN", "reason": "javac unavailable",
    })
    report = probe.score_task(FakeModel(), FakeTokenizer(), "ex_allergies")
    assert report["status"] == "NOT_RUN"
    assert report["reason"] == "COMPILER_UNAVAILABLE"
    assert report["java_executed"] is False


def test_model_context_shortage_is_skipped_before_generation():
    class ExpansiveTokenizer(FakeTokenizer):
        def encode(self, text):
            return [5] * 384

    class NoGenerate(FakeModel):
        def generate(self, *a, **k):
            raise AssertionError("Generation must be skipped")

    report = probe.score_task(NoGenerate(), ExpansiveTokenizer(), "ex_atbash_cipher")
    assert report["status"] == "SKIPPED"
    assert report["reason"] == "INSUFFICIENT_CONTEXT"
    assert report["java_executed"] is False


def test_unregistered_prompt_and_context_rejected():
    with pytest.raises(ValueError):
        probe.score_task(FakeModel(), FakeTokenizer(), "ex_unsupported")
    model = FakeModel()
    model.config = SimpleNamespace(max_position_embeddings=192)
    with pytest.raises(ValueError):
        probe.score_task(model, FakeTokenizer(), "ex_hamming")
