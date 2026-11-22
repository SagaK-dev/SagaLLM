"""Regression tests for compile-only generation evidence (no Java execution)."""
from __future__ import annotations

import torch

from sagallm import java_instruction_sft as sft


class FakeTokenizer:
    SPECIAL = {"<eos>": 3}
    vocab_size = 260

    def decode(self, ids):
        return "public class Main { }" if ids else ""


class FakeModel:
    def __init__(self):
        self.config = type("Config", (), {"max_position_embeddings": 384})()

    def generate(self, prefix, **kwargs):
        return torch.cat((prefix, torch.tensor([[4, 3]], dtype=torch.long)), dim=1)


def test_compile_evidence_is_bounded_and_does_not_execute(monkeypatch):
    calls = []

    def compile_only(source, root, timeout):
        calls.append((source.name, timeout))
        return {"status": "FAIL", "exit_code": 1,
                "diagnostics": "Main.java:1: error: unexpected token"}

    monkeypatch.setattr(sft, "_compile", compile_only)
    rows = [{"id": "case-1", "family": "new-family",
             "prefix_ids": [1, 2], "source": "expected"}]
    report = sft.heldout_compilation(FakeModel(), FakeTokenizer(), rows, max_cases=1)
    case = report["cases"][0]
    assert report["behavior"] == "NOT_RUN"
    assert case["java_behavior"] == "NOT_RUN"
    assert case["compile"] == "FAIL"
    assert case["generated_tokens"] == 2
    assert case["generation_eos_seen"] is True
    assert case["generation_stop_reason"] == "EOS"
    assert case["javac_diagnostic_category"] == "JAVAC_DIAGNOSTIC"
    assert len(case["javac_stderr_sha256"]) == 64
    assert "diagnostics" not in case
    assert case["source_persisted"] is False
    assert calls == [("Main.java", 12)]


def test_missing_class_is_not_compiled(monkeypatch):
    class EmptyTokenizer(FakeTokenizer):
        def decode(self, ids):
            return "incomplete"

    def forbid(*args, **kwargs):
        raise AssertionError("javac must not be invoked")

    monkeypatch.setattr(sft, "_compile", forbid)
    rows = [{"id": "case-2", "family": "new-family",
             "prefix_ids": [1, 2], "source": "expected"}]
    report = sft.heldout_compilation(FakeModel(), EmptyTokenizer(), rows, max_cases=1)
    assert report["cases"][0]["compile_reason"] == "MISSING_PUBLIC_MAIN"
    assert report["cases"][0]["javac_stderr_sha256"] is None
