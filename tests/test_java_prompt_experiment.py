"""Pre-registered validation prompt experiment: deterministic, compile-only."""
from __future__ import annotations

import pytest
import torch

from sagallm import java_instruction_sft as sft


class Tokenizer:
    SPECIAL = {"<bos>": 1, "<eos>": 3}
    vocab_size = 256

    def encode(self, text):
        return [4 + ord(ch) % 200 for ch in text]

    def decode(self, ids):
        return "public class Main { }"


class Model:
    def __init__(self):
        self.config = type("Config", (), {"max_position_embeddings": 384})()
        self.last_prefix = None

    def generate(self, prefix, **kwargs):
        self.last_prefix = prefix[0].tolist()
        assert kwargs["temperature"] == 0.0
        return torch.cat((prefix, torch.tensor([[6, 3]], dtype=torch.long)), 1)


def _rows():
    return [{"id": f"case-{i}", "family": f"family-{i}",
             "task": f"Print value {i}", "prefix_ids": [1, 2, 5],
             "source": f"public class Main {{ int x{i}; }}"}
            for i in range(4)]


def test_style_selection_tie_canonical_and_no_unknown_styles():
    styles = sft.JAVA_EVAL_PROMPT_STYLES
    assert styles == ("canonical", "declare_variables")
    assert sft.select_java_eval_style(
        {"canonical": 0, "declare_variables": 0}) == "canonical"
    assert sft.select_java_eval_style(
        {"canonical": 1, "declare_variables": 2}) == "declare_variables"
    assert sft.select_java_eval_style(
        {"canonical": 2, "declare_variables": 1}) == "canonical"
    with pytest.raises(ValueError):
        sft.select_java_eval_style({"canonical": 2})
    with pytest.raises(ValueError):
        sft.select_java_eval_style(
            {"canonical": True, "declare_variables": 0})


def test_style_modifier_only_reencodes_prompt_and_retains_original(monkeypatch):
    tok = Tokenizer()
    row = _rows()[0]
    canonical = sft.java_eval_prefix(row, tok, "canonical")
    alternate = sft.java_eval_prefix(row, tok, "declare_variables")
    assert canonical == row["prefix_ids"]
    assert alternate[0] == tok.SPECIAL["<bos>"]
    assert alternate == [1] + tok.encode(
        sft.compact_prompt(row["task"] + " Declare every variable before use."))
    assert canonical != alternate
    with pytest.raises(ValueError):
        sft.java_eval_prefix(row, tok, "unregistered")

    calls = []
    def javac_only(source, root, timeout):
        calls.append((source.name, timeout))
        return {"status": "PASS", "diagnostics": ""}
    monkeypatch.setattr(sft, "_compile", javac_only)
    model = Model()
    report = sft.heldout_compilation(
        model, tok, [row], max_cases=1, prompt_style="declare_variables")
    assert model.last_prefix == alternate
    assert report["prompt_style"] == "declare_variables"
    assert report["compiled"] == 1
    assert report["behavior"] == "NOT_RUN"
    assert report["cases"][0]["java_behavior"] == "NOT_RUN"
    assert calls == [("Main.java", 12)]


def test_case_window_is_disjoint_and_deterministic(monkeypatch):
    monkeypatch.setattr(sft, "_compile", lambda *a, **k: {
        "status": "PASS", "diagnostics": ""})
    rows = _rows()
    tok = Tokenizer()
    model = Model()
    order = sft.eval_sample(rows, 3, seed=67)
    first = sft.heldout_compilation(model, tok, rows, max_cases=1)
    second = sft.heldout_compilation(
        model, tok, rows, max_cases=1, case_offset=1)
    assert first["cases"][0]["id"] == order[0]["id"]
    assert second["cases"][0]["id"] == order[1]["id"]
    assert first["cases"][0]["id"] != second["cases"][0]["id"]
    assert second["case_offset"] == 1
    with pytest.raises(ValueError):
        sft.heldout_compilation(model, tok, rows, case_offset=-1)
    with pytest.raises(ValueError):
        sft.heldout_compilation(model, tok, rows, max_cases=0)
    with pytest.raises(ValueError):
        sft.heldout_compilation(
            model, tok, rows, max_cases=1, prompt_style="unregistered")


def test_reference_offset_tracks_same_seed_without_java_execution(monkeypatch):
    rows = _rows()
    sources = []
    def javac_only(source, root, timeout):
        sources.append(source.read_text(encoding="utf-8"))
        return {"status": "PASS", "diagnostics": ""}
    monkeypatch.setattr(sft, "_compile", javac_only)
    report = sft.compile_heldout_references(rows, max_cases=1, case_offset=2)
    assert report["cases"][0]["id"] == sft.eval_sample(rows, 3, seed=67)[2]["id"]
    assert sources == [sft.eval_sample(rows, 3, seed=67)[2]["source"]]
    assert report["compiled"] == 1
    assert report["behavior"] == "NOT_RUN"
    assert report["used_for_model_selection"] is False
    with pytest.raises(ValueError):
        sft.compile_heldout_references(rows, max_cases=0)
    with pytest.raises(ValueError):
        sft.compile_heldout_references(rows, case_offset=-1)
