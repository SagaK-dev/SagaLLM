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
    assert case["javac_error_category"] == "OTHER"
    assert case["javac_symbol_kind"] == "NONE"
    assert case["javac_variable_declaration_hint"] == "NONE"
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
    assert report["cases"][0]["javac_symbol_kind"] == "NONE"


def test_javac_error_classifier_allowlist_and_unknown():
    cases = {
        "Main.java:1: error: cannot find symbol\\n  symbol: variable secret": "MISSING_SYMBOL",
        "Main.java:1: error: incompatible types: String cannot be converted to int": "TYPE_ERROR",
        "Main.java:1: error: missing return statement": "MISSING_RETURN",
        "Main.java:1: error: ';' expected": "SYNTAX_ERROR",
        "Main.java:1: error: proprietary detail /home/user/secret.java": "OTHER",
        "Main.java:1: エラー: シンボルが見つかりません": "OTHER",
        "": "NONE",
    }
    for stderr, expected in cases.items():
        assert sft.classify_javac_diagnostic(stderr) == expected


def test_compilation_report_never_persists_raw_diagnostic(monkeypatch):
    secret = "/home/private/VerySensitive.java"
    stderr = "Main.java:1: error: cannot find symbol\\n" + secret + ("X" * 20000)
    monkeypatch.setattr(sft, "_compile", lambda *a, **k: {
        "status": "FAIL", "diagnostics": stderr, "reason": "JAVAC"
    })
    rows = [{"id": "case-3", "family": "family", "prefix_ids": [1, 2],
             "source": "reference source must not leak"}]
    report = sft.heldout_compilation(FakeModel(), FakeTokenizer(), rows, max_cases=1)
    case = report["cases"][0]
    assert case["javac_error_category"] == "MISSING_SYMBOL"
    assert case["source_persisted"] is False
    import json
    encoded = json.dumps(report)
    assert secret not in encoded
    assert "reference source must not leak" not in encoded
    assert "cannot find symbol" not in encoded
    assert "X" * 100 not in encoded


def test_javac_symbol_kind_records_kind_not_identifier():
    kinds = {
        "variable": "VARIABLE",
        "method": "METHOD",
        "class": "CLASS",
        "constructor": "CONSTRUCTOR",
    }
    for raw, expected in kinds.items():
        stderr = ("Main.java:1: error: cannot find symbol\n"
                  "   System.out.println(privateSecret);\n"
                  f"  symbol:   {raw} confidentialName\n"
                  "  location: class Main\n")
        assert sft.classify_javac_symbol_kind(stderr) == expected
    assert sft.classify_javac_symbol_kind(
        "Main.java:1: error: cannot find symbol\n no symbol marker") == "UNKNOWN"
    assert sft.classify_javac_symbol_kind(
        "Main.java:1: error: incompatible types\n  symbol: variable hideMe") == "NONE"
    assert sft.classify_javac_symbol_kind(
        "Main.java:1: error: cannot find symbol\n"
        "Main.java:2: error: other\n  symbol: variable hideMe") == "UNKNOWN"


def test_compilation_report_exposes_only_javac_symbol_kind(monkeypatch):
    secret = "privateSecret"
    diagnostics = ("Main.java:1: error: cannot find symbol\n"
                   f"  symbol:   method {secret}()\n"
                   "  location: class Main\n")
    monkeypatch.setattr(sft, "_compile", lambda *args, **kwargs: {
        "status": "FAIL", "diagnostics": diagnostics, "reason": "JAVAC"
    })
    rows = [{"id": "case-symbol", "family": "family", "prefix_ids": [1, 2],
             "source": "private reference source"}]
    report = sft.heldout_compilation(FakeModel(), FakeTokenizer(), rows, max_cases=1)
    case = report["cases"][0]
    assert case["javac_error_category"] == "MISSING_SYMBOL"
    assert case["javac_symbol_kind"] == "METHOD"
    import json
    encoded = json.dumps(report)
    assert secret not in encoded
    assert "private reference source" not in encoded
    assert "symbol:" not in encoded
    assert case["source_persisted"] is False



def test_missing_variable_declaration_hint_is_bounded_and_lexical_only():
    stderr = ("Main.java:4: error: cannot find symbol\n"
              " System.out.println(hiddenName);\n"
              "                    ^\n"
              "  symbol:   variable hiddenName\n"
              "  location: class Main\n")
    no_declaration = ("public class Main { void foo() {"
                      "System.out.println(hiddenName); } }")
    simple_declaration = ("public class Main { void foo() {"
                          "int hiddenName = 1; System.out.println(hiddenName); } }")
    array_declaration = ("public class Main { void foo() {"
                         "int[] hiddenName = new int[1]; } }")
    generic_declaration = ("public class Main { void foo() {"
                           "List<String> hiddenName = null; } }")
    assert sft.unresolved_variable_declaration_hint(stderr, no_declaration) == (
        "NO_SIMPLE_DECLARATION_PATTERN_SEEN")
    for source in (simple_declaration, array_declaration, generic_declaration):
        assert sft.unresolved_variable_declaration_hint(stderr, source) == (
            "SIMPLE_DECLARATION_PATTERN_SEEN")
    # A custom declaration is outside the intentionally limited type allowlist.
    assert sft.unresolved_variable_declaration_hint(
        stderr, "public class Main { Custom hiddenName; }") == (
        "NO_SIMPLE_DECLARATION_PATTERN_SEEN")
    assert sft.unresolved_variable_declaration_hint(
        "Main.java:1: error: cannot find symbol\n  symbol: variable confidential.name",
        simple_declaration) == "UNKNOWN"
    assert sft.unresolved_variable_declaration_hint(
        "Main.java:1: error: another error\n  symbol: variable hiddenName",
        simple_declaration) == "UNKNOWN"
    assert sft.unresolved_variable_declaration_hint(
        "Main.java:1: error: cannot find symbol\n"
        "Main.java:2: error: other\n  symbol: variable hiddenName",
        simple_declaration) == "UNKNOWN"
    assert sft.unresolved_variable_declaration_hint(
        None, simple_declaration) == "UNKNOWN"


def test_generated_variable_hint_does_not_leak_identifier(monkeypatch):
    secret = "nonPublicSecret"
    diagnostics = ("Main.java:1: error: cannot find symbol\n"
                   f"  symbol: variable {secret}\n"
                   "  location: class Main\n")
    class SourceTokenizer(FakeTokenizer):
        def decode(self, ids):
            return ("public class Main { public static void main(String[] a)"
                    f" {{ int {secret} = 3; System.out.println({secret}); }} }}")

    monkeypatch.setattr(sft, "_compile", lambda *args, **kwargs: {
        "status": "FAIL", "diagnostics": diagnostics, "reason": "JAVAC"
    })
    rows = [{"id": "case-private", "family": "family", "prefix_ids": [1, 2],
             "source": "private reference source"}]
    report = sft.heldout_compilation(FakeModel(), SourceTokenizer(), rows,
                                     max_cases=1)
    case = report["cases"][0]
    assert case["javac_symbol_kind"] == "VARIABLE"
    assert case["javac_variable_declaration_hint"] == (
        "SIMPLE_DECLARATION_PATTERN_SEEN")
    assert case["java_behavior"] == "NOT_RUN"
    assert case["source_persisted"] is False
    import json
    rendered = json.dumps(report)
    assert secret not in rendered
    assert "symbol:" not in rendered
    assert "private reference source" not in rendered


def test_reference_compilation_is_separate_and_does_not_execute(monkeypatch):
    reference_a = "public class Main { int markerA; }"
    reference_b = "public class Main { broken markerB; }"
    calls = []
    def compile_only(source_file, root, timeout):
        code = source_file.read_text(encoding="utf8")
        calls.append((source_file.name, timeout))
        if "markerB" in code:
            return {"status": "FAIL", "reason": "JAVAC",
                    "diagnostics": "Main.java:1: error: ';' expected"}
        return {"status": "PASS", "diagnostics": ""}
    monkeypatch.setattr(sft, "_compile", compile_only)
    rows = [{"id": "a", "family": "one", "source": reference_a},
            {"id": "b", "family": "two", "source": reference_b}]
    report = sft.compile_heldout_references(rows, max_cases=2)
    assert report["evaluated"] == 2
    assert report["compiled"] == 1
    assert report["behavior"] == "NOT_RUN"
    assert report["used_for_model_selection"] is False
    assert len(calls) == 2
    assert set(calls) == {("Main.java", 12)}
    assert sorted(case["javac_error_category"] for case in report["cases"]) == (
        ["NONE", "SYNTAX_ERROR"])
    assert all(case["source_persisted"] is False for case in report["cases"])
    import json
    rendered = json.dumps(report)
    assert "markerA" not in rendered and "markerB" not in rendered
    assert "broken" not in rendered
