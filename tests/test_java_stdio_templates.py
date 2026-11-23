"""Tests of the opt-in reliable NON-LLM Java standard-input generator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from sagallm.java_stdio_templates import (
    OPERATIONS, create_stdio_program, render_stdio,
)


PARAMS = {
    "sum-two": {"offset": 3},
    "product-two": {"offset": 3},
    "min-two": {"offset": 3},
    "max-two": {"offset": 3},
    "absolute-gap": {"offset": 3},
    "count-greater": {"threshold": 2},
    "sum-even": {"offset": 3},
    "count-odd": {"offset": 3},
    "digit-sum": {"offset": 3},
    "upper-line": {"suffix": "_ok"},
    "lower-line": {"suffix": "_ok"},
    "reverse-line": {"suffix": "_ok"},
    "clamp": {"lower_bound": -2, "upper_bound": 2},
}


def _cases(op: str):
    pair = ("7 -3\n", "-5 2\n", "0 0\n")
    if op == "sum-two":
        return list(zip(pair, ("7\n", "0\n", "3\n")))
    if op == "product-two":
        return list(zip(pair, ("-18\n", "-7\n", "3\n")))
    if op == "min-two":
        return list(zip(pair, ("0\n", "-2\n", "3\n")))
    if op == "max-two":
        return list(zip(pair, ("10\n", "5\n", "3\n")))
    if op == "absolute-gap":
        return list(zip(pair, ("13\n", "10\n", "3\n")))
    if op in {"count-greater", "sum-even", "count-odd"}:
        inp = ("5\n3 -2 0 7 4\n", "0\n", "4\n-3 -1 2 2\n")
        expected = {
            "count-greater": ("3\n", "0\n", "0\n"),
            "sum-even": ("5\n", "3\n", "7\n"),
            "count-odd": ("5\n", "3\n", "5\n"),
        }[op]
        return list(zip(inp, expected))
    if op == "digit-sum":
        return [("0\n", "3\n"), ("1005\n", "9\n"), ("789\n", "27\n")]
    if op in {"upper-line", "lower-line", "reverse-line"}:
        inputs = ["MiXeD Case", "  Spaces  ", ""]
        transform = {
            "upper-line": str.upper,
            "lower-line": str.lower,
            "reverse-line": lambda value: value[::-1],
        }[op]
        return [(line + "\n", transform(line) + "_ok\n") for line in inputs]
    if op == "clamp":
        return [("-30\n", "-2\n"), ("0\n", "0\n"), ("30\n", "2\n")]
    raise ValueError("No reviewed test cases")


def test_operations_have_explicit_contract_and_no_model_claim():
    assert len(OPERATIONS) == 13
    assert set(OPERATIONS) == set(PARAMS)
    for operation in OPERATIONS:
        output = render_stdio(operation, **PARAMS[operation])
        assert output.startswith("public class Main {")
        assert "new java.util.Scanner(System.in" in output
        assert "System.out.println" in output
        assert "Enter" not in output
        assert "public static void main" in output
        assert len(output) < 2000


def test_strict_options_never_silently_ignored():
    with pytest.raises(ValueError, match="Unsupported"):
        render_stdio("ex_luhn")
    with pytest.raises(ValueError, match="Unused offset"):
        render_stdio("count-greater", offset=1)
    with pytest.raises(ValueError, match="Unused threshold"):
        render_stdio("sum-two", threshold=2)
    with pytest.raises(ValueError, match="Unused clamp"):
        render_stdio("sum-two", lower_bound=3)
    with pytest.raises(ValueError, match="only supported for line"):
        render_stdio("sum-two", suffix="a")
    with pytest.raises(ValueError, match="Invalid clamp bounds"):
        render_stdio("clamp", lower_bound=10, upper_bound=-10)
    with pytest.raises(ValueError, match="unsafe ASCII suffix"):
        render_stdio("upper-line", suffix='";Runtime.getRuntime().exec("x")')
    with pytest.raises(ValueError, match="integer"):
        render_stdio("sum-two", offset=True)
    with pytest.raises(ValueError, match="integer"):
        render_stdio("sum-two", offset=10001)
    with pytest.raises(ValueError):
        render_stdio("sum-two", class_name="Main; System.exit(1);")


def test_cli_writes_new_project_and_preserves_manifest(tmp_path, capsys):
    from sagallm.cli import main
    directory = tmp_path / "example"
    main(["java", "stdio", "--operation", "sum-two",
          "--output", str(directory), "--offset", "3"])
    console = capsys.readouterr().out
    assert '"DETERMINISTIC_STDIO_TEMPLATE_NOT_LLM"' in console
    source = (directory / "Main.java").read_text(encoding="utf8")
    doc = json.loads((directory / "manifest.json").read_text(encoding="utf8"))
    assert doc["model_used"] is False
    assert doc["java_executed"] is False
    assert doc["status"] == "CREATED"
    assert doc["operation"] == "sum-two"
    assert doc["stdout"] == "EXACT_NEWLINE_TERMINATED_NO_EXTRA_PROMPTS"
    assert doc["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    with pytest.raises(FileExistsError):
        create_stdio_program(directory, "min-two")


@pytest.mark.skipif(not shutil.which("javac") or not shutil.which("java"),
                    reason="Trusted deterministic Java requires JDK")
def test_thirteen_trusted_programs_all_match_39_input_output_examples(tmp_path):
    """Only deterministic project-authored code executes locally; NOT model output."""
    for op in OPERATIONS:
        root = tmp_path / op
        info = create_stdio_program(root, op, **PARAMS[op])
        assert info["model_used"] is False
        assert info["java_executed"] is False
        compiled = subprocess.run(
            ["javac", "-proc:none", "--release", "17", "Main.java"],
            cwd=root, capture_output=True, timeout=15, check=False)
        assert compiled.returncode == 0, (op, compiled.stderr[:320])
        cases = _cases(op)
        assert len(cases) == 3
        for stdin, expected in cases:
            finished = subprocess.run(
                ["java", "-cp", str(root), "Main"], cwd=root,
                input=stdin.encode("utf8"), capture_output=True,
                timeout=5, check=False)
            assert finished.returncode == 0, (op, finished.stderr[:320])
            assert finished.stdout == expected.encode("utf8"), (
                op, stdin, finished.stdout, expected)


def test_custom_class_name_and_no_extra_message():
    source = render_stdio("min-two", class_name="Result", offset=2)
    assert "public class Result" in source
    assert "System.out.println" in source
    assert "Enter an integer" not in source
    assert "import java." not in source
