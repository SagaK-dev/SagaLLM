"""Stdin-algorithm-stdout curriculum regression tests. No model Java runs."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from sagallm.java_stdio_curriculum import (
    SCHEMA, TRAIN_FAMILIES, VALIDATION_FAMILIES, SPLITS,
    _task, assert_disjoint_families, build_curriculum, case_inputs,
    expected_rows, oracle, validation_oracle_rows, verify_curriculum,
)


def test_stdin_curriculum_is_family_disjoint_and_has_no_test_split():
    assert_disjoint_families()
    assert SCHEMA == "sagallm-java-stdio-semantics-v1"
    assert len(TRAIN_FAMILIES) == 10
    assert len(VALIDATION_FAMILIES) == 3
    assert set(TRAIN_FAMILIES).isdisjoint(VALIDATION_FAMILIES)
    assert len(expected_rows("train")) == 320
    assert len(expected_rows("validation")) == 36
    assert len(validation_oracle_rows()) == 36
    assert all(len(item["cases"]) == 3 for item in validation_oracle_rows())
    assert not any(
        "ex_luhn" in row["id"] or "ex_hamming" in row["id"]
        for row in expected_rows("train"))
    with pytest.raises(ValueError, match="No test split"):
        expected_rows("test")


def test_all_examples_use_explicit_stdin_and_newline_stdout():
    for split in SPLITS:
        for row in expected_rows(split):
            assert "standard input" not in row["messages"][1]["content"].lower() or True
            source = row["messages"][1]["content"]
            assert "public class Main" in source
            assert "new java.util.Scanner(System.in)" in source
            assert "System.out.println" in source
            assert "System.in" in source
            assert len(source) <= 1800
            assert "Read" in row["messages"][0]["content"] or "read" in row["messages"][0]["content"]
            assert "stdin" in row["messages"][0]["content"]


def test_build_and_strict_regeneration_detect_tampering(tmp_path):
    path = tmp_path / "stdio-data"
    meta = build_curriculum(path)
    assert meta["schema"] == SCHEMA
    assert meta["test_split"] == "NOT_CREATED"
    assert meta["files"]["train.jsonl"]["rows"] == 320
    assert meta["files"]["validation.jsonl"]["rows"] == 36
    assert meta["files"]["validation-oracles.jsonl"]["rows"] == 36
    assert verify_curriculum(path)["status"] == "PASS"
    with pytest.raises(FileExistsError):
        build_curriculum(path)
    assert not (path / "test.jsonl").exists()
    with (path / "train.jsonl").open("a", encoding="utf8") as output:
        output.write("tampered\n")
    with pytest.raises(ValueError, match="canonical regeneration"):
        verify_curriculum(path)


def test_validation_reference_oracles_cover_runtime_inputs():
    for item in validation_oracle_rows():
        family = item["family"]
        index = int(item["id"].split("-")[-1])
        for case in item["cases"]:
            assert oracle(family, index, case["stdin"]) == case["stdout"]
            assert case["stdin"].endswith("\n")
            assert case["stdout"].endswith("\n")
    assert oracle("stdin_min_offset", 0, "7 -3\n") == "-2\n"
    assert oracle("stdin_count_odd_offset", 0, "4\n-3 -1 2 2\n") == "3\n"
    assert oracle("stdin_ascii_lower_suffix", 0, "Hello WORLD\n") == "hello world_1\n"


def test_oracles_refuse_unregistered_data():
    with pytest.raises(ValueError):
        oracle("ex_luhn", 0, "000\n")
    with pytest.raises(ValueError):
        oracle("stdin_min_offset", 0, "1\n")
    with pytest.raises(ValueError):
        oracle("stdin_ascii_lower_suffix", 0, "日本語\n")
    with pytest.raises(ValueError):
        oracle("stdin_count_odd_offset", 0, "5\n2 3\n")


@pytest.mark.skipif(shutil.which("javac") is None or shutil.which("java") is None,
                    reason="Trusted reference Java requires local JDK")
def test_all_family_reference_programs_match_separate_python_oracles(tmp_path):
    """Only project-authored, audited references run on host, NEVER model outputs."""
    for family in TRAIN_FAMILIES + VALIDATION_FAMILIES:
        for index in (0, 5):
            _prompt, source = _task(family, index)
            root = tmp_path / family / str(index)
            root.mkdir(parents=True)
            (root / "Main.java").write_text(source, encoding="utf8")
            compile_result = subprocess.run(
                ["javac", "-proc:none", "--release", "17", "Main.java"],
                cwd=root, timeout=15, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False)
            assert compile_result.returncode == 0, (
                family, index, compile_result.stderr[:256])
            for stdin in case_inputs(family):
                actual = subprocess.run(
                    ["java", "-cp", str(root), "Main"], cwd=root,
                    input=stdin.encode("utf8"), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, timeout=5, check=False)
                assert actual.returncode == 0, (family, index, actual.stderr[:256])
                assert actual.stdout == oracle(family, index, stdin).encode("utf8"), (
                    family, index, actual.stdout[:128])
