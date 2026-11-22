"""Train-only identifier variants: deterministic, split-safe, compile-only."""
from __future__ import annotations

import json
import shutil

import pytest

from sagallm import java_identifier_curriculum as curriculum
from sagallm.java_instruction_sft import source_task


def test_identifier_rename_ignores_literals_and_comments():
    before = (
        'public class Main {\n'
        '  // x is a note, not a variable\n'
        '  /* y is in a block comment */\n'
        '  int x=1,y=2;\n'
        '  String s="x and y"; char c=\'x\';\n'
        '  void f(){System.out.println(x+y);}\n'
        '}\n'
    )
    after = curriculum.rename_identifiers(
        before, {"x": "currentValue", "y": "trackedTotal"})
    assert "int currentValue=1,trackedTotal=2;" in after
    assert "println(currentValue+trackedTotal)" in after
    assert "// x is a note" in after
    assert "/* y is in a block comment */" in after
    assert '"x and y"' in after
    assert "char c='x'" in after
    assert after.count("currentValue") == 2
    assert after.count("trackedTotal") == 2


def test_identifier_rename_rejects_ambiguous_and_unused_mapping():
    before = "public class Main { int x=1; System.out.println(x); }"
    with pytest.raises(ValueError):
        curriculum.rename_identifiers(before, {"x": "Main"})
    with pytest.raises(ValueError):
        curriculum.rename_identifiers(before, {"x": "firstOperand",
                                              "y": "secondOperand"})
    with pytest.raises(ValueError):
        curriculum.rename_identifiers(before, {"x": "int"})
    with pytest.raises(ValueError):
        curriculum.rename_identifiers(before + ' String x="""hello""";',
                                      {"x": "firstOperand"})


def test_candidate_generation_is_deterministic_and_train_only(tmp_path):
    target = tmp_path / "candidates"
    created = curriculum.build_identifier_train_candidates(target)
    verified = curriculum.verify_identifier_train_candidates(target)
    assert created["examples"] == 120
    assert len(created["families"]) == 10
    assert created["train_only"] is True
    assert created["canonical_data_modified"] is False
    assert created["model_training_performed"] is False
    assert verified["status"] == "PASS"

    lines = (target / created["candidate_file"]).read_text(
        encoding="utf8").splitlines()
    assert len(lines) == 120
    rows = [json.loads(line) for line in lines]
    assert len({row["id"] for row in rows}) == 120
    assert {row["family"] for row in rows} == set(curriculum.VARIABLES)
    for row in rows:
        task, source = source_task(row)
        assert task and "public class Main" in source
        assert row["family"] in (set(curriculum.NEW_TRAIN) |
                                 set(curriculum.TRAIN_FAMILIES))
        assert row["family"] not in (set(curriculum.NEW_TEST) |
                                     set(curriculum.NEW_VALIDATION) |
                                     set(curriculum.TEST_FAMILIES) |
                                     set(curriculum.VALIDATION_FAMILIES))
    with pytest.raises(FileExistsError):
        curriculum.build_identifier_train_candidates(target)
    target_2 = tmp_path / "again"
    curriculum.build_identifier_train_candidates(target_2)
    assert (target / created["candidate_file"]).read_bytes() == (
        target_2 / created["candidate_file"]).read_bytes()


def test_candidate_forgery_and_heldout_boundary_rejected(tmp_path, monkeypatch):
    folder = tmp_path / "variants"
    curriculum.build_identifier_train_candidates(folder, cases_per_family=1)
    file = folder / "train-identifier-variants.jsonl"
    file.write_bytes(file.read_bytes().replace(b"public class Main",
                                                b"public class Fake", 1))
    with pytest.raises(ValueError, match="canonical regeneration"):
        curriculum.verify_identifier_train_candidates(folder)

    monkeypatch.setitem(curriculum.VARIABLES, "collatz_max", ("x",))
    with pytest.raises(ValueError, match="non-train family|heldout"):
        curriculum.check_train_family_boundary()


def test_compiles_all_train_candidate_java_without_executing(tmp_path):
    if shutil.which("javac") is None:
        pytest.skip("Java compiler unavailable")
    from pathlib import Path
    from scripts.audit_java_identifier_curriculum import audit
    folder = tmp_path / "compile-audit"
    report = audit(folder, cases_per_family=2, variants=2)
    assert report["candidate_examples"] == 40
    assert report["compiler"]["status"] == "PASS"
    assert report["compiler"]["compiled"] == 40
    assert report["compiler"]["java_executed"] is False
    assert report["generated_java_executed"] is False
    assert report["validation_or_test_tasks_used_for_augmentation"] is False
    assert report["used_for_training"] is False
    assert report["new_identifier_vocabulary_used"] > 0
    assert not (folder / "_compile_only").exists()
    recorded = json.loads((folder / "audit-summary.json").read_text())
    assert recorded == report
