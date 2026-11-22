"""Offline Java data integrity and *compile-only* coding evaluation tests."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from sagallm.java_coding_eval import (
    BASE_SOURCE, SPLIT_FAMILIES, TASKS,
    build_coding_fixture, evaluate_java_compilation, verify_coding_fixture,
)
from sagallm.java_vibe import render

JDK = shutil.which("javac")


def make(tmp_path):
    root = tmp_path / "fixture"
    report = build_coding_fixture(root)
    return root, report


def test_create_and_verify_family_separated_fixture(tmp_path):
    root, result = make(tmp_path)
    assert result["counts"] == {"train": 10, "validation": 1, "test": 2}
    audit = verify_coding_fixture(root)
    assert audit["status"] == "PASS" and audit["split_integrity"] == "PASS"
    assert not (set(SPLIT_FAMILIES["train"]) & set(SPLIT_FAMILIES["test"]))
    assert audit["dataset_license_review"] == "NOT_ATTESTED"
    assert not (root / "model.pt").exists()


def test_fixture_deterministic_across_paths(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    build_coding_fixture(a)
    build_coding_fixture(b)
    assert [(a / f).read_bytes() for f in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")] == [
        (b / f).read_bytes() for f in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl")]


def test_no_overwrite(tmp_path):
    root, _ = make(tmp_path)
    with pytest.raises(FileExistsError):
        build_coding_fixture(root)
    assert verify_coding_fixture(root)["status"] == "PASS"


def test_sha256_detects_file_change(tmp_path):
    root, _ = make(tmp_path)
    with (root / "train.jsonl").open("a", encoding="utf-8") as out:
        out.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_coding_fixture(root)


def test_tamper_after_rehash_is_still_rejected(tmp_path):
    root, _ = make(tmp_path)
    file = root / "test.jsonl"
    rows = [json.loads(x) for x in file.read_text().splitlines()]
    rows[0]["messages"][1]["content"] = "public class Main {}\n"
    raw = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    file.write_text(raw)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["split_files"]["test"]["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unexpected instruction/answer"):
        verify_coding_fixture(root)


def test_manifest_rejects_unexpected_filename(tmp_path):
    root, _ = make(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["split_files"]["test"]["file"] = "../evil.jsonl"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Split inventory mismatch"):
        verify_coding_fixture(root)


def test_manifest_rejects_symlink(tmp_path):
    root, _ = make(tmp_path)
    dest = root / "validation.jsonl"
    extra = tmp_path / "copy.jsonl"
    extra.write_bytes(dest.read_bytes())
    dest.unlink()
    dest.symlink_to(extra)
    with pytest.raises(ValueError, match="unsafe"):
        verify_coding_fixture(root)


def test_missing_family_detected_even_if_rehashed(tmp_path):
    root, _ = make(tmp_path)
    target = root / "test.jsonl"
    lines = target.read_text().splitlines()
    target.write_text(lines[0] + "\n")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["split_files"]["test"]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["split_files"]["test"]["examples"] = 1
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="missing expected tasks"):
        verify_coding_fixture(root)


@pytest.mark.skipif(not JDK, reason="JDK missing; compile-only protocol not verified")
def test_compile_only_heldout_reference_generation(tmp_path):
    root, _ = make(tmp_path)
    def fake_generate(prompt, *, max_new_tokens):
        if "factorial" in prompt.lower():
            return render("factorial", "Main")
        return render("calculator", "Main")
    result = evaluate_java_compilation(root, generator=fake_generate)
    assert result["compile_passes"] == 2
    assert result["exact_reference_matches"] == 2
    assert result["runtime_behavior_test"] == "NOT_RUN"
    assert result["compile_pass_rate"] == 1
    assert result["comparability"].startswith("SYNTHETIC_")
    assert all("source" not in row for row in result["items"])


@pytest.mark.skipif(not JDK, reason="JDK missing")
def test_compiles_different_valid_code_but_not_reference(tmp_path):
    root, _ = make(tmp_path)
    result = evaluate_java_compilation(root, generator=lambda p, max_new_tokens: (
        'public class Main { public static void main(String[] args) { System.out.println("not the answer"); } }\n'),
        max_examples=1)
    assert result["compile_passes"] == 1
    assert result["exact_reference_matches"] == 0
    assert result["runtime_behavior_test"] == "NOT_RUN"


def test_fails_closed_on_broken_model_output(tmp_path):
    root, _ = make(tmp_path)
    result = evaluate_java_compilation(root, generator=lambda p, max_new_tokens: "wrong")
    assert result["compile_passes"] == 0
    assert all(row["status"] == "REJECTED" for row in result["items"])
    assert all(row["runtime_tests"] == "NOT_RUN" for row in result["items"])


def test_noop_does_not_count_as_model_success(tmp_path):
    root, _ = make(tmp_path)
    result = evaluate_java_compilation(root, generator=lambda p, max_new_tokens: BASE_SOURCE)
    assert result["compile_passes"] == 0
    assert all(row["status"] == "UNCHANGED" for row in result["items"])


def test_rejects_train_as_evaluation(tmp_path):
    root, _ = make(tmp_path)
    with pytest.raises(ValueError, match="held-out"):
        evaluate_java_compilation(root, split="train", generator=lambda p, max_new_tokens: "")


def test_rejects_excessive_eval_budget(tmp_path):
    root, _ = make(tmp_path)
    with pytest.raises(ValueError, match="1..20"):
        evaluate_java_compilation(root, max_examples=100, generator=lambda p, max_new_tokens: "")


def test_real_tiny_cpu_sft_smoke_reports_no_claim_of_java_competence():
    # Real random-weight initialization, a two-step optimizer update and
    # held-out token losses; no generated Java is executed.
    from scripts.java_sft_smoke import run
    report = run()
    assert report["status"] == "PASS_RUNTIME"
    assert report["train_examples"] == 10
    assert report["sft_optimizer_steps"] == 2
    assert report["eval_total"] == 2
    assert report["runtime_behavior"] == "NOT_RUN"
    assert report["general_java_capability"] == "NOT_ESTABLISHED"
    for key in ("heldout_test_loss_before", "heldout_test_loss_after", "sft_final_training_loss"):
        assert 0 < report[key] < 100


def test_forged_license_review_cannot_be_claimed(tmp_path):
    root, _ = make(tmp_path)
    manifest_path = root / "manifest.json"
    record = json.loads(manifest_path.read_text())
    record["dataset_license_review"] = "APPROVED"
    manifest_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="does not certify"):
        verify_coding_fixture(root)