"""Regression checks for entirely original Java SFT examples and CPU training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from sagallm.java_generated_data import (
    TRAIN_FAMILIES, VALIDATION_FAMILIES, TEST_FAMILIES,
    build_java_dataset, verify_java_dataset,
)


def test_generated_corpus_counts_and_family_separation(tmp_path):
    dataset = tmp_path / "java"
    result = build_java_dataset(dataset)
    assert result["counts"] == {"train": 320, "validation": 15, "test": 15}
    report = verify_java_dataset(dataset)
    assert report["family_disjoint"]
    assert not report["license_attested"]
    assert not (set(TRAIN_FAMILIES) & (set(VALIDATION_FAMILIES) | set(TEST_FAMILIES)))
    assert not (set(VALIDATION_FAMILIES) & set(TEST_FAMILIES))
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_weights_used_to_create_examples"] is False
    for split in ("train", "validation", "test"):
        assert hashlib.sha256((dataset / (split + ".jsonl")).read_bytes()).hexdigest() == manifest["files"][split]["sha256"]


def test_new_data_never_overwrites_existing_directory(tmp_path):
    destination = tmp_path / "java"
    build_java_dataset(destination)
    with pytest.raises(FileExistsError):
        build_java_dataset(destination)


def test_modified_source_fails_even_if_manifest_digest_is_forged(tmp_path):
    dataset = tmp_path / "java"
    build_java_dataset(dataset)
    train = dataset / "train.jsonl"
    raw = train.read_text(encoding="utf-8").replace("System.out.println", "System.err.println", 1)
    train.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        verify_java_dataset(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["train"]["sha256"] = hashlib.sha256(train.read_bytes()).hexdigest()
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="content"):
        verify_java_dataset(dataset)


def test_review_claim_forgery_fails(tmp_path):
    dataset = tmp_path / "java"
    build_java_dataset(dataset)
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    manifest["dataset_license_review"] = "APPROVED"
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        verify_java_dataset(dataset)


def test_unsafe_symlink_refused(tmp_path):
    dataset = tmp_path / "java"
    build_java_dataset(dataset)
    (dataset / "test.jsonl").rename(dataset / "test-original.jsonl")
    (dataset / "test.jsonl").symlink_to(dataset / "test-original.jsonl")
    with pytest.raises(ValueError, match="Unexpected dataset"):
        verify_java_dataset(dataset)


@pytest.mark.skipif(shutil.which("javac") is None, reason="JDK is unavailable")
def test_original_java_references_compile_for_every_family(tmp_path):
    dataset = tmp_path / "java"
    build_java_dataset(dataset)
    families = set()
    for split in ("train", "validation", "test"):
        for line in (dataset / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["family"] in families:
                continue
            families.add(row["family"])
            work = tmp_path / "compile" / row["family"]
            work.mkdir(parents=True)
            java_file = work / "Main.java"
            java_file.write_text(row["messages"][1]["content"], encoding="utf-8")
            result = subprocess.run(["javac", "-proc:none", "-d", str(work), str(java_file)],
                                    capture_output=True, text=True, timeout=15)
            assert result.returncode == 0, (row["family"], result.stderr)
    assert len(families) == 26


def test_cpu_training_produces_checkpoint_without_using_test_answers(tmp_path):
    # Tiny real training loop; no Java generated source execution, no download.
    from scripts.train_java_original import train
    from sagallm.java_ai import LocalCheckpointGenerator
    dataset = tmp_path / "java"
    build_java_dataset(dataset)
    output = tmp_path / "trained"
    result = train(dataset, output, steps=3, max_seconds=20,
                   hidden_size=96, layers=2, progress_every=1, compile_examples=0)
    assert result["steps_completed"] == 3
    assert result["compile_only_heldout"]["status"] == "NOT_RUN"
    assert result["base_weights"] == "RANDOM_INIT_ONLY"
    assert result["parameter_count"] < 300_000
    assert (output / "java-trained.pt").exists()
    assert (output / "tokenizer.json").exists()
    assert result["test_examples"] == 15
    assert LocalCheckpointGenerator(output / "java-trained.pt", output / "tokenizer.json").max_context == 384