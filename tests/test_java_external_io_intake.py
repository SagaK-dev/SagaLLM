"""External Java I/O intake checks are never behavioral model tests."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from sagallm.java_external_io_intake import (
    _adapt, _blob_sha1, LICENSE_BLOB_SHA1, PINNED_REVISION,
    TARGET_FAMILIES, verify_intake,
)
from sagallm.java_independent_benchmark import validate_suite

ROOT = Path(__file__).resolve().parents[1]


def _copy_packet(target: Path) -> Path:
    for rel in (
        "examples/java_external_io_INTAKE.json",
        "examples/java_external_io_sources.json",
        "third_party_notices/exercism-problem-specifications-LICENSE.txt",
    ):
        to = target / rel
        to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, to)
    return target


def test_six_source_candidates_remain_unreviewed_and_nonexecuting():
    report = verify_intake(ROOT)
    assert report["status"] == "INTAKE_STRUCTURE_ONLY"
    assert report["source_revision"] == PINNED_REVISION
    assert report["upstream_git_blobs_checked"] == 0
    assert report["license_file_git_blob_checked"] is True
    assert report["exercise_candidates"] == 6
    assert report["io_cases"] == 27
    assert report["exact_family_label_overlap_count"] == 0
    assert report["review_gate"] == "REVIEW_REQUIRED"
    assert report["semantic_overlap_review"] == "PENDING_HUMAN_REVIEW"
    assert report["exercise_level_rights_review"] == "PENDING_HUMAN_REVIEW"
    assert report["assistant_java_io_adaptation_review"] == "PENDING_HUMAN_REVIEW"
    assert report["human_reviewers_verified"] == 0
    assert report["behavior"] == "NOT_RUN"
    assert report["model_outputs_evaluated"] is False
    assert report["independence_proven"] is False
    assert report["model_selection_permitted"] is False

    suite = json.loads((ROOT/"examples/java_external_io_INTAKE.json").read_text())
    assert len(suite["tasks"]) == 6
    assert validate_suite(suite)["review_gate"] == "REVIEW_REQUIRED"
    assert set(item["family"] for item in suite["tasks"]) == {
        fam for _, fam, _ in TARGET_FAMILIES.values()
    }


def test_external_packet_license_and_blob_identity():
    raw = (ROOT / "third_party_notices/exercism-problem-specifications-LICENSE.txt").read_bytes()
    assert _blob_sha1(raw) == LICENSE_BLOB_SHA1
    assert b"The MIT License" in raw
    assert b"Copyright (c) 2014, 2019, 2021 Exercism" in raw
    assert _blob_sha1(b"hi") == "32f6e19b3b9b67e973d8a3083f3f9863f034b7a8"


def test_known_canonical_case_adapters_are_explicit_and_bounded():
    assert _adapt("hamming", {
        "input": {"strand1": "G", "strand2": "T"}, "expected": 1
    }) == {"stdin": "G\nT\n", "stdout": "1\n"}
    assert _adapt("rna-transcription", {
        "input": {"dna": "ACGT"}, "expected": "UGCA"
    }) == {"stdin": "ACGT\n", "stdout": "UGCA\n"}
    assert _adapt("luhn", {
        "input": {"value": "059"}, "expected": True
    }) == {"stdin": "059\n", "stdout": "true\n"}
    assert _adapt("roman-numerals", {
        "input": {"number": 4}, "expected": "IV"
    }) == {"stdin": "4\n", "stdout": "IV\n"}
    assert _adapt("allergies", {
        "input": {"score": 3, "item": "eggs"}, "expected": True
    }) == {"stdin": "3 eggs\n", "stdout": "true\n"}
    assert _adapt("atbash-cipher", {
        "input": {"phrase": "yes"}, "expected": "bvh"
    }) == {"stdin": "yes\n", "stdout": "bvh\n"}

    with pytest.raises(ValueError):
        _adapt("hamming", {"input": {"strand1": "A", "strand2": ""},
                           "expected": {"error": "different lengths"}})
    with pytest.raises(ValueError):
        _adapt("allergies", {"input": {"score": 3}, "expected": ["eggs"]})
    with pytest.raises(ValueError):
        _adapt("unknown", {"input": {}, "expected": None})


def test_rejects_fake_review_metadata_and_provenance_changes(tmp_path):
    root = _copy_packet(tmp_path)
    suite_path = root / "examples/java_external_io_INTAKE.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["review_status"] = "APPROVED_FOR_INDEPENDENT_EVALUATION"
    suite_path.write_text(json.dumps(suite), encoding="utf-8")
    with pytest.raises(ValueError, match="falsely promoted"):
        verify_intake(root)

    root = _copy_packet(tmp_path / "second")
    meta_path = root / "examples/java_external_io_sources.json"
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest["source_revision"] = "main"
    meta_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unpinned"):
        verify_intake(root)


def test_upstream_files_are_required_for_strong_validation(tmp_path):
    root = _copy_packet(tmp_path / "packet")
    upstream = tmp_path / "source-repo"
    upstream.mkdir()
    with pytest.raises(ValueError, match="Missing"):
        verify_intake(root, upstream_root=upstream)

    meta_path = root / "examples/java_external_io_sources.json"
    manifest = json.loads(meta_path.read_text(encoding="utf-8"))
    manifest["sources"][0]["selected_case_uuids"][0] = "not-a-valid-uuid"
    meta_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="UUID"):
        verify_intake(root)


def test_symlinked_intake_files_are_rejected(tmp_path):
    root = _copy_packet(tmp_path / "packet")
    path = root / "examples/java_external_io_INTAKE.json"
    saved = tmp_path / "saved.json"
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises(ValueError, match="unsafe"):
        verify_intake(root)
