"""Human worklist must preserve six-task evidence and all pending gates."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts.build_java_external_review_packet import (
    EXPECTED_SUITE_SHA256, EXPECTED_PROVENANCE_SHA256, render,
)

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "examples/java_external_io_INTAKE.json",
    "examples/java_external_io_sources.json",
    "third_party_notices/exercism-problem-specifications-LICENSE.txt",
)
TASK_IDS = (
    "ex_hamming", "ex_rna_transcription", "ex_luhn",
    "ex_roman_numerals", "ex_allergies", "ex_atbash_cipher",
)


def _packet(dest: Path) -> Path:
    for rel in FILES:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)
    return dest


def test_all_six_review_packets_and_two_pending_human_slots():
    document = render(ROOT)
    assert "UNREVIEWED / REVIEW_REQUIRED / NOT_RUN" in document
    assert EXPECTED_SUITE_SHA256 in document
    assert EXPECTED_PROVENANCE_SHA256 in document
    assert document.count("### ex_") == 6
    assert document.count("Independent Reviewer A dated verdict + URL: **PENDING**") == 6
    assert document.count("Independent Reviewer B dated verdict + URL: **PENDING**") == 6
    assert document.count("Human legal/rights disposition: **PENDING**") == 6
    assert document.count("Human Java console adapter disposition: **PENDING**") == 6
    assert document.count("Human semantic contamination disposition: **PENDING**") == 6
    assert document.count("NOT APPROVED FOR MODEL SELECTION") == 6
    assert "END: REVIEW REQUIRED / NOT_RUN — 0 HUMAN SIGN-OFFS" in document
    for tid in TASK_IDS:
        assert "### " + tid in document
    assert "https://github.com/exercism/problem-specifications/blob/" in document
    assert "public Exercism" in document
    assert "0 HUMAN SIGN-OFFS" in document
    assert "No model code" not in document  # use precise claims, not false shortcuts
    assert document == render(ROOT)


def test_packet_detects_prompt_edit_that_could_bypass_basic_oracles(tmp_path):
    packet = _packet(tmp_path)
    suite_file = packet / FILES[0]
    suite = json.loads(suite_file.read_text(encoding="utf8"))
    suite["tasks"][0]["prompt"] += " (different prompt without human approval)"
    suite_file.write_text(json.dumps(suite), encoding="utf8")
    # Oracle checks alone do not pin task prose; the frozen SHA does.
    with pytest.raises(ValueError, match="approval gate moved"):
        render(packet)


def test_packet_detects_unreviewed_metadata_or_rights_changes(tmp_path):
    packet = _packet(tmp_path)
    manifest_file = packet / FILES[1]
    manifest = json.loads(manifest_file.read_text(encoding="utf8"))
    manifest["sources"][0]["metadata_source_url"] = "https://example.org/changed"
    manifest_file.write_text(json.dumps(manifest), encoding="utf8")
    with pytest.raises(ValueError, match="Provenance changed"):
        render(packet)


def test_output_does_not_embed_test_vectors_or_fake_reviews():
    document = render(ROOT)
    assert "055 444 285" not in document
    assert "GGACGGATTCTG" not in document
    assert "public class Main" not in document
    assert "https://example.org/review/1" not in document
    assert "ELIGIBLE_SELF_ATTESTED" not in document
    assert "benchmark_independence_proven=true" not in document


def test_script_refuses_to_overwrite_existing_worklist(tmp_path, monkeypatch):
    from scripts.build_java_external_review_packet import main
    result = tmp_path / "review-packet.md"
    monkeypatch.setattr("sys.argv", [
        "build_java_external_review_packet",
        "--root", str(ROOT), "--output", str(result),
    ])
    main()
    assert result.exists()
    assert result.read_text(encoding="utf8") == render(ROOT)
    with pytest.raises(FileExistsError, match="Refusing"):
        main()
