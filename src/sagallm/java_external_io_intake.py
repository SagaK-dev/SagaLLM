"""Strict provenance checks for unreviewed, externally sourced Java I/O intake.

The imported Exercism canonical cases are NOT a human-approved Java console
benchmark. This verifies raw pinned Git blobs and exact case transformations;
it does not establish exercise-level redistribution rights, human authorship,
lack of semantic overlap or Java model behavior.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .java_diverse_corpus import NEW_TEST, NEW_TRAIN, NEW_VALIDATION
from .java_independent_benchmark import validate_suite
from .java_large_corpus import TRAIN_FAMILIES, TEST_FAMILIES, VALIDATION_FAMILIES

SCHEMA = "sagallm-external-java-io-intake-v1"
SOURCE_REPOSITORY = "exercism/problem-specifications"
PINNED_REVISION = "4812c213a2647e2034ad5ac78155ce9265cdd171"
LICENSE_BLOB_SHA1 = "6813d6298df9a46da6485050e8a5fcd61ab79879"
PINNED_DATA_BLOBS = {
    "hamming": "93b438b75f2953b154b407eb73d3f63dadbe9ef9",
    "rna-transcription": "d05146ed00c9af5eaeedf7a1fb2292e305420ed9",
    "luhn": "78174b9128a1666ce43190eaafb6b00cd3c33479",
    "roman-numerals": "1edb968d321bc4b9473e446a5d3f1daa173a4d56",
    "allergies": "8fcc359a472dcd70773663e0b039a1d75796d123",
    "atbash-cipher": "984ea2d6be61592eaa54e3cd434f47646db8bb1d",
}
PINNED_METADATA_BLOBS = {
    "hamming": "320d71cddf972b51cd501a75ca51dace670fe441",
    "rna-transcription": "4b495a9a0990bd301b0ef733fcbe2edf414e3052",
    "luhn": "5c2924db9de766bafbf6c6ce60784efc3051ccec",
    "roman-numerals": "16ddf8c335cdf234b6c7aca2b5ac784e9cdf2082",
    "allergies": "9c15b4c1cf7827407338402bd8aa345770b51c27",
    "atbash-cipher": "aa2098892180f717cdd6d562c846252ccfe1a302",
}
TARGET_FAMILIES = {
    "hamming": ("ex_hamming", "dna_hamming_distance", "distance"),
    "rna-transcription": ("ex_rna_transcription", "dna_rna_complement", "toRna"),
    "luhn": ("ex_luhn", "luhn_checksum", "valid"),
    "roman-numerals": ("ex_roman_numerals", "roman_numeral_conversion", "roman"),
    "allergies": ("ex_allergies", "allergy_bitmask_lookup", "allergicTo"),
    "atbash-cipher": ("ex_atbash_cipher", "atbash_text_encoding", "encode"),
}
UUID = re.compile(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")


def _read(path: Path, limit: int) -> bytes:
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size > limit):
        raise ValueError("Missing, unsafe, or oversized intake file")
    return path.read_bytes()


def _blob_sha1(raw: bytes) -> str:
    # Git blob object, not the SHA256 of the downloaded file.
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") +
                        b"\x00" + raw).hexdigest()


def _cases(container: list) -> list[dict]:
    result = []
    for obj in container:
        if not isinstance(obj, dict):
            raise ValueError("Malformed upstream case")
        if "uuid" in obj:
            result.append(obj)
        nested = obj.get("cases")
        if isinstance(nested, list):
            result.extend(_cases(nested))
    return result


def _adapt(exercise: str, row: dict) -> dict[str, str]:
    """Draft console adapter (NOT part of upstream Exercism contract)."""
    value = row["input"]
    expected = row["expected"]
    if exercise == "hamming":
        if (not isinstance(expected, int) or isinstance(expected, bool)
                or len(value["strand1"]) != len(value["strand2"])):
            raise ValueError("Unsupported unequal-length Hamming case")
        return {"stdin": value["strand1"] + "\n" + value["strand2"] + "\n",
                "stdout": str(expected) + "\n"}
    if exercise == "rna-transcription":
        if not isinstance(expected, str):
            raise ValueError("Unsupported RNA case")
        return {"stdin": value["dna"] + "\n", "stdout": expected + "\n"}
    if exercise == "luhn":
        if type(expected) is not bool:
            raise ValueError("Unsupported Luhn case")
        return {"stdin": value["value"] + "\n",
                "stdout": str(expected).lower() + "\n"}
    if exercise == "roman-numerals":
        if not isinstance(expected, str) or not 1 <= value["number"] <= 3999:
            raise ValueError("Unsupported Roman numeral case")
        return {"stdin": str(value["number"]) + "\n",
                "stdout": expected + "\n"}
    if exercise == "allergies":
        if type(expected) is not bool:
            raise ValueError("Only allergy Boolean predicate is supported")
        return {"stdin": str(value["score"]) + " " + value["item"] + "\n",
                "stdout": str(expected).lower() + "\n"}
    if exercise == "atbash-cipher":
        if not isinstance(expected, str):
            raise ValueError("Only Atbash string encode is supported")
        return {"stdin": value["phrase"] + "\n", "stdout": expected + "\n"}
    raise ValueError("Exercise outside pinned allowlist")


def verify_intake(root: Path, *, upstream_root: Path | None = None) -> dict:
    root = Path(root)
    suite_raw = _read(root / "examples/java_external_io_INTAKE.json", 128_000)
    meta_raw = _read(root / "examples/java_external_io_sources.json", 64_000)
    license_raw = _read(root / "third_party_notices/exercism-problem-specifications-LICENSE.txt", 8192)
    if _blob_sha1(license_raw) != LICENSE_BLOB_SHA1:
        raise ValueError("Bundled upstream MIT license blob changed")
    suite = json.loads(suite_raw)
    manifest = json.loads(meta_raw)
    protocol = validate_suite(suite)
    if (protocol["review_gate"] != "REVIEW_REQUIRED" or
            suite["origin"] != "EXTERNAL_SOURCE_INTAKE_UNREVIEWED" or
            suite["review_status"] != "PENDING_HUMAN_REVIEW" or
            suite["license_id"] != "NOT_REVIEWED" or
            suite["train_overlap_review"] != "NOT_REVIEWED" or
            suite["reviewers"] != []):
        raise ValueError("Intake was falsely promoted to reviewed status")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema", "source_repository", "source_revision",
        "source_license_file", "source_license_git_blob_sha1",
        "upstream_repo_license_label", "candidate_suite",
        "candidate_is_not_approved", "reviewers_attested",
        "semantic_overlap_audited", "java_executed",
        "model_generated_outputs_evaluated", "console_adapter_author",
        "copyright_notice", "sources",
    }:
        raise ValueError("Invalid source provenance manifest")
    if (manifest["schema"] != SCHEMA or
            manifest["source_repository"] != SOURCE_REPOSITORY or
            manifest["source_revision"] != PINNED_REVISION or
            manifest["source_license_git_blob_sha1"] != LICENSE_BLOB_SHA1 or
            manifest["source_license_file"] !=
            "https://github.com/exercism/problem-specifications/blob/" +
            PINNED_REVISION + "/LICENSE" or
            manifest["candidate_suite"] != "java_external_io_INTAKE.json" or
            manifest["candidate_is_not_approved"] is not True or
            manifest["reviewers_attested"] != 0 or
            manifest["semantic_overlap_audited"] is not False or
            manifest["java_executed"] is not False or
            manifest["model_generated_outputs_evaluated"] is not False or
            manifest["console_adapter_author"] != "assistant, unreviewed" or
            "Exercism" not in manifest["copyright_notice"]):
        raise ValueError("Unpinned or incorrectly reviewed source manifest")
    sources = manifest["sources"]
    if (not isinstance(sources, list) or
            len(sources) != len(TARGET_FAMILIES) or
            not isinstance(suite["tasks"], list) or
            len(suite["tasks"]) != len(sources)):
        raise ValueError("Incomplete external exercise candidate coverage")
    if (suite["provenance_url"] !=
            "https://github.com/exercism/problem-specifications/tree/" +
            PINNED_REVISION + "/exercises"):
        raise ValueError("Incorrect upstream commit link")

    all_known_families = (set(TRAIN_FAMILIES) | set(NEW_TRAIN) |
                          set(VALIDATION_FAMILIES) | set(NEW_VALIDATION) |
                          set(TEST_FAMILIES) | set(NEW_TEST))
    seen_exercises, checked_cases, pinned_files = set(), 0, 0
    for source, task in zip(sources, suite["tasks"]):
        exercise = source.get("exercise")
        if exercise not in TARGET_FAMILIES or exercise in seen_exercises:
            raise ValueError("Unknown or duplicated upstream exercise")
        seen_exercises.add(exercise)
        task_id, family, property_name = TARGET_FAMILIES[exercise]
        expected_path = "exercises/" + exercise
        if (source.get("task_id") != task_id or
                source.get("family") != family or
                source.get("test_property") != property_name or
                source.get("canonical_data_path") != expected_path + "/canonical-data.json" or
                source.get("metadata_path") != expected_path + "/metadata.toml" or
                source.get("canonical_data_git_blob_sha1") != PINNED_DATA_BLOBS[exercise] or
                source.get("metadata_git_blob_sha1") != PINNED_METADATA_BLOBS[exercise] or
                source.get("console_adapter") != "ASSISTANT_DRAFT_UNREVIEWED" or
                source.get("source_license_review") != "PENDING_HUMAN_REVIEW" or
                source.get("semantic_overlap_review") != "PENDING_HUMAN_REVIEW" or
                not isinstance(source.get("metadata_source"), str)):
            raise ValueError("Source, license, or exact case mapping changed")
        if task["id"] != task_id or task["family"] != family:
            raise ValueError("Task not aligned with provenance source")
        ids = source.get("selected_case_uuids")
        if (not isinstance(ids, list) or len(ids) != len(task["cases"]) or
                len(ids) != len(set(ids)) or
                any(not isinstance(uid, str) or not UUID.fullmatch(uid) for uid in ids)):
            raise ValueError("Invalid UUID selection or mismatched case count")
        checked_cases += len(ids)
        if upstream_root is not None:
            upstream = Path(upstream_root)
            canonical_file = upstream / expected_path / "canonical-data.json"
            metadata_file = upstream / expected_path / "metadata.toml"
            canonical_bytes = _read(canonical_file, 1_000_000)
            metadata_bytes = _read(metadata_file, 100_000)
            if (_blob_sha1(canonical_bytes) != PINNED_DATA_BLOBS[exercise] or
                    _blob_sha1(metadata_bytes) != PINNED_METADATA_BLOBS[exercise]):
                raise ValueError("Upstream Git blob identity mismatch")
            info = metadata_bytes.decode("utf-8")
            for name, value in (("source", source.get("metadata_source")),
                                ("source_url", source.get("metadata_source_url"))):
                match = re.search(r'^' + name + r'\s*=\s*"([^"]*)"', info, re.MULTILINE)
                if (match.group(1) if match else None) != value:
                    raise ValueError("Exercise attribution changed")
            original = json.loads(canonical_bytes)
            if original.get("exercise") != exercise:
                raise ValueError("Incorrect upstream canonical exercise")
            original_cases = {row["uuid"]: row for row in
                              _cases(original.get("cases", [])) if "uuid" in row}
            for uuid, item in zip(ids, task["cases"]):
                ref = original_cases.get(uuid)
                if ref is None or ref.get("property") != property_name:
                    raise ValueError("Selected upstream case UUID not found")
                if _adapt(exercise, ref) != item:
                    raise ValueError("Draft Java stdin/stdout adaptation diverges")
            pinned_files += 2

    if seen_exercises != set(TARGET_FAMILIES):
        raise ValueError("Missing upstream source candidate")
    return {
        "status": ("PINNED_UPSTREAM_CASES_VERIFIED" if upstream_root is not None
                   else "INTAKE_STRUCTURE_ONLY"),
        "suite_sha256": hashlib.sha256(suite_raw).hexdigest(),
        "provenance_sha256": hashlib.sha256(meta_raw).hexdigest(),
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": PINNED_REVISION,
        "exercise_candidates": len(sources),
        "io_cases": checked_cases,
        "upstream_git_blobs_checked": pinned_files,
        "license_file_git_blob_checked": True,
        "exact_family_label_overlap_count": len(
            all_known_families & {task["family"] for task in suite["tasks"]}),
        "semantic_overlap_review": "PENDING_HUMAN_REVIEW",
        "exercise_level_rights_review": "PENDING_HUMAN_REVIEW",
        "assistant_java_io_adaptation_review": "PENDING_HUMAN_REVIEW",
        "human_reviewers_verified": 0,
        "behavior": "NOT_RUN",
        "model_outputs_evaluated": False,
        "model_selection_permitted": False,
        "independence_proven": False,
        "review_gate": protocol["review_gate"],
    }
