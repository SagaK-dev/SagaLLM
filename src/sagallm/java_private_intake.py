"""Offline, non-executing preflight for privately held-back Java tasks.

This is a limited lexical screen and provenance consistency check, not proof
of independence, legal permission, authorship or reviewer authenticity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unicodedata

from . import java_coding_eval, java_generated_data, java_large_corpus
from . import java_diverse_corpus, java_stdio_curriculum
from .java_independent_benchmark import (
    ID_PATTERN, SHA_PATTERN, _https, _text, _unique_object,
    _valid_review_date, read_suite,
)

SCHEMA = "sagallm-private-java-intake-v1"
INVENTORY_SCHEMA = "sagallm-private-java-exposure-v1"
GROUPS = ("train", "validation", "test", "development", "pretraining")
MAX_META = 128_000
MAX_FILES = 64
MAX_FILE = 100_000_000
MAX_TOTAL = 512_000_000
MAX_LINE = 262_144
MAX_ROWS = 500_000
# These nine previously published examples are not new holdback tasks.
EXPOSED = {
    "ex_hamming", "dna_hamming_distance", "ex_rna_transcription",
    "dna_rna_complement", "ex_luhn", "luhn_checksum", "ex_roman_numerals",
    "roman_numeral_conversion", "ex_allergies", "allergy_bitmask_lookup",
    "ex_atbash_cipher", "atbash_text_encoding", "interval_overlap_minutes",
    "half_open_intervals", "normalize_ascii_spaces",
    "ascii_whitespace_normalization", "first_missing_positive_integer",
    "positive_integer_set_gap",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def task_sha256(task: dict) -> str:
    """Canonical task object digest; the suite digest also pins exact bytes."""
    return _sha(json.dumps(task, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _metadata(path: Path, expected: str) -> tuple[dict, str]:
    if not isinstance(expected, str) or not SHA_PATTERN.fullmatch(expected):
        raise ValueError("Expected lowercase SHA-256 pin")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Metadata missing or symlinked")
    with path.open("rb") as handle:
        raw = handle.read(MAX_META + 1)
    if len(raw) > MAX_META or _sha(raw) != expected:
        raise ValueError("Metadata size or SHA mismatch")
    obj = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(obj, dict):
        raise ValueError("Metadata must be an object")
    return obj, expected


def known_families() -> dict[str, set[str]]:
    """Actual current synthetic family labels; not a complete exposure inventory."""
    result = {group: set() for group in GROUPS}
    for group in ("train", "validation", "test"):
        result[group].update(java_coding_eval.SPLIT_FAMILIES[group])
        result[group].update(getattr(java_generated_data, group.upper() + "_FAMILIES"))
        result[group].update(getattr(java_large_corpus, group.upper() + "_FAMILIES"))
        result[group].update(getattr(java_diverse_corpus, "NEW_" + group.upper()))
        result[group].update(getattr(java_stdio_curriculum, group.upper() + "_FAMILIES", ()))
    result["development"].update(EXPOSED)
    return result


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _shape(value: str) -> str:
    return re.sub(r"\d+", "#", value)


def _provenance(suite: dict, suite_sha: str, meta: dict) -> list[str]:
    if (set(meta) != {"schema", "suite_sha256", "tasks"}
            or meta["schema"] != SCHEMA or meta["suite_sha256"] != suite_sha):
        raise ValueError("Provenance schema or suite pin mismatch")
    rows = meta["tasks"]
    if not isinstance(rows, list) or len(rows) != len(suite["tasks"]):
        raise ValueError("Provenance must cover every task")
    tasks = {task["id"]: task for task in suite["tasks"]}
    seen = set()
    blockers = set()
    fields = {"task_id", "task_sha256", "author_id", "authored_on", "revision",
              "source_url", "license_id", "permission_url", "allowed_uses",
              "held_back", "rights_review", "io_review", "overlap_review"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError("Unexpected task provenance fields")
        ident = row["task_id"]
        if not isinstance(ident, str) or ident not in tasks or ident in seen:
            raise ValueError("Unknown or duplicated task provenance")
        seen.add(ident)
        if row["task_sha256"] != task_sha256(tasks[ident]):
            raise ValueError("Task provenance hash mismatch")
        for key in ("author_id", "revision", "license_id"):
            if not _text(row[key], 200, nonempty=True):
                raise ValueError("Missing task provenance value")
        if not _valid_review_date(row["authored_on"]):
            raise ValueError("Invalid task authorship date")
        if not _https(row["source_url"]) or not _https(row["permission_url"]):
            raise ValueError("Missing traceable source or permission URL")
        uses = row["allowed_uses"]
        if (not isinstance(uses, list) or any(not isinstance(x, str) for x in uses)
                or len(uses) != len(set(uses))
                or not set(uses) <= {"evaluate", "redistribute"}):
            raise ValueError("Private holdback must not grant training access")
        if "evaluate" not in uses or "redistribute" not in uses:
            blockers.add("TASK_PERMISSION_PENDING")
        if row["held_back"] is not True:
            blockers.add("HOLD_BACK_NOT_ATTESTED")
        for key in ("rights_review", "io_review", "overlap_review"):
            if row[key] not in ("PENDING", "ATTESTED"):
                raise ValueError("Unknown review status")
            if row[key] != "ATTESTED":
                blockers.add(key.upper() + "_PENDING")
    return sorted(blockers)


def _inventory(meta: dict) -> list[dict]:
    if (set(meta) != {"schema", "files"} or meta["schema"] != INVENTORY_SCHEMA
            or not isinstance(meta["files"], list) or len(meta["files"]) > MAX_FILES):
        raise ValueError("Invalid exposure inventory")
    seen = set()
    for row in meta["files"]:
        if (not isinstance(row, dict)
                or set(row) != {"group", "path", "sha256", "format"}
                or row["group"] not in GROUPS
                or row["format"] not in ("chat-jsonl", "text-jsonl", "prompt-jsonl")
                or not isinstance(row["sha256"], str)
                or not SHA_PATTERN.fullmatch(row["sha256"])
                or not isinstance(row["path"], str)):
            raise ValueError("Invalid exposure entry")
        path = Path(row["path"])
        if (not row["path"] or path.is_absolute() or ".." in path.parts
                or "\\" in row["path"] or str(path) != row["path"]
                or path.suffix != ".jsonl" or path in seen):
            raise ValueError("Unsafe or duplicate exposure path")
        seen.add(path)
    return meta["files"]


def _texts(row: dict, kind: str) -> list[str]:
    if not isinstance(row, dict):
        raise ValueError("Expected exposure JSON object")
    if kind == "chat-jsonl":
        messages = row.get("messages")
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(m, dict) or not isinstance(m.get("content"), str)
                       or m.get("role") not in ("system", "user", "assistant")
                       for m in messages)):
            raise ValueError("Malformed chat exposure row")
        values = [m["content"] for m in messages]
    else:
        values = [row.get("text" if kind == "text-jsonl" else "prompt")]
    if not values or any(not isinstance(v, str) for v in values):
        raise ValueError("Missing exposure text")
    return values


def _scan(root: Path, entries: list[dict], tasks: list[dict]) -> tuple[list[dict], list[dict]]:
    normalized = [_normalize(t["prompt"]) for t in tasks]
    shapes = [_shape(t) for t in normalized]
    families = known_families()
    hits = set()
    for i, task in enumerate(tasks):
        for group, labels in families.items():
            if task["family"] in labels or task["id"] in labels:
                hits.add((i, group, "KNOWN_FAMILY_OR_ID"))
        for previous in range(i):
            if task["family"] == tasks[previous]["family"]:
                hits.add((i, "suite", "DUPLICATE_FAMILY"))
            if normalized[i] == normalized[previous]:
                hits.add((i, "suite", "NORMALIZED_PROMPT_MATCH"))
            elif shapes[i] == shapes[previous]:
                hits.add((i, "suite", "NUMERIC_TEMPLATE_MATCH"))
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Exposure root must be a real directory")
    total_bytes = total_rows = 0
    summaries = []
    for entry in entries:
        rel = Path(entry["path"])
        target = root / rel
        if (any((root / Path(*rel.parts[:i])).is_symlink()
                for i in range(1, len(rel.parts) + 1)) or not target.is_file()):
            raise ValueError("Exposure file missing or symlinked")
        digest = hashlib.sha256()
        size = count = 0
        with target.open("rb") as handle:
            while True:
                raw = handle.readline(MAX_LINE + 1)
                if not raw:
                    break
                size += len(raw)
                total_bytes += len(raw)
                count += 1
                total_rows += 1
                if (len(raw) > MAX_LINE or size > MAX_FILE
                        or total_bytes > MAX_TOTAL or total_rows > MAX_ROWS):
                    raise ValueError("Exposure scan limit exceeded")
                digest.update(raw)
                row = json.loads(raw, object_pairs_hook=_unique_object)
                group = entry["group"]
                for value in _texts(row, entry["format"]):
                    text = _normalize(value)
                    shape = _shape(text)
                    for i, task in enumerate(tasks):
                        if text == normalized[i]:
                            hits.add((i, group, "NORMALIZED_PROMPT_MATCH"))
                        elif shape == shapes[i]:
                            hits.add((i, group, "NUMERIC_TEMPLATE_MATCH"))
                        elif len(normalized[i]) >= 40 and normalized[i] in text:
                            hits.add((i, group, "PROMPT_SUBSTRING_MATCH"))
                        if row.get("family") == task["family"] or row.get("id") == task["id"]:
                            hits.add((i, group, "EXPOSURE_FAMILY_OR_ID"))
        if not count or digest.hexdigest() != entry["sha256"]:
            raise ValueError("Empty exposure file or SHA mismatch")
        summaries.append({"group": entry["group"], "sha256": entry["sha256"],
                          "records": count, "bytes": size})
    return summaries, [{"task_index": i, "group": g, "kind": k}
                       for i, g, k in sorted(hits)]


def audit_private_intake(suite_path: Path, *, expected_suite_sha256: str,
                         provenance_path: Path, expected_provenance_sha256: str,
                         inventory_path: Path, expected_inventory_sha256: str,
                         exposure_root: Path) -> dict:
    suite, digest, protocol = read_suite(suite_path)
    if digest != expected_suite_sha256:
        raise ValueError("Suite SHA mismatch")
    provenance, p_sha = _metadata(provenance_path, expected_provenance_sha256)
    inventory, i_sha = _metadata(inventory_path, expected_inventory_sha256)
    blockers = _provenance(suite, digest, provenance)
    files, hits = _scan(exposure_root, _inventory(inventory), suite["tasks"])
    missing = sorted(set(GROUPS) - {x["group"] for x in files})
    if missing:
        blockers.append("EXPOSURE_GROUPS_MISSING")
    if hits:
        blockers.append("OVERLAP_REVIEW_LEADS_FOUND")
    blockers.extend(protocol["review_blockers"])
    return {
        "schema": SCHEMA, "suite_sha256": digest,
        "provenance_sha256": p_sha, "inventory_sha256": i_sha,
        "task_count": protocol["tasks"], "case_count": protocol["io_cases"],
        "scanned_files": files, "missing_exposure_groups": missing,
        "overlap_leads": hits, "blockers": sorted(set(blockers)),
        "machine_status": "BLOCKED" if blockers else "PREFLIGHT_ONLY",
        "inventory_completeness_verified": False,
        "semantic_overlap_resolved": False, "human_reviews_verified": 0,
        "review_gate": "REVIEW_REQUIRED", "independence_proven": False,
        "model_selection_permitted": False, "behavior": "NOT_RUN",
        "generated_java_executed": False,
    }
