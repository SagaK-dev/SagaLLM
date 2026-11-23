"""A fail-closed protocol for future independently sourced Java I/O benchmarks.

This module does NOT supply a human-reviewed independent dataset. Review
metadata are self-reported and cannot cryptographically prove independence.
Never run untrusted Java on the host. Opt-in behavior tests use the existing
best-effort OCI runner and should run only inside a disposable Linux VM.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from .java_behavior import MAX_SOURCE_BYTES, oci_image_ready, run_java_in_oci

SCHEMA = "sagallm-independent-java-io-v1"
MAX_SUITE_BYTES = 128_000
MAX_TASKS = 12
MAX_CASES_PER_TASK = 6
MAX_STDIN_BYTES = 4096
MAX_STDOUT_BYTES = 8192
REVIEW_STATUS = "APPROVED_FOR_INDEPENDENT_EVALUATION"
ID_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,48}\Z")
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
DATE_PATTERN = re.compile(r"20[0-9]{2}-[01][0-9]-[0-3][0-9]\Z")


def _https(url: object) -> bool:
    if not isinstance(url, str) or len(url) > 500:
        return False
    parts = urlsplit(url)
    return (parts.scheme == "https" and bool(parts.hostname)
            and not parts.username and not parts.password
            and all(ord(ch) >= 32 for ch in url))


def _text(value: object, max_bytes: int, *, nonempty: bool = False) -> bool:
    return (isinstance(value, str)
            and (bool(value.strip()) or not nonempty)
            and "\x00" not in value
            and len(value.encode("utf-8")) <= max_bytes)


def _review_structure(suite: dict) -> tuple[bool, list[str]]:
    blockers = []
    if suite.get("origin") != "EXTERNALLY_SOURCED_HUMAN_TASKS":
        blockers.append("ORIGIN_NOT_ATTESTED")
    if suite.get("review_status") != REVIEW_STATUS:
        blockers.append("HUMAN_REVIEW_PENDING")
    if not _https(suite.get("provenance_url")):
        blockers.append("PROVENANCE_MISSING")
    if not _text(suite.get("license_id"), 96, nonempty=True) or suite.get("license_id") == "NOT_REVIEWED":
        blockers.append("LICENSE_UNREVIEWED")
    if suite.get("train_overlap_review") != "ATTESTED_NO_KNOWN_OVERLAP":
        blockers.append("TRAIN_OVERLAP_REVIEW_PENDING")
    reviewers = suite.get("reviewers")
    if not isinstance(reviewers, list) or len(reviewers) < 2:
        blockers.append("TWO_HUMAN_REVIEWS_REQUIRED")
    else:
        identities = set()
        for entry in reviewers:
            if (not isinstance(entry, dict)
                    or not _text(entry.get("reviewer_id"), 100, nonempty=True)
                    or not DATE_PATTERN.fullmatch(str(entry.get("date", "")))
                    or not _https(entry.get("review_url"))):
                blockers.append("INVALID_REVIEW_RECORD")
                break
            identities.add(entry["reviewer_id"])
        if len(identities) < 2:
            blockers.append("DISTINCT_REVIEWERS_REQUIRED")
    # This is only a schema check. In particular we do not authenticate
    # any reviewer or verify the external license / source provenance.
    return not blockers, sorted(set(blockers))


def validate_suite(suite: object) -> dict:
    if not isinstance(suite, dict) or suite.get("schema") != SCHEMA:
        raise ValueError("Unsupported Java benchmark schema")
    if set(suite) != {
        "schema", "benchmark_id", "origin", "review_status",
        "provenance_url", "license_id", "train_overlap_review",
        "reviewers", "tasks",
    }:
        raise ValueError("Unexpected Java benchmark fields")
    if not ID_PATTERN.fullmatch(str(suite.get("benchmark_id", ""))):
        raise ValueError("Invalid benchmark ID")
    tasks = suite.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= MAX_TASKS:
        raise ValueError("Benchmark needs 1..12 tasks")
    ids, families, total = set(), set(), 0
    for item in tasks:
        if not isinstance(item, dict) or set(item) != {"id", "family", "prompt", "cases"}:
            raise ValueError("Invalid task structure")
        task_id = item["id"]
        family = item["family"]
        if (not isinstance(task_id, str) or not ID_PATTERN.fullmatch(task_id)
                or task_id in ids):
            raise ValueError("Duplicate or invalid task ID")
        if not isinstance(family, str) or not ID_PATTERN.fullmatch(family):
            raise ValueError("Invalid task family")
        if not _text(item["prompt"], 1800, nonempty=True):
            raise ValueError("Invalid task prompt")
        cases = item["cases"]
        if not isinstance(cases, list) or not 2 <= len(cases) <= MAX_CASES_PER_TASK:
            raise ValueError("Each task needs 2..6 I/O tests")
        for case in cases:
            if (not isinstance(case, dict) or set(case) != {"stdin", "stdout"} or
                    not _text(case["stdin"], MAX_STDIN_BYTES) or
                    not _text(case["stdout"], MAX_STDOUT_BYTES)):
                raise ValueError("Invalid or unbounded I/O example")
            total += 1
        ids.add(task_id)
        families.add(family)
    reviewed, blockers = _review_structure(suite)
    return {
        "status": "STRUCTURALLY_VALID",
        "review_gate": "ELIGIBLE_SELF_ATTESTED" if reviewed else "REVIEW_REQUIRED",
        "review_blockers": blockers,
        "review_metadata_verified_externally": False,
        "tasks": len(tasks),
        "task_families": len(families),
        "io_cases": total,
        "ready_for_model_selection": False,
        "independence_proven": False,
    }


def read_suite(path: Path) -> tuple[dict, str, dict]:
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size > MAX_SUITE_BYTES):
        raise ValueError("Missing, symlinked or oversized benchmark suite")
    raw = path.read_bytes()
    suite = json.loads(raw)
    report = validate_suite(suite)
    # This hash pins the entire authored/reviewed material, not only prompts.
    return suite, hashlib.sha256(raw).hexdigest(), report


def _candidate(path: Path) -> str | None:
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size > MAX_SOURCE_BYTES):
        return None
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_SOURCE_BYTES or b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def evaluate_suite(suite_path: Path, candidate_dir: Path, *,
                   allow_execution: bool = False, expected_suite_sha256: str | None = None,
                   backend: str = "docker", image: str = "eclipse-temurin:21-jdk") -> dict:
    suite, fingerprint, protocol = read_suite(suite_path)
    result = {
        "protocol": SCHEMA,
        "suite_sha256": fingerprint,
        "review_gate": protocol["review_gate"],
        "review_metadata_verified_externally": False,
        "benchmark_independence_proven": False,
        "task_count": protocol["tasks"],
        "io_case_count": protocol["io_cases"],
        "behavioral_status": "NOT_RUN",
        "behavior_passed_tasks": None,
        "java_executed": False,
        "model_selection": "NOT_PERMITTED",
        "safety": "OCI_BEST_EFFORT_NOT_SECURITY_PROOF",
    }
    if not allow_execution:
        result["reason"] = "EXPLICIT_OPT_IN_REQUIRED"
        return result
    if protocol["review_gate"] != "ELIGIBLE_SELF_ATTESTED":
        result["reason"] = "HUMAN_REVIEW_REQUIRED"
        result["review_blockers"] = protocol["review_blockers"]
        return result
    if (not isinstance(expected_suite_sha256, str)
            or not SHA_PATTERN.fullmatch(expected_suite_sha256)
            or expected_suite_sha256 != fingerprint):
        result["reason"] = "PINNED_SUITE_SHA256_REQUIRED"
        return result
    if (not candidate_dir.is_dir() or candidate_dir.is_symlink()
            or not candidate_dir.resolve().is_dir()):
        result["reason"] = "CANDIDATE_DIRECTORY_MISSING"
        return result
    if not oci_image_ready(backend, image):
        result["reason"] = "LOCAL_OCI_RUNTIME_OR_IMAGE_UNAVAILABLE"
        return result
    results = []
    any_unavailable = False
    for task in suite["tasks"]:
        task_id = task["id"]
        code = _candidate(candidate_dir / (task_id + ".java"))
        if code is None:
            results.append({"id": task_id, "status": "CANDIDATE_MISSING",
                            "cases_passed": 0, "cases_total": len(task["cases"])})
            continue
        outcomes = []
        for case in task["cases"]:
            outcome = run_java_in_oci(
                code, case["stdout"], backend=backend, image=image,
                allow_execution=True, stdin_data=case["stdin"])
            outcomes.append(outcome["status"])
        any_unavailable |= any(s in ("NOT_RUN", "RUNTIME_ERROR") for s in outcomes)
        results.append({
            "id": task_id,
            "status": ("PASS" if all(s == "PASS" for s in outcomes)
                       else "UNAVAILABLE" if any(s in ("NOT_RUN", "RUNTIME_ERROR") for s in outcomes)
                       else "FAIL"),
            "case_statuses": outcomes,
            "cases_passed": sum(s == "PASS" for s in outcomes),
            "cases_total": len(outcomes),
        })
    result.update({
        "behavioral_status": "PARTIAL_OR_UNAVAILABLE" if any_unavailable else "EVALUATED",
        "behavior_passed_tasks": (None if any_unavailable else
                                  sum(x["status"] == "PASS" for x in results)),
        "java_executed": True,
        "model_selection": "NOT_PERMITTED_NO_INDEPENDENCE_PROOF",
        "results": results,
        "source_persisted": False,
        "raw_stdout_stderr_persisted": False,
    })
    return result
