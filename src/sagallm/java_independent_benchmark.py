"""A fail-closed protocol for future independently sourced Java I/O benchmarks.

This module does NOT supply a human-reviewed independent dataset. Review
metadata are self-reported and cannot cryptographically prove independence.
Never run untrusted Java on the host. Opt-in behavior tests use the existing
best-effort OCI runner and should run only inside a disposable Linux VM.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from .java_behavior import (MAX_SOURCE_BYTES, oci_image_ready, run_java_in_oci,
    STAGED_EXECUTION_PROTOCOL, STAGED_STATUSES, STAGED_UNAVAILABLE, compile_phase_counts)
from .java_vm_review import local_docker_context, local_image_has_no_volumes, review_gate

SCHEMA = "sagallm-independent-java-io-v1"
MAX_SUITE_BYTES = 128_000
MAX_TASKS = 12
MIN_REVIEWED_TASKS = 5
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
    if any(ch.isspace() or ord(ch) < 32 for ch in url):
        return False
    try:
        parts = urlsplit(url)
        return (parts.scheme == "https" and bool(parts.hostname)
                and not parts.username and not parts.password)
    except ValueError:
        return False


def _text(value: object, max_bytes: int, *, nonempty: bool = False) -> bool:
    return (isinstance(value, str)
            and (bool(value.strip()) or not nonempty)
            and "\x00" not in value
            and len(value.encode("utf-8")) <= max_bytes)


def _valid_review_date(value: object) -> bool:
    if not isinstance(value, str) or not DATE_PATTERN.fullmatch(value):
        return False
    try:
        return date.fromisoformat(value) <= date.today()
    except ValueError:
        return False


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _review_structure(suite: dict) -> tuple[bool, list[str]]:
    blockers = []
    task_ids = {task["id"] for task in suite["tasks"]}
    if len(task_ids) < MIN_REVIEWED_TASKS:
        blockers.append("MINIMUM_FIVE_TASKS_REQUIRED")
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
        links = set()
        for entry in reviewers:
            if (not isinstance(entry, dict)
                    or not _text(entry.get("reviewer_id"), 100, nonempty=True)
                    or not _valid_review_date(entry.get("date"))
                    or not _https(entry.get("review_url"))):
                blockers.append("INVALID_REVIEW_RECORD")
                continue
            identities.add(entry["reviewer_id"].strip().casefold())
            links.add(entry["review_url"])
            scope = entry.get("task_ids")
            if (not isinstance(scope, list)
                    or any(not isinstance(item, str) for item in scope)
                    or len(scope) != len(task_ids) or set(scope) != task_ids):
                blockers.append("PER_TASK_REVIEW_SCOPE_REQUIRED")
        if len(identities) < 2 or len(links) < 2:
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
        inputs = set()
        for case in cases:
            if (not isinstance(case, dict) or set(case) != {"stdin", "stdout"} or
                    not _text(case["stdin"], MAX_STDIN_BYTES) or
                    not _text(case["stdout"], MAX_STDOUT_BYTES)):
                raise ValueError("Invalid or unbounded I/O example")
            if case["stdin"] in inputs:
                raise ValueError("Duplicate stdin in task")
            inputs.add(case["stdin"])
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
    with path.open("rb") as handle:
        raw = handle.read(MAX_SUITE_BYTES + 1)
    if len(raw) > MAX_SUITE_BYTES:
        raise ValueError("Oversized benchmark suite")
    suite = json.loads(raw, object_pairs_hook=_unique_object)
    report = validate_suite(suite)
    # This hash pins the entire authored/reviewed material, not only prompts.
    return suite, hashlib.sha256(raw).hexdigest(), report


def _candidate(path: Path) -> str | None:
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size > MAX_SOURCE_BYTES):
        return None
    with path.open("rb") as handle:
        raw = handle.read(MAX_SOURCE_BYTES + 1)
    if not raw or len(raw) > MAX_SOURCE_BYTES or b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _private_intake_gate(suite_path, fingerprint, *, provenance_path,
                         expected_provenance_sha256, inventory_path,
                         expected_inventory_sha256, exposure_root):
    # Import locally: the intake module itself uses suite validation.
    from .java_private_intake import audit_private_intake
    if any(value is None for value in (
            provenance_path, expected_provenance_sha256, inventory_path,
            expected_inventory_sha256, exposure_root)):
        return "PINNED_PRIVATE_INTAKE_REQUIRED", None
    try:
        report = audit_private_intake(
            suite_path, expected_suite_sha256=fingerprint,
            provenance_path=provenance_path,
            expected_provenance_sha256=expected_provenance_sha256,
            inventory_path=inventory_path,
            expected_inventory_sha256=expected_inventory_sha256,
            exposure_root=exposure_root)
    except (OSError, ValueError, TypeError, RecursionError):
        # Do not disclose private file paths or malformed record content.
        return "PRIVATE_INTAKE_INVALID", None
    if report["machine_status"] != "PREFLIGHT_ONLY" or report["blockers"]:
        return "PRIVATE_INTAKE_BLOCKED", report
    return "PREFLIGHT_ONLY", report


def _model_freeze_gate(suite_path, fingerprint, *, freeze_path,
                       expected_freeze_sha256, provenance_path,
                       expected_provenance_sha256, inventory_path,
                       expected_inventory_sha256, shard_root, artifact_root, image):
    from .java_model_freeze import verify_model_freeze
    if any(value is None for value in (
            freeze_path, expected_freeze_sha256, provenance_path,
            expected_provenance_sha256, inventory_path, expected_inventory_sha256,
            shard_root, artifact_root)):
        return "PINNED_MODEL_FREEZE_REQUIRED", None
    try:
        report = verify_model_freeze(
            suite_path, fingerprint, freeze_path=freeze_path,
            expected_freeze_sha256=expected_freeze_sha256,
            provenance_path=provenance_path,
            expected_provenance_sha256=expected_provenance_sha256,
            inventory_path=inventory_path,
            expected_inventory_sha256=expected_inventory_sha256,
            shard_root=shard_root, artifact_root=artifact_root, image=image)
    except (OSError, ValueError, TypeError, RecursionError):
        return "MODEL_FREEZE_INVALID", None
    if report["machine_status"] != "PREFLIGHT_ONLY" or report["blockers"]:
        return "MODEL_FREEZE_BLOCKED", report
    return "PREFLIGHT_ONLY", report


def _pinned_candidates(path, digest, freeze_sha, task_ids, candidate_dir):
    from .java_model_freeze import read_candidate_freeze
    return read_candidate_freeze(path, digest, freeze_sha, task_ids, candidate_dir)


def evaluate_suite(suite_path: Path, candidate_dir: Path, *,
                   allow_execution: bool = False, expected_suite_sha256: str | None = None,
                   disposable_vm_acknowledged: bool = False,
                   backend: str = "docker", image: str = "eclipse-temurin:21-jdk",
                   vm_review_path: Path | None = None,
                   expected_vm_review_sha256: str | None = None,
                   provenance_path: Path | None = None,
                   expected_provenance_sha256: str | None = None,
                   inventory_path: Path | None = None,
                   expected_inventory_sha256: str | None = None,
                   exposure_root: Path | None = None,
                   freeze_path: Path | None = None,
                   expected_freeze_sha256: str | None = None,
                   shard_root: Path | None = None,
                   artifact_root: Path | None = None,
                   candidate_freeze_path: Path | None = None,
                   expected_candidate_freeze_sha256: str | None = None) -> dict:
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
    if not disposable_vm_acknowledged:
        result["reason"] = "DISPOSABLE_VM_ACKNOWLEDGEMENT_REQUIRED"
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
    vm_status, vm_sha = review_gate(
        vm_review_path, expected_vm_review_sha256,
        suite_sha=fingerprint, backend=backend, image=image)
    result["vm_review_gate"] = vm_status
    result["vm_review_metadata_verified_externally"] = False
    if vm_status != "ELIGIBLE_SELF_ATTESTED":
        result["reason"] = vm_status
        return result
    result["vm_review_sha256"] = vm_sha
    result["oci_image"] = image
    freeze_status, freeze_report = _model_freeze_gate(
        suite_path, fingerprint, freeze_path=freeze_path,
        expected_freeze_sha256=expected_freeze_sha256,
        provenance_path=provenance_path,
        expected_provenance_sha256=expected_provenance_sha256,
        inventory_path=inventory_path,
        expected_inventory_sha256=expected_inventory_sha256,
        shard_root=shard_root, artifact_root=artifact_root, image=image)
    result["model_freeze_gate"] = freeze_status
    if freeze_report is not None:
        result["model_freeze"] = freeze_report
        result["model_freeze_report_canonical_sha256"] = hashlib.sha256(
            json.dumps(freeze_report, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":")).encode("utf-8")).hexdigest()
    if freeze_status != "PREFLIGHT_ONLY":
        result["reason"] = freeze_status
        return result
    if (not candidate_dir.is_dir() or candidate_dir.is_symlink()
            or not candidate_dir.resolve().is_dir()):
        result["reason"] = "CANDIDATE_DIRECTORY_MISSING"
        return result
    if candidate_freeze_path is None or expected_candidate_freeze_sha256 is None:
        result["reason"] = "PINNED_CANDIDATE_FREEZE_REQUIRED"
        return result
    try:
        pinned_codes = _pinned_candidates(
            candidate_freeze_path, expected_candidate_freeze_sha256,
            expected_freeze_sha256, [t["id"] for t in suite["tasks"]], candidate_dir)
    except (OSError, ValueError, TypeError, RecursionError):
        result["reason"] = "CANDIDATE_FREEZE_INVALID"
        return result
    result["candidate_freeze_sha256"] = expected_candidate_freeze_sha256
    if not local_docker_context():
        result["reason"] = "LOCAL_DOCKER_CONTEXT_REQUIRED"
        return result
    if not oci_image_ready(backend, image):
        result["reason"] = "LOCAL_OCI_RUNTIME_OR_IMAGE_UNAVAILABLE"
        return result
    if not local_image_has_no_volumes(image):
        result["reason"] = "OCI_IMAGE_VOLUME_CHECK_FAILED"
        return result
    results = []
    all_case_statuses = []
    any_unavailable = False
    attempted = False
    for task in suite["tasks"]:
        task_id = task["id"]
        code = pinned_codes[task_id]
        if code is None:
            any_unavailable = True
            all_case_statuses.extend(["CANDIDATE_MISSING"] * len(task["cases"]))
            results.append({"id": task_id, "status": "CANDIDATE_MISSING",
                            "cases_passed": 0, "cases_total": len(task["cases"])})
            continue
        outcomes = []
        attempted = True
        for case in task["cases"]:
            outcome = run_java_in_oci(
                code, case["stdout"], backend=backend, image=image,
                allow_execution=True, stdin_data=case["stdin"], staged=True)
            if outcome["status"] not in STAGED_STATUSES:
                raise ValueError("Unknown staged OCI outcome")
            outcomes.append(outcome["status"])
        all_case_statuses.extend(outcomes)
        any_unavailable |= any(s in STAGED_UNAVAILABLE for s in outcomes)
        results.append({
            "id": task_id,
            "status": ("PASS" if all(s == "PASS" for s in outcomes)
                       else "UNAVAILABLE" if any(s in STAGED_UNAVAILABLE for s in outcomes)
                       else "FAIL"),
            "candidate_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
            "case_statuses": outcomes,
            "cases_passed": sum(s == "PASS" for s in outcomes),
            "cases_total": len(outcomes),
        })
    result.update({
        "behavioral_status": "PARTIAL_OR_UNAVAILABLE" if any_unavailable else "EVALUATED",
        "behavior_passed_tasks": (None if any_unavailable else
                                  sum(x["status"] == "PASS" for x in results)),
        "java_executed": "POSSIBLE_IN_OCI_ONLY" if attempted else False,
        "oci_execution_attempted": attempted,
        "model_selection": "NOT_PERMITTED_NO_INDEPENDENCE_PROOF",
        "results": results,
        "execution_protocol": STAGED_EXECUTION_PROTOCOL,
        "compile_case_counts": compile_phase_counts(all_case_statuses),
        "source_persisted": False,
        "raw_stdout_stderr_persisted": False,
    })
    return result
