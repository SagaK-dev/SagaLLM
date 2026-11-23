"""Independent-Java benchmark protocol tests do not execute untrusted Java."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sagallm import java_independent_benchmark as bench
from sagallm.java_behavior import build_oci_command, run_java_in_oci


DRAFT = Path(__file__).resolve().parents[1] / "examples" / "java_independent_io_DRAFT.json"


def _draft():
    return json.loads(DRAFT.read_text(encoding="utf-8"))


def _attested():
    suite = _draft()
    suite["origin"] = "EXTERNALLY_SOURCED_HUMAN_TASKS"
    suite["review_status"] = bench.REVIEW_STATUS
    suite["license_id"] = "CC0-1.0"
    suite["provenance_url"] = "https://example.org/benchmark/original-questions"
    suite["train_overlap_review"] = "ATTESTED_NO_KNOWN_OVERLAP"
    suite["reviewers"] = [
        {"reviewer_id": "reviewer_a", "date": "2026-11-23",
         "review_url": "https://example.org/review/1"},
        {"reviewer_id": "reviewer_b", "date": "2026-11-23",
         "review_url": "https://example.org/review/2"},
    ]
    # Mock-only fixture; credentials are fabricated, never evidence of real review.
    return suite


def _save(tmp_path, suite, name="suite.json"):
    path = tmp_path / name
    path.write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def test_draft_is_structurally_valid_but_not_independent():
    report = bench.validate_suite(_draft())
    assert report["status"] == "STRUCTURALLY_VALID"
    assert report["review_gate"] == "REVIEW_REQUIRED"
    assert report["tasks"] == 3
    assert report["io_cases"] == 9
    assert report["independence_proven"] is False
    assert "HUMAN_REVIEW_PENDING" in report["review_blockers"]
    assert "TWO_HUMAN_REVIEWS_REQUIRED" in report["review_blockers"]
    assert report["ready_for_model_selection"] is False


def test_no_opt_in_means_no_oci_or_candidate_access(tmp_path, monkeypatch):
    suite = _save(tmp_path, _draft())
    def no_runtime(*args, **kwargs):
        raise AssertionError("Never probe OCI engine or read source on default run")
    monkeypatch.setattr(bench, "oci_image_ready", no_runtime)
    monkeypatch.setattr(bench, "_candidate", no_runtime)
    report = bench.evaluate_suite(suite, tmp_path / "no-candidates")
    assert report["behavioral_status"] == "NOT_RUN"
    assert report["behavior_passed_tasks"] is None
    assert report["reason"] == "EXPLICIT_OPT_IN_REQUIRED"
    assert report["model_selection"] == "NOT_PERMITTED"
    assert report["java_executed"] is False


def test_explicit_exec_requires_disposable_vm_review_and_frozen_sha(tmp_path, monkeypatch):
    suite = _save(tmp_path, _draft())
    monkeypatch.setattr(bench, "oci_image_ready", lambda *a, **k: True)
    base = bench.evaluate_suite(suite, tmp_path, allow_execution=True)
    assert base["reason"] == "DISPOSABLE_VM_ACKNOWLEDGEMENT_REQUIRED"
    approved_vm = bench.evaluate_suite(
        suite, tmp_path, allow_execution=True,
        disposable_vm_acknowledged=True)
    assert approved_vm["reason"] == "HUMAN_REVIEW_REQUIRED"
    assert approved_vm["review_gate"] == "REVIEW_REQUIRED"

    approved = _save(tmp_path, _attested(), "self_attested.json")
    no_digest = bench.evaluate_suite(
        approved, tmp_path, allow_execution=True,
        disposable_vm_acknowledged=True)
    assert no_digest["reason"] == "PINNED_SUITE_SHA256_REQUIRED"
    bad_digest = bench.evaluate_suite(
        approved, tmp_path, allow_execution=True,
        disposable_vm_acknowledged=True,
        expected_suite_sha256="0" * 64)
    assert bad_digest["reason"] == "PINNED_SUITE_SHA256_REQUIRED"

    actual = hashlib.sha256(approved.read_bytes()).hexdigest()
    no_candidates = bench.evaluate_suite(
        approved, tmp_path / "not-present", allow_execution=True,
        disposable_vm_acknowledged=True,
        expected_suite_sha256=actual)
    assert no_candidates["reason"] == "CANDIDATE_DIRECTORY_MISSING"


def test_mocked_oci_receives_each_stdin_and_exact_expected_stdout(tmp_path, monkeypatch):
    suite_data = _attested()
    suite = _save(tmp_path, suite_data)
    candidates = tmp_path / "generated"
    candidates.mkdir()
    for task in suite_data["tasks"]:
        (candidates / (task["id"] + ".java")).write_text(
            "public class Main {public static void main(String[] args) {}}\n",
            encoding="utf-8")
    calls = []
    monkeypatch.setattr(bench, "oci_image_ready", lambda *a, **k: True)
    def fake_oci(source, expected, *, backend, image, allow_execution, stdin_data):
        assert allow_execution is True
        assert "public class Main" in source
        calls.append((stdin_data, expected))
        return {"status": "PASS", "behavior_match": True}
    monkeypatch.setattr(bench, "run_java_in_oci", fake_oci)
    fingerprint = hashlib.sha256(suite.read_bytes()).hexdigest()
    report = bench.evaluate_suite(
        suite, candidates, allow_execution=True,
        disposable_vm_acknowledged=True,
        expected_suite_sha256=fingerprint)
    assert len(calls) == 9
    assert calls[0] == ("10 90 60 120\n", "30\n")
    assert report["behavioral_status"] == "EVALUATED"
    assert report["behavior_passed_tasks"] == 3
    assert report["java_executed"] == "POSSIBLE_IN_OCI_ONLY"
    assert report["model_selection"] == "NOT_PERMITTED_NO_INDEPENDENCE_PROOF"
    assert report["review_metadata_verified_externally"] is False
    rendered = json.dumps(report)
    assert "public class Main" not in rendered
    assert "10 90 60" not in rendered
    assert "30\\n" not in rendered


def test_missing_and_symlinked_candidates_do_not_run(tmp_path, monkeypatch):
    suite = _save(tmp_path, _attested())
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    source = candidate_dir / "private.java"
    source.write_text("public class Main {}", encoding="utf-8")
    (candidate_dir / "interval_overlap_minutes.java").symlink_to(source)
    monkeypatch.setattr(bench, "oci_image_ready", lambda *a, **k: True)
    monkeypatch.setattr(bench, "run_java_in_oci", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("Must not execute")))
    report = bench.evaluate_suite(
        suite, candidate_dir, allow_execution=True,
        disposable_vm_acknowledged=True,
        expected_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest())
    assert report["behavioral_status"] == "PARTIAL_OR_UNAVAILABLE"
    assert report["behavior_passed_tasks"] is None
    assert all(x["status"] == "CANDIDATE_MISSING" for x in report["results"])


def test_schema_rejects_duplicate_ids_and_oversize_and_bad_provenance():
    first = _draft()
    first["tasks"].append(copy.deepcopy(first["tasks"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        bench.validate_suite(first)
    first = _draft()
    first["tasks"][0]["cases"][0]["stdout"] = "x" * 8193
    with pytest.raises(ValueError, match="unbounded"):
        bench.validate_suite(first)
    approved = _attested()
    approved["reviewers"][1]["reviewer_id"] = approved["reviewers"][0]["reviewer_id"]
    report = bench.validate_suite(approved)
    assert report["review_gate"] == "REVIEW_REQUIRED"
    assert "DISTINCT_REVIEWERS_REQUIRED" in report["review_blockers"]
    approved = _attested()
    approved["provenance_url"] = "file:///etc/passwd"
    assert "PROVENANCE_MISSING" in bench.validate_suite(approved)["review_blockers"]


def test_stdin_is_only_fixed_readonly_oci_input_and_never_shell_interpolated(tmp_path, monkeypatch):
    cmd = build_oci_command(
        "docker", image="eclipse-temurin:21-jdk", mount_dir=tmp_path / "source",
        cid_file=tmp_path / "cid", stdin_file_present=True)
    assert "--network=none" in cmd and "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd and "--pull=never" in cmd
    assert " < /input/stdin.txt" in cmd[-1]
    assert "java -Xmx64m" in cmd[-1]
    assert " < /input/stdin.txt" not in build_oci_command(
        "docker", image="eclipse-temurin:21-jdk",
        mount_dir=tmp_path, cid_file=tmp_path / "cid")[-1]

    monkeypatch.setattr("sagallm.java_behavior.oci_image_ready",
                        lambda backend, image: True)
    seen = []
    def fake_launcher(argv, timeout):
        assert argv[-1].endswith(" < /input/stdin.txt")
        mount = argv[argv.index("--mount") + 1]
        src = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
        assert (src / "stdin.txt").read_text(encoding="utf-8") == "3\nsecret\n"
        seen.append(True)
        return 0, b"ok\n"
    monkeypatch.setattr("sagallm.java_behavior._bounded_capture", fake_launcher)
    result = run_java_in_oci(
        "public class Main {}",
        "ok\n", stdin_data="3\nsecret\n",
        allow_execution=True)
    assert seen and result["status"] == "PASS"
    assert "secret" not in json.dumps(result)
    with pytest.raises(ValueError):
        run_java_in_oci("public class Main {}", "", stdin_data="x" * 4097,
                        allow_execution=True)



def test_stdout_grading_ignores_stderr_from_subprocess():
    # Safe Python fixture, NOT Java model code or a container bypass.
    import sys
    from sagallm.java_behavior import _bounded_capture
    code, output = _bounded_capture([
        sys.executable, "-c",
        "import sys;sys.stderr.write('expected-but-wrong\\n');"
        "sys.stdout.write('actual\\n')",
    ], timeout=4)
    assert code == 0
    assert output == b"actual\n"
