"""Public model Java executes only after OCI smoke, pinned hashes, explicit opt-in."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from scripts import evaluate_java_public_behavior_oci as exp

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples/java_external_io_INTAKE.json"
IMAGE = "sha256:" + "a" * 64


def test_frozen_four_sources_are_previous_javac_passes_and_case_count():
    tasks, ids, suite_sha = exp.freeze_inputs(SUITE)
    assert ids == list(exp.PROMPTS)
    assert suite_sha == exp.PINNED_SUITE_SHA256
    assert sum(len(tasks[tid]["cases"]) for arm in
               exp.VERIFIED_COMPILED_OUTPUTS.values() for tid in arm) == 19
    assert set(exp.VERIFIED_COMPILED_OUTPUTS["v017"]) == {
        "ex_luhn", "ex_atbash_cipher"}
    assert set(exp.VERIFIED_COMPILED_OUTPUTS["v018"]) == {
        "ex_hamming", "ex_luhn"}
    assert all(len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)
               for arm in exp.VERIFIED_COMPILED_OUTPUTS.values()
               for sha in arm.values())
    assert exp.EXPECTED_EVALUATED_CASES == 19


def test_explicit_opt_in_and_vm_ack_are_required_before_any_oci_probes(monkeypatch):
    def fail(*a, **k):
        raise AssertionError("Should not inspect OCI without dual consent")
    monkeypatch.setattr(exp, "oci_image_ready", fail)
    with pytest.raises(PermissionError):
        exp.run(Path("missing"), Path("missing"), Path("missing"), SUITE,
                image=IMAGE, allow_public_unreviewed=False,
                acknowledge_disposable_vm=True)
    with pytest.raises(PermissionError):
        exp.run(Path("missing"), Path("missing"), Path("missing"), SUITE,
                image=IMAGE, allow_public_unreviewed=True,
                acknowledge_disposable_vm=False)
    with pytest.raises(ValueError, match="sha256 IMAGE ID"):
        exp.run(Path("missing"), Path("missing"), Path("missing"), SUITE,
                image="eclipse-temurin:21-jdk", allow_public_unreviewed=True,
                acknowledge_disposable_vm=True)


def test_smoke_must_pass_before_model_outputs_are_generated(monkeypatch):
    monkeypatch.setattr(exp, "oci_image_ready", lambda *a: True)
    calls = []
    def fake_runtime(source, expected, **kwargs):
        calls.append((source, expected, kwargs))
        return {"status": "NOT_RUN", "behavior_match": None}
    monkeypatch.setattr(exp, "run_java_in_oci", fake_runtime)
    with pytest.raises(RuntimeError, match="abort BEFORE any generated Java"):
        exp.run(Path("missing"), Path("missing"), Path("missing"), SUITE,
                image=IMAGE, allow_public_unreviewed=True,
                acknowledge_disposable_vm=True)
    assert len(calls) == 1
    assert "java.util.Scanner" in calls[0][0]
    assert calls[0][1] == "SAGALLM\n"
    assert calls[0][2]["stdin_data"] == "sagallm\n"
    assert calls[0][2]["allow_execution"] is True
    assert calls[0][2]["image"] == IMAGE


def test_expected_original_io_is_only_passed_to_oci_no_host_java(monkeypatch):
    tasks, _, _ = exp.freeze_inputs(SUITE)
    observed = []
    def fake_oci(source, expected, *, backend, image, stdin_data,
                 allow_execution, timeout):
        assert backend == "docker" and image == IMAGE and allow_execution
        assert timeout == 20
        observed.append((stdin_data, expected))
        return {"status": "PASS", "behavior_match": True}
    monkeypatch.setattr(exp, "run_java_in_oci", fake_oci)
    score = exp.assess_task("class Main{}", tasks["ex_hamming"], image=IMAGE)
    assert score["cases_passed"] == 4
    assert score["task_status"] == "PASS"
    assert score["generated_source_stored"] is False
    assert score["host_java_execution"] is False
    assert observed == [(c["stdin"], c["stdout"]) for c in tasks["ex_hamming"]["cases"]]
    assert "GGACGGATTCTG" not in json.dumps(score)


def test_separates_zero_exit_wrong_stdout_from_nonzero_java_exit(monkeypatch):
    tasks, _, _ = exp.freeze_inputs(SUITE)
    observed = iter([
        {"status": "FAIL", "returncode": 0, "behavior_match": False},
        {"status": "FAIL", "returncode": 1, "behavior_match": False},
        {"status": "PASS", "returncode": 0, "behavior_match": True},
        {"status": "TIMEOUT", "returncode": None, "behavior_match": False},
        {"status": "OUTPUT_LIMIT", "returncode": None, "behavior_match": False},
    ])
    monkeypatch.setattr(exp, "run_java_in_oci", lambda *a, **k: next(observed))
    result = exp.assess_task("class Main{}", tasks["ex_luhn"], image=IMAGE)
    assert result["cases_passed"] == 1
    assert result["task_status"] == "FAIL"
    assert result["case_failure_categories"] == [
        "OUTPUT_MISMATCH_ZERO_EXIT", "NONZERO_EXIT", "PASS",
        "TIMEOUT", "OUTPUT_LIMIT",
    ]
    assert "stdout" not in json.dumps(result).lower()
    assert "Main{}" not in json.dumps(result)


def test_unavailable_oci_never_looks_like_behavior_failure_or_success(monkeypatch):
    tasks, _, _ = exp.freeze_inputs(SUITE)
    seq = iter(["PASS", "RUNTIME_ERROR", "PASS", "FAIL", "PASS"])
    monkeypatch.setattr(exp, "run_java_in_oci", lambda *a, **k:
                        {"status": next(seq)})
    score = exp.assess_task("class Main{}", tasks["ex_luhn"], image=IMAGE)
    assert score["task_status"] == "UNAVAILABLE"
    assert score["cases_passed"] == 3
    assert score["case_statuses"] == [
        "PASS", "RUNTIME_ERROR", "PASS", "FAIL", "PASS"]


def test_changed_model_java_hash_is_rejected_before_oci(monkeypatch):
    class TinyModel:
        config = SimpleNamespace(max_position_embeddings=384)
        def generate(self, prefix, **kwargs):
            assert kwargs["temperature"] == 0.0
            assert kwargs["max_new_tokens"] <= 170
            return torch.cat([prefix, torch.tensor([[3, 2]])], dim=1)
    class FakeTokenizer:
        SPECIAL = {"<bos>": 1, "<eos>": 2}
        vocab_size = 520
        def encode(self, value):
            return [5] * 20
        def decode(self, value):
            return "public class Main { public static void main(String[] a){} }"
    def no_oci(*a, **kw):
        raise AssertionError("Mismatched output may never reach OCI")
    monkeypatch.setattr(exp, "run_java_in_oci", no_oci)
    with pytest.raises(ValueError, match="source hash mismatch"):
        exp.frozen_model_source(TinyModel(), FakeTokenizer(), "v017", "ex_luhn")


def test_public_suite_has_not_been_relabelled_as_reviewed(tmp_path):
    suite = json.loads(SUITE.read_text(encoding="utf8"))
    suite["review_status"] = "APPROVED_FOR_INDEPENDENT_EVALUATION"
    candidate = tmp_path / "changed.json"
    candidate.write_text(json.dumps(suite), encoding="utf8")
    with pytest.raises(ValueError):
        exp.freeze_inputs(candidate)


def test_original_oci_runner_has_strict_limits(tmp_path):
    from sagallm.java_behavior import build_oci_command
    cmd = build_oci_command(
        "docker", image=IMAGE, mount_dir=tmp_path / "source",
        cid_file=tmp_path / "cid", stdin_file_present=True)
    assert "--network=none" in cmd
    assert "--read-only" in cmd
    assert "--cap-drop=ALL" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--pids-limit=64" in cmd
    assert "--memory=384m" in cmd
    assert "--user=65534:65534" in cmd
    assert "--pull=never" in cmd
    assert " < /input/stdin.txt" in cmd[-1]
    assert "type=bind,src=" in cmd[cmd.index("--mount")+1]
    assert "readonly" in cmd[cmd.index("--mount")+1]
