"""A passing test never represents real OCI isolation unless a daemon ran."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from sagallm.java_behavior import (
    MAX_OUTPUT_BYTES, _bounded_capture, build_oci_command,
    oci_available, run_java_in_oci, task_expected_stdout,
)
from sagallm.java_generated_data import build_java_dataset
from scripts.evaluate_java_behavior import evaluate


def test_oci_command_restricts_execution(tmp_path):
    argv = build_oci_command("docker", image="eclipse-temurin:21-jdk",
                             mount_dir=tmp_path / "source", cid_file=tmp_path / "cid")
    joined = " ".join(argv)
    for token in ("--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
                  "--security-opt=no-new-privileges", "--pids-limit=64", "--memory=384m",
                  "--memory-swap=384m", "--cpus=1.0", "--user=65534:65534",
                  "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=32m,mode=1777", "readonly",
                  "javac -proc:none"):
        assert token in joined
    assert "--log-driver=none" in argv
    assert "--entrypoint=/bin/sh" in argv
    assert argv[-2] == "-c"
    with pytest.raises(ValueError):
        build_oci_command("docker", image="--privileged", mount_dir=tmp_path, cid_file=tmp_path / "cid")
    with pytest.raises(ValueError):
        build_oci_command("bash", image="valid", mount_dir=tmp_path, cid_file=tmp_path / "cid")


def test_no_automatic_java_execution(tmp_path, monkeypatch):
    def should_not_execute(*args, **kwargs):
        raise AssertionError("OCI process was invoked without explicit opt-in")
    monkeypatch.setattr("sagallm.java_behavior._bounded_capture", should_not_execute)
    source = 'public class Main { public static void main(String[] args) {System.out.println("hi");}}'
    assert run_java_in_oci(source, "hi\n")["status"] == "NOT_RUN"
    dataset = tmp_path / "dataset"
    build_java_dataset(dataset)
    report = evaluate(dataset, tmp_path / "missing.pt", tmp_path / "missing.json")
    assert report["status"] == "NOT_RUN" and report["evaluated_items"] == 0
    assert report["behavior_passes"] is None


def test_missing_runtime_explicit_not_run(monkeypatch):
    monkeypatch.setattr("sagallm.java_behavior.oci_image_ready", lambda backend, image: False)
    result = run_java_in_oci("public class Main {}", "", allow_execution=True)
    assert result["status"] == "NOT_RUN" and result["behavior_match"] is None


def test_forbid_remote_docker(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://other-server:2375")
    assert oci_available("docker") is False
    with pytest.raises(ValueError):
        oci_available("command")


def test_opt_in_oci_pass_and_fail_with_mocked_launcher(monkeypatch):
    # Simulated launcher verifies command and read-only staging, not sandbox security.
    monkeypatch.setattr("sagallm.java_behavior.oci_image_ready", lambda backend, image: True)
    source = 'public class Main {public static void main(String[] args){System.out.println("hi");}}'
    def fake_launcher(argv, timeout):
        assert argv[:2] == ["docker", "run"]
        assert "--network=none" in argv and "--pull=never" in argv
        mount = argv[argv.index("--mount") + 1]
        input_dir = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
        assert (input_dir / "Main.java").read_text(encoding="utf-8") == source
        return 0, b"hi\n"
    monkeypatch.setattr("sagallm.java_behavior._bounded_capture", fake_launcher)
    assert run_java_in_oci(source, "hi\n", allow_execution=True)["status"] == "PASS"
    bad = run_java_in_oci(source, "other\n", allow_execution=True)
    assert bad["status"] == "FAIL" and bad["behavior_match"] is False
    huge = run_java_in_oci("x" * 25000, "", allow_execution=True)
    assert huge["status"] == "REJECTED"


def test_bounded_capture_rejects_large_stdout_and_hangs():
    if os.name != "posix":
        pytest.skip("POSIX selectors only")
    with pytest.raises(OverflowError):
        _bounded_capture([sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'x'*70000)"], 4)
    with pytest.raises(TimeoutError):
        _bounded_capture([sys.executable, "-c", "import time;time.sleep(5)"], 0.15)


def test_heldout_oracle_is_deterministic_and_rejects_invalid_tasks():
    assert task_expected_stdout("fibonacci-00", "fibonacci").startswith("0\n1\n")
    assert task_expected_stdout("prime-00", "prime") in ("true\n", "false\n")
    assert task_expected_stdout("palindrome-00", "palindrome") == "false\n"
    with pytest.raises(ValueError):
        task_expected_stdout("train-00", "hello")
    with pytest.raises(ValueError):
        task_expected_stdout("prime-10", "prime")
    with pytest.raises(ValueError):
        task_expected_stdout("prime-00", "fibonacci")


@pytest.mark.skipif(shutil.which("javac") is None or shutil.which("java") is None,
                    reason="Local JDK is absent; only project-authored reference code would execute")
def test_oracle_matches_all_15_trusted_reference_programs(tmp_path):
    """Only trusted project-authored reference Java may execute on the host."""
    corpus = tmp_path / "corpus"
    build_java_dataset(corpus)
    for line in (corpus / "test.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        work = tmp_path / "java" / row["id"]
        work.mkdir(parents=True)
        (work / "Main.java").write_text(row["messages"][1]["content"], encoding="utf-8")
        compile_run = subprocess.run(["javac", "-proc:none", "Main.java"], cwd=work,
                                     capture_output=True, timeout=12)
        assert compile_run.returncode == 0, (row["id"], compile_run.stderr)
        runtime = subprocess.run(["java", "-cp", str(work), "Main"], cwd=work,
                                 capture_output=True, timeout=6)
        assert runtime.returncode == 0, (row["id"], runtime.stderr)
        assert runtime.stdout.decode("utf-8") == task_expected_stdout(row["id"], row["family"])