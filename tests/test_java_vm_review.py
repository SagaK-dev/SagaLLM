"""Mock metadata only: no Docker or Java code execution."""
import hashlib
import json
from datetime import date
from types import SimpleNamespace

import pytest

from sagallm import java_vm_review as vm

IMAGE = "example.invalid/jdk@sha256:" + "a" * 64
SUITE = "b" * 64


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(vm.sys, "platform", "linux")
    for key in (*vm.CI_KEYS, "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
                "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"):
        monkeypatch.delenv(key, raising=False)


def document():
    return {"schema": vm.SCHEMA, "suite_sha256": SUITE, "image": IMAGE,
            "operator": {"id": "mock-a", "date": date.today().isoformat(),
                         "review_url": "https://example.invalid/a"},
            "reviewer": {"id": "mock-b", "date": date.today().isoformat(),
                         "review_url": "https://example.invalid/b"},
            "controls": dict.fromkeys(vm.CONTROLS, True)}


def save(tmp_path, doc):
    path = tmp_path / "review.json"
    raw = json.dumps(doc).encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def check(path=None, sha=None, **kwargs):
    return vm.review_gate(path, sha, **({"suite_sha": SUITE, "backend": "docker",
                                        "image": IMAGE} | kwargs))


def test_valid_metadata_is_only_self_attested(local, tmp_path):
    path, sha = save(tmp_path, document())
    assert check(path, sha) == ("ELIGIBLE_SELF_ATTESTED", sha)


@pytest.mark.parametrize("key", ["CI", "GITHUB_ACTIONS", "RUNNER_OS"])
def test_ci_forbidden(local, monkeypatch, key):
    monkeypatch.setenv(key, "true")
    assert check()[0] == "CI_EXECUTION_FORBIDDEN"


def test_nonlinux(local, monkeypatch):
    monkeypatch.setattr(vm.sys, "platform", "darwin")
    assert check()[0] == "LINUX_VM_REQUIRED"


def test_remote_and_tag_forbidden(local, monkeypatch):
    assert check(backend="podman")[0] == "M4_LOCAL_DOCKER_REQUIRED"
    assert check(image="eclipse-temurin:17-jdk")[0] == "PINNED_OCI_IMAGE_REQUIRED"
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.invalid:2375")
    assert check()[0] == "REMOTE_OR_CUSTOM_ENGINE_FORBIDDEN"


def test_missing_and_tampered_review(local, tmp_path):
    assert check()[0] == "PINNED_VM_REVIEW_REQUIRED"
    path, sha = save(tmp_path, document())
    path.write_bytes(path.read_bytes() + b" ")
    assert check(path, sha)[0] == "VM_REVIEW_SHA_MISMATCH"
    assert check(tmp_path / "missing", sha)[0] == "VM_REVIEW_UNAVAILABLE"


def test_scope_mismatch(local, tmp_path):
    path, sha = save(tmp_path, document())
    assert check(path, sha, suite_sha="c"*64)[0] == "VM_REVIEW_SCOPE_MISMATCH"
    assert check(path, sha, image=IMAGE.replace("a"*64, "c"*64))[0] == "VM_REVIEW_SCOPE_MISMATCH"


def test_distinct_review_and_controls(local, tmp_path):
    doc = document()
    doc["reviewer"]["id"] = " MOCK-A "
    assert check(*save(tmp_path, doc))[0] == "DISTINCT_VM_REVIEW_REQUIRED"
    doc = document()
    doc["controls"]["network_disabled"] = 1
    assert check(*save(tmp_path, doc))[0] == "VM_CONTROLS_NOT_ATTESTED"
    doc = document()
    doc["reviewer"]["date"] = "2026-02-30"
    assert check(*save(tmp_path, doc))[0] == "VM_REVIEW_INVALID"


def test_duplicate_json_and_symlink(local, tmp_path):
    path, _ = save(tmp_path, document())
    raw = b'{"schema":1,"schema":2}'
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    assert check(path, sha)[0] == "VM_REVIEW_INVALID"
    link = tmp_path / "link"
    link.symlink_to(path)
    assert check(link, sha)[0] == "VM_REVIEW_UNAVAILABLE"


def test_docker_context_only_local_mock(monkeypatch):
    calls = []
    def fake(argv, **kwargs):
        calls.append(argv)
        output = b"default\n" if argv[-1] == "show" else b"unix:///var/run/docker.sock\n"
        return SimpleNamespace(returncode=0, stdout=output)
    monkeypatch.setattr(vm.subprocess, "run", fake)
    assert vm.local_docker_context()
    assert len(calls) == 2
    monkeypatch.setattr(vm.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=b"remote\n"))
    assert not vm.local_docker_context()
