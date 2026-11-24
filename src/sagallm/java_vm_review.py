"""Offline M4 gate: metadata checks do not prove disposable VM isolation."""
from __future__ import annotations

from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit

SCHEMA = "sagallm-java-m4-vm-review-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:-]{0,126}@sha256:[0-9a-f]{64}\Z")
CONTROLS = ["disposable_linux_vm","no_host_secrets","no_persistent_mounts","no_engine_socket_mount","network_disabled","nonroot","resource_limits_reviewed","image_reviewed","jdk17_verified","destroy_after_run","local_engine_reviewed"]
CI_KEYS = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "TF_BUILD", "JENKINS_URL",
           "BUILDKITE", "CIRCLECI", "RUNNER_OS")
MAX_REVIEW_BYTES = 16384


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _person(record):
    if not isinstance(record, dict) or set(record) != {"id", "date", "review_url"}:
        return False
    ident, day, url = record["id"], record["date"], record["review_url"]
    if not isinstance(ident, str) or not 1 <= len(ident.strip()) <= 100:
        return False
    if not isinstance(day, str) or not re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", day):
        return False
    try:
        if date.fromisoformat(day) > date.today():
            return False
        if not isinstance(url, str) or len(url) > 500 or any(c.isspace() for c in url):
            return False
        parts = urlsplit(url)
        return parts.scheme == "https" and bool(parts.hostname) and not (
            parts.username or parts.password)
    except ValueError:
        return False


def review_gate(path: Path | None, expected_sha: str | None, *,
                suite_sha: str, backend: str, image: str) -> tuple[str, str | None]:
    if any(os.environ.get(k) for k in CI_KEYS):
        return "CI_EXECUTION_FORBIDDEN", None
    if sys.platform != "linux":
        return "LINUX_VM_REQUIRED", None
    if backend != "docker":
        return "M4_LOCAL_DOCKER_REQUIRED", None
    if (os.environ.get("DOCKER_HOST", "") not in ("", "unix:///var/run/docker.sock")
            or os.environ.get("DOCKER_CONTEXT", "") not in ("", "default")
            or any(os.environ.get(k) for k in ("DOCKER_CONFIG", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"))):
        return "REMOTE_OR_CUSTOM_ENGINE_FORBIDDEN", None
    if (not isinstance(image, str) or not IMAGE.fullmatch(image)
            or ".." in image or "//" in image):
        return "PINNED_OCI_IMAGE_REQUIRED", None
    if path is None or not isinstance(expected_sha, str) or not SHA.fullmatch(expected_sha):
        return "PINNED_VM_REVIEW_REQUIRED", None
    try:
        if path.is_symlink() or not path.is_file():
            return "VM_REVIEW_UNAVAILABLE", None
        with path.open("rb") as handle:
            raw = handle.read(MAX_REVIEW_BYTES + 1)
        if len(raw) > MAX_REVIEW_BYTES:
            return "VM_REVIEW_INVALID", None
        fingerprint = hashlib.sha256(raw).hexdigest()
        if fingerprint != expected_sha:
            return "VM_REVIEW_SHA_MISMATCH", None
        review = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(review, dict) or set(review) != {
            "schema", "suite_sha256", "image", "operator", "reviewer", "controls"
        }:
            return "VM_REVIEW_INVALID", None
        if (review["schema"] != SCHEMA or review["suite_sha256"] != suite_sha
                or not SHA.fullmatch(suite_sha) or review["image"] != image):
            return "VM_REVIEW_SCOPE_MISMATCH", None
        if not _person(review["operator"]) or not _person(review["reviewer"]):
            return "VM_REVIEW_INVALID", None
        if (review["operator"]["id"].strip().casefold()
                == review["reviewer"]["id"].strip().casefold()
                or review["operator"]["review_url"] == review["reviewer"]["review_url"]):
            return "DISTINCT_VM_REVIEW_REQUIRED", None
        controls = review["controls"]
        if (not isinstance(controls, dict) or set(controls) != set(CONTROLS)
                or any(value is not True for value in controls.values())):
            return "VM_CONTROLS_NOT_ATTESTED", None
    except (OSError, ValueError, UnicodeError, RecursionError):
        return "VM_REVIEW_INVALID", None
    return "ELIGIBLE_SELF_ATTESTED", fingerprint


def local_docker_context() -> bool:
    """Check the default local endpoint before asking the engine for an image."""
    try:
        for args, expected in (
            (["docker", "context", "show"], "default"),
            (["docker", "context", "inspect", "default", "--format",
              "{{.Endpoints.docker.Host}}"], "unix:///var/run/docker.sock"),
        ):
            result = subprocess.run(args, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                    timeout=5, check=False)
            if result.returncode or result.stdout.decode("utf-8").strip() != expected:
                return False
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        return False
    return True


def local_image_has_no_volumes(image: str) -> bool:
    """Reject image-declared writable anonymous volumes for this path."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .Config.Volumes}}", image],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=5, check=False)
        if result.returncode or len(result.stdout) > 16384:
            return False
        volumes = json.loads(result.stdout)
        return volumes is None or volumes == {}
    except (OSError, ValueError, UnicodeError, subprocess.TimeoutExpired):
        return False
