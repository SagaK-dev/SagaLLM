"""Opt-in, offline, resource-bounded OCI execution for synthetic Java evaluation.

NOT a security proof: an OCI sandbox may still have vulnerabilities. Run only
on a dedicated disposable host/VM. No generated Java ever runs directly on the
Python host. Without an explicitly enabled, locally available OCI engine the
result is NOT_RUN; compilation alone is never a behavioral success.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time

MAX_SOURCE_BYTES = 24_000
MAX_OUTPUT_BYTES = 64 * 1024
ALLOWED_BACKENDS = ("docker", "podman")
JAVA_IMAGE_DEFAULT = "eclipse-temurin:21-jdk"
# Only untrusted tasks in the v0.12 original fixture have behavioral oracles.
HELDOUT_FAMILIES = ("fibonacci", "prime", "palindrome")
WORDS = ("Saga", "Java", "学習", "Hello", "コード", "練習", "Sunny", "Blue")


def expected_stdout(family: str, index: int) -> str:
    """Independent Python oracle for the fixed original 3 test task families.

    Reproduce problem INPUT PARAMETERS, not the Java reference implementation.
    The JDK reference-output parity test independently checks these oracles.
    """
    if family not in HELDOUT_FAMILIES or not 0 <= index <= 4:
        raise ValueError("No independently defined oracle for this task")
    rng = random.Random(1117 + 101 * index + sum(ord(ch) for ch in family))
    a, _b = rng.randint(2, 90), rng.randint(2, 40)
    n = rng.randint(3, 15)
    word = WORDS[index % len(WORDS)]
    if family == "prime":
        prime = a >= 2 and all(a % divisor != 0 for divisor in range(2, int(a**0.5) + 1))
        return str(prime).lower() + "\n"
    if family == "palindrome":
        return str(word == word[::-1]).lower() + "\n"
    sequence, first, second = [], 0, 1
    for _ in range(n):
        sequence.append(str(first))
        first, second = second, first + second
    return "\n".join(sequence) + "\n"


def task_expected_stdout(task_id: str, family: str) -> str:
    match = re.fullmatch(re.escape(family) + r"-([0-9]{2})", task_id)
    if not match:
        raise ValueError("Task ID does not match its task family")
    return expected_stdout(family, int(match.group(1)))


def oci_available(backend: str) -> bool:
    if backend not in ALLOWED_BACKENDS:
        raise ValueError("Only docker/podman backends are accepted")
    if os.name != "posix":
        return False
    # Refuse inadvertently remote Docker contexts that would undermine local mount assumptions.
    if backend == "docker" and (os.environ.get("DOCKER_HOST", "") not in ("", "unix:///var/run/docker.sock")
                                or os.environ.get("DOCKER_CONTEXT", "") not in ("", "default")):
        return False
    return shutil.which(backend) is not None


def oci_image_ready(backend: str, image: str) -> bool:
    """Only run with a reachable local engine and an image already present."""
    if not oci_available(backend):
        return False
    if not (1 <= len(image) <= 200 and re.fullmatch(r"[A-Za-z0-9._/@:-]+", image)
            and not image.startswith("-") and "/../" not in image):
        raise ValueError("Unsafe OCI image identifier")
    try:
        check = subprocess.run([backend, "image", "inspect", image],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=5, check=False)
        return check.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def build_oci_command(backend: str, *, image: str, mount_dir: Path, cid_file: Path,
                      stdin_file_present: bool = False) -> list[str]:
    """Return an argv list; no generated text is ever interpolated as shell code."""
    if backend not in ALLOWED_BACKENDS:
        raise ValueError("Unsupported local OCI backend")
    if not (1 <= len(image) <= 200 and re.fullmatch(r"[A-Za-z0-9._/@:-]+", image) and
            not image.startswith("-") and "/../" not in image):
        raise ValueError("Unsafe OCI image identifier")
    # Container shell script is static; the untrusted program is read-only input data.
    shell_script = (
        "mkdir -p /tmp/classes && "
        "javac -proc:none -encoding UTF-8 -d /tmp/classes /input/Main.java && "
        "java -Xmx64m -Dfile.encoding=UTF-8 -cp /tmp/classes Main"
        + (" < /input/stdin.txt" if stdin_file_present else "")
    )
    return [
        backend, "run", "--rm", "--pull=never", "--cidfile", str(cid_file),
        "--network=none", "--read-only", "--cap-drop=ALL", "--log-driver=none",
        "--security-opt=no-new-privileges", "--pids-limit=64",
        "--memory=384m", "--memory-swap=384m", "--cpus=1.0",
        "--user=65534:65534", "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=32m,mode=1777",
        "--mount", "type=bind,src=" + str(mount_dir) + ",dst=/input,readonly",
        "--workdir=/tmp", "--entrypoint=/bin/sh", image, "-c", shell_script,
    ]


def _bounded_capture(argv: list[str], timeout: float, output_cap: int = MAX_OUTPUT_BYTES) -> tuple[int, bytes]:
    """Avoid unbounded host-side stdout capture, even if model code prints forever."""
    if os.name != "posix":
        raise RuntimeError("Bounded OCI execution currently requires POSIX (use Linux/WSL)")
    # Evaluate stdout only. Compiler/runtime stderr must never make a wrong
    # stdout-only model answer appear correct, and is not persisted.
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    assert process.stdout is not None
    observed = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    try:
        while selector.get_map():
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError("OCI Java process timed out")
            for key, _ in selector.select(min(left, 0.2)):
                piece = os.read(key.fd, min(4096, output_cap + 1 - len(observed)))
                if not piece:
                    selector.unregister(key.fileobj)
                    continue
                observed.extend(piece)
                if len(observed) > output_cap:
                    raise OverflowError("OCI Java output exceeded bounded capture limit")
        # The process can linger after closing output; do not wait indefinitely.
        left = max(0.001, deadline - time.monotonic())
        return process.wait(timeout=left), bytes(observed)
    finally:
        selector.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        process.stdout.close()


def _cleanup_stale_container(backend: str, cid_file: Path) -> None:
    """Best effort cleanup if docker CLI dies before `--rm` can take effect."""
    try:
        cid = cid_file.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-f]{12,64}", cid):
            subprocess.run([backend, "rm", "-f", cid], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=3, check=False)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_java_in_oci(source: str, expected: str, *, backend: str = "docker",
                    image: str = JAVA_IMAGE_DEFAULT, allow_execution: bool = False,
                    timeout: float = 12.0, stdin_data: str = "") -> dict:
    """Run a generated single-file Java program only with explicit local OCI opt-in.

    Does not pull images, mount the user's working directory or forward network.
    Output matching is an exact stdout comparison, NOT a security attestation.
    """
    if not allow_execution:
        return {"status": "NOT_RUN", "reason": "Explicit allow_execution required", "behavior_match": None}
    if not 2 <= timeout <= 30:
        raise ValueError("OCI runtime timeout must be 2..30 seconds")
    if backend not in ALLOWED_BACKENDS:
        raise ValueError("Unsupported backend")
    if not oci_image_ready(backend, image):
        return {"status": "NOT_RUN", "reason": "Local OCI engine or preinstalled image unavailable", "behavior_match": None}
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_SOURCE_BYTES or "\x00" in source:
        return {"status": "REJECTED", "reason": "Invalid/oversized Java source", "behavior_match": None}
    if not isinstance(expected, str) or len(expected.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ValueError("Invalid/oversized reference output")
    if (not isinstance(stdin_data, str) or "\x00" in stdin_data or
            len(stdin_data.encode("utf-8")) > 4096):
        raise ValueError("Invalid/oversized standard input")
    with tempfile.TemporaryDirectory(prefix="sagallm-oci-java-") as td:
        root = Path(td)
        source_dir = root / "source"
        source_dir.mkdir(mode=0o755)
        source_file = source_dir / "Main.java"
        source_file.write_text(source, encoding="utf-8")
        source_file.chmod(0o644)
        if stdin_data:
            stdin_file = source_dir / "stdin.txt"
            stdin_file.write_text(stdin_data, encoding="utf-8")
            stdin_file.chmod(0o644)
        # Docker's unprivileged uid must traverse all source path components.
        root.chmod(0o755)
        cid_file = root / "container.cid"
        cmd = build_oci_command(backend, image=image, mount_dir=source_dir,
                                cid_file=cid_file, stdin_file_present=bool(stdin_data))
        try:
            code, output = _bounded_capture(cmd, timeout)
        except (TimeoutError, subprocess.TimeoutExpired):
            _cleanup_stale_container(backend, cid_file)
            return {"status": "TIMEOUT", "behavior_match": False, "returncode": None}
        except OverflowError:
            _cleanup_stale_container(backend, cid_file)
            return {"status": "OUTPUT_LIMIT", "behavior_match": False, "returncode": None}
        except OSError:
            _cleanup_stale_container(backend, cid_file)
            return {"status": "RUNTIME_ERROR", "behavior_match": None, "returncode": None}
        if code in (125, 126, 127):
            return {"status": "RUNTIME_ERROR", "behavior_match": None,
                    "reason": "OCI launcher failed; not a Java behavior failure", "returncode": code}
        match = (code == 0 and output == expected.encode("utf-8"))
        return {"status": "PASS" if match else "FAIL", "behavior_match": match,
                "returncode": code, "observed_stdout_sha256": hashlib.sha256(output).hexdigest(),
                "expected_stdout_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
                "isolation": "OCI_BEST_EFFORT_NOT_SECURITY_PROOF",
                "network": "none", "image_pull": "never"}