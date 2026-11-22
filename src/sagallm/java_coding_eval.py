"""Original Java coding fixtures and compile-only checkpoint evaluation.

All tasks are generated from SagaLLM's own source templates; no external dataset
is downloaded. Held-out *task families* are never present in training examples.
Compiling generated code is not proof of correct behavior or a security sandbox.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from .java_ai import _model_user_message, generate_revision
from .java_vibe import create_project, render

SCHEMA = "sagallm-java-coding-fixture-v1"
MAX_BYTES = 2_000_000
BASE_SOURCE = render("hello", "Main")
# Entire template families are held out; not random row-level splits.
SPLIT_FAMILIES = {
    "train": ("hello", "fizzbuzz", "todo", "guess", "file", "swing"),
    "validation": ("prime",),
    "test": ("calculator", "factorial"),
}
TASKS = (
    ("hello-ja", "hello", "Change the greeting to こんにちは、Java学習！", {"message": "こんにちは、Java学習！"}),
    ("hello-en", "hello", "Print Hello from SagaLLM", {"message": "Hello from SagaLLM"}),
    ("hello-hello", "hello", "Print Hello, world!", {"message": "Hello, world!"}),
    ("fizzbuzz-20", "fizzbuzz", "Implement FizzBuzz from 1 to 20", {"limit": 20}),
    ("fizzbuzz-30", "fizzbuzz", "Implement FizzBuzz from 1 to 30", {"limit": 30}),
    ("fizzbuzz-50", "fizzbuzz", "Implement FizzBuzz from 1 to 50", {"limit": 50}),
    ("todo-1", "todo", "Create a console-based TODO list that can add, list and remove tasks", {}),
    ("guess-1", "guess", "Implement a number guessing game for numbers from 1 to 100", {}),
    ("file-1", "file", "Read a UTF-8 text file specified on the command line", {}),
    ("swing-1", "swing", "Create a small Swing GUI with a button", {}),
    ("prime-1", "prime", "Implement an integer prime checker", {}),
    ("calculator-1", "calculator", "Build a console calculator with + - * /", {}),
    ("factorial-1", "factorial", "Compute the factorial of an integer using BigInteger", {}),
)


def _canonical(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _path(root: Path, filename: str) -> Path:
    # Never accept a manifest-controlled filesystem path or symlink.
    if filename not in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        raise ValueError("Unexpected manifest file path")
    path = root / filename
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Missing, unsafe or oversized Java dataset file: " + filename)
    return path


def _reference(task: tuple) -> str:
    _, family, _, options = task
    return render(family, "Main", options)


def build_coding_fixture(destination: str | Path) -> dict:
    """Create a small, explicitly synthetic, family-separated SFT/eval fixture."""
    root = Path(destination).expanduser().absolute()
    if root.exists():
        raise FileExistsError("Destination already exists; Java fixture must not overwrite")
    root.mkdir(parents=True)
    try:
        info = {}
        for split in ("train", "validation", "test"):
            rows = []
            for task in TASKS:
                task_id, family, instruction, options = task
                if family not in SPLIT_FAMILIES[split]:
                    continue
                row = {"id": task_id, "family": family,
                       "messages": [{"role": "user", "content": _model_user_message(instruction, BASE_SOURCE, "Main")},
                                    {"role": "assistant", "content": _reference(task)}]}
                rows.append(_canonical(row))
            name = split + ".jsonl"
            raw = "\n".join(rows) + "\n"
            (root / name).write_text(raw, encoding="utf-8")
            info[split] = {"file": name, "sha256": _digest(raw), "examples": len(rows),
                           "families": list(SPLIT_FAMILIES[split])}
        manifest = {
            "schema": SCHEMA,
            "origin": "original source templates implemented in SagaLLM; no third-party Java corpus",
            "external_data_used": False,
            "dataset_license_review": "NOT_ATTESTED",
            "use": "CPU SFT and regression experiments only; not production-quality Java model training",
            "split_strategy": "disjoint task/template families, not row-level random split",
            "split_files": info,
        }
        (root / "manifest.json").write_text(_canonical(manifest) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(root)
        raise
    return {"status": "CREATED", "directory": str(root), "schema": SCHEMA,
            "counts": {k: v["examples"] for k, v in info.items()},
            "license_review": "NOT_ATTESTED", "capability": "NOT_ESTABLISHED"}


def verify_coding_fixture(directory: str | Path) -> dict:
    """Verify hashes, split separation, reference templates and SFT rows."""
    root = Path(directory).expanduser().absolute()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink() or manifest_path.stat().st_size > 65536:
        raise ValueError("Missing or unsafe manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("external_data_used") is not False:
        raise ValueError("Unrecognized fixture schema or unsupported external data")
    if manifest.get("split_strategy") != "disjoint task/template families, not row-level random split":
        raise ValueError("Invalid split strategy")
    if manifest.get("dataset_license_review") != "NOT_ATTESTED":
        raise ValueError("Synthetic fixture does not certify a legal license review")
    if set(manifest.get("split_files", {})) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test split files")
    expected = {task[0]: task for task in TASKS}
    seen, observed = set(), {}
    for split in ("train", "validation", "test"):
        record = manifest["split_files"][split]
        name = split + ".jsonl"
        if record.get("file") != name or record.get("families") != list(SPLIT_FAMILIES[split]):
            raise ValueError("Split inventory mismatch for " + split)
        file = _path(root, name)
        raw = file.read_bytes()
        if _digest(raw) != record.get("sha256"):
            raise ValueError("Dataset hash mismatch for " + name)
        items = []
        for line in raw.decode("utf-8").splitlines():
            if not line or len(line.encode("utf-8")) > 65536:
                raise ValueError("Invalid or oversized dataset row")
            item = json.loads(line)
            if not isinstance(item, dict) or set(item) != {"id", "family", "messages"}:
                raise ValueError("Unsupported SFT row")
            task = expected.get(item["id"])
            if task is None or item["id"] in seen or task[1] != item["family"]:
                raise ValueError("Unknown, duplicated or mismatched Java task")
            if task[1] not in SPLIT_FAMILIES[split]:
                raise ValueError("Train/evaluation family overlap detected")
            truth = [
                {"role": "user", "content": _model_user_message(task[2], BASE_SOURCE, "Main")},
                {"role": "assistant", "content": _reference(task)},
            ]
            if item["messages"] != truth:
                raise ValueError("Unexpected instruction/answer or reference code for " + item["id"])
            seen.add(item["id"])
            items.append(item)
        if len(items) != record.get("examples") or len(items) == 0:
            raise ValueError("Split count mismatch")
        observed[split] = len(items)
    if seen != set(expected):
        raise ValueError("Fixture is missing expected tasks")
    return {"status": "PASS", "schema": SCHEMA, "counts": observed,
            "split_integrity": "PASS", "dataset_license_review": manifest.get("dataset_license_review"),
            "training_data_license_status": "ORIGINAL_LOCAL_FIXTURE_ONLY",
            "limits": "Reference answers share scaffolding; task families are held out; no real-world generalization claim"}


def evaluate_java_compilation(directory: str | Path, *, generator: Callable[..., str],
                              split: str = "test", max_examples: int = 10,
                              max_new_tokens: int = 512, attempts: int = 1) -> dict:
    """Score model proposals by javac and strict reference equality; never run Java.

    The generator should be a bounded SagaLLM local checkpoint adapter. It is
    injectable to make the protocol independently testable on CPU without a
    trained coding model. No model-generated program is executed.
    """
    if split not in ("validation", "test") or not 1 <= max_examples <= 20:
        raise ValueError("Choose held-out validation/test and 1..20 examples")
    verified = verify_coding_fixture(directory)
    root = Path(directory).expanduser().absolute()
    tasks = {task[0]: task for task in TASKS}
    items = [json.loads(line) for line in _path(root, split + ".jsonl").read_text(encoding="utf-8").splitlines()]
    results = []
    for item in items[:max_examples]:
        task = tasks[item["id"]]
        with tempfile.TemporaryDirectory(prefix="sagallm-java-heldout-") as tmp:
            project = Path(tmp) / "java"
            create_project(project, "Hello world", template="hello")
            try:
                result = generate_revision(project, task[2], generator=generator,
                                           max_new_tokens=max_new_tokens, attempts=attempts,
                                           return_candidate=True)
                candidate = result.pop("_candidate", None)
                compiled = result["status"] == "PREVIEW"
                reference_match = bool(candidate and candidate == _reference(task))
                status = "COMPILED" if compiled else result["status"]
            except (ValueError, RuntimeError, TimeoutError) as exc:
                compiled = reference_match = False
                status = "REJECTED"
            results.append({"id": item["id"], "family": item["family"],
                            "status": status, "compiled": compiled,
                            "exact_reference_match": reference_match,
                            "runtime_tests": "NOT_RUN"})
    done = sum(x["compiled"] for x in results)
    return {
        "protocol": "sagallm-java-compile-only-v1",
        "data_integrity": verified["status"],
        "evaluation_split": split,
        "dataset_sha256": _digest(_path(root, split + ".jsonl").read_bytes()),
        "examples": len(results),
        "compile_passes": done,
        "compile_pass_rate": done / len(results),
        "exact_reference_matches": sum(x["exact_reference_match"] for x in results),
        "runtime_behavior_test": "NOT_RUN",
        "security": "javac -proc:none is not a sandbox; no generated program executed",
        "comparability": "SYNTHETIC_FIXTURE_ONLY_NOT_PUBLIC_BENCHMARK",
        "items": results,
    }