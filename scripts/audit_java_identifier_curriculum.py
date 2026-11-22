"""Build and compile-check a separate train-only Java identifier candidate set.

The generated candidate dataset is not used for SFT, not mixed into test or
validation, and does not assert model quality. Only javac is invoked; no java
program is ever run. Requires a local Java 17+ JDK.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from sagallm.java_identifier_curriculum import (
    NAMES, VARIABLES, build_identifier_train_candidates,
    verify_identifier_train_candidates,
)


def compile_candidates(folder: Path, *, timeout: int = 120) -> dict:
    # Absolute paths are required because javac is run with cwd=compile_root.
    folder = folder.resolve()
    if not 1 <= timeout <= 180:
        raise ValueError("Invalid compiler timeout")
    java_compiler = shutil.which("javac")
    if java_compiler is None:
        raise RuntimeError("JDK javac is required for candidate verification")
    data = folder / "train-identifier-variants.jsonl"
    rows = [json.loads(line) for line in data.read_text(encoding="utf8").splitlines()]
    compile_root = folder / "_compile_only"
    if compile_root.exists() or compile_root.is_symlink():
        raise FileExistsError("Refuse to overwrite compiler scratch output")
    compile_root.mkdir()
    source_files = []
    try:
        for index, row in enumerate(rows):
            source = row["messages"][1]["content"]
            cls = f"Candidate{index:04d}"
            if source.count("public class Main") != 1:
                raise ValueError("Unexpected Java class declaration")
            # Rewriting Main is for isolated compilation only, not training.
            renamed = source.replace("public class Main", f"public class {cls}", 1)
            java_file = compile_root / f"{cls}.java"
            java_file.write_text(renamed, encoding="utf8")
            source_files.append(str(java_file))
        outcome = subprocess.run(
            [java_compiler, "-J-Xmx256m", "-proc:none",
             "-encoding", "UTF-8", "--release", "17",
             "-d", str(compile_root), *source_files],
            cwd=compile_root, timeout=timeout, capture_output=True,
            text=True, errors="replace", check=False,
        )
        # Never persist javac stdout/stderr, source snippets or private paths.
        passed = outcome.returncode == 0
        # Only finite, allowlisted diagnostics are emitted; never store the
        # compiler's raw stderr or its temporary source paths.
        failed = sorted({int(index) for index in
                         re.findall(r"Candidate([0-9]{4})\\.java:\\d+: error:",
                                    outcome.stderr) if int(index) < len(rows)})
        stderr_lower = outcome.stderr.lower()
        if passed:
            failure_kind = "NONE"
        elif "cannot find symbol" in stderr_lower:
            failure_kind = "MISSING_SYMBOL"
        elif "already defined" in stderr_lower:
            failure_kind = "DUPLICATE_DECLARATION"
        elif "expected" in stderr_lower or "illegal start" in stderr_lower:
            failure_kind = "SYNTAX_ERROR"
        else:
            failure_kind = "OTHER"
        return {
            "status": "PASS" if passed else "FAIL",
            "compiled": len(rows) if passed else None,
            "attempted": len(rows),
            "failed_candidate_indices": failed[:10],
            "failure_kind": failure_kind,
            "compiler_exit_code": outcome.returncode,
            "compile_only": True,
            "java_executed": False,
            "javac_annotation_processing": "DISABLED",
            "raw_javac_diagnostics_persisted": False,
        }
    finally:
        shutil.rmtree(compile_root)


def audit(folder: Path, *, cases_per_family: int = 6,
          variants: int = 2) -> dict:
    created = build_identifier_train_candidates(
        folder, cases_per_family=cases_per_family, variants=variants)
    verified = verify_identifier_train_candidates(folder)
    result = compile_candidates(folder)
    # The candidate names appear only in *training* source text, not in
    # diagnostics. Diversity is a corpus statistic, not a quality score.
    rows = [json.loads(line) for line in
            (folder / created["candidate_file"]).read_text(encoding="utf8").splitlines()]
    found = {name for name in NAMES
             if any(re.search(r"(?<![A-Za-z0-9_$])" + re.escape(name) +
                              r"(?![A-Za-z0-9_$])",
                              row["messages"][1]["content"]) for row in rows)}
    summary = {
        "status": "CANDIDATE_TRAIN_ONLY_NOT_USED_FOR_TRAINING",
        "schema": created["schema"],
        "candidate_sha256": created["candidate_sha256"],
        "families": len(VARIABLES),
        "candidate_examples": len(rows),
        "original_identifier_vocabulary_in_scope":
            len(set().union(*[set(v) for v in VARIABLES.values()])),
        "new_identifier_vocabulary_used": len(found),
        "reproducibility": verified["status"],
        "compiler": result,
        "canonical_dataset_unchanged": True,
        "used_for_training": False,
        "validation_or_test_tasks_used_for_augmentation": False,
        "generated_java_executed": False,
        "generalization_improvement_established": False,
    }
    (folder / "audit-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    if result["status"] != "PASS":
        raise RuntimeError(
            "Compiler rejected candidate training examples; "
            f"category={result['failure_kind']}; "
            f"indices={result['failed_candidate_indices']}; "
            f"exit={result['compiler_exit_code']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases-per-family", type=int, default=6)
    parser.add_argument("--variants", type=int, default=2)
    args = parser.parse_args()
    report = audit(
        args.output, cases_per_family=args.cases_per_family,
        variants=args.variants)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
