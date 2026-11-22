"""Analyze recorded v0.18 compile regressions without running generated Java.

This is a report-only diagnostic: hashes cannot reconstruct Java source or
javac errors. No claim about root cause is made without those artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(report: dict) -> dict:
    if report.get("status") != "TRAINED_CPU":
        raise ValueError("Expected a completed CPU training report")
    base = report["compile_only_base"]
    best = report["compile_only_best"]
    before = {row["id"]: row for row in base["cases"]}
    after = {row["id"]: row for row in best["cases"]}
    if len(before) != len(base["cases"]) or len(after) != len(best["cases"]):
        raise ValueError("Duplicate case IDs")
    if set(before) != set(after) or not before:
        raise ValueError("Base and best must evaluate the same nonempty case IDs")
    if base["evaluated"] != len(before) or best["evaluated"] != len(after):
        raise ValueError("Case counts do not match")
    cases = []
    for key in sorted(before):
        a, b = before[key], after[key]
        if a.get("family") != b.get("family"):
            raise ValueError("Task-family mismatch")
        x, y = a.get("compile"), b.get("compile")
        if x not in ("PASS", "FAIL") or y not in ("PASS", "FAIL"):
            raise ValueError("Unknown compile status")
        cases.append({
            "id": key, "family": a["family"],
            "base_compile": x, "best_compile": y,
            "transition": "REGRESSED" if x == "PASS" and y == "FAIL"
                          else "IMPROVED" if x == "FAIL" and y == "PASS"
                          else "UNCHANGED",
            "generation_changed": a.get("generated_sha256") != b.get("generated_sha256"),
        })
    baseline = sum(c["base_compile"] == "PASS" for c in cases)
    selected = sum(c["best_compile"] == "PASS" for c in cases)
    if baseline != base["compiled"] or selected != best["compiled"]:
        raise ValueError("Reported compilation totals disagree with case records")
    history = report["history"]
    if not history:
        raise ValueError("Missing validation history")
    best_record = min(history, key=lambda r: r["validation_response_loss"])
    if best_record["step"] != report["best_checkpoint_step"]:
        raise ValueError("Selected checkpoint step differs from validation history")
    return {
        "status": "REPORT_ONLY_DIAGNOSTIC",
        "model_context": report["model_context_length"],
        "case_count": len(cases),
        "base_compiled": baseline,
        "best_compiled": selected,
        "regressed": sum(c["transition"] == "REGRESSED" for c in cases),
        "improved": sum(c["transition"] == "IMPROVED" for c in cases),
        "generation_changed": sum(c["generation_changed"] for c in cases),
        "best_validation_step": best_record["step"],
        "best_validation_loss": best_record["validation_response_loss"],
        "last_validation_loss": history[-1]["validation_response_loss"],
        "last_test_loss": report.get("heldout_test_response_loss_last_sft"),
        "cases": cases,
        "root_cause": "NOT_ESTABLISHED",
        "missing_evidence": [
            "generated Java source for each case at each checkpoint",
            "javac stderr and return code for each generation",
            "generation stop reason, output token count, and truncation indicator",
            "independent Java tasks with no template-family overlap",
        ],
        "generated_code_execution": "NOT_RUN",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="v0.18 JSON result file")
    parser.add_argument("--output", type=Path, help="Optional output JSON (refuses overwrite)")
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text(encoding="utf-8")))
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
