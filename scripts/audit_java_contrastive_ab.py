"""Fail-closed, offline audit of a SHA-pinned SagaLLM contrastive A/B report.

This script validates *reported* synthetic outcomes. It does not fetch GitHub
metadata, authenticate an Actions run, execute Java, or establish model quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

EXPERIMENT = "JAVA_CONTRASTIVE_STDIO_MATCHED_AB_V1"
BASE_SHA256 = "9802e848f0abea9979623f461b5e63cd7aed838acd087d06b20d747db1e2a0fc"
TOKENIZER_SHA256 = "186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe"
FAMILIES = ("stdin_min_offset", "stdin_count_odd_offset", "stdin_ascii_lower_suffix")
ARMS = ("control_interleaved", "contrastive_same_variant")
CATEGORIES = {"PASS", "OUTPUT_MISMATCH_ZERO_EXIT", "NONZERO_EXIT_OR_JAVAC_ERROR",
              "TIMEOUT", "OUTPUT_LIMIT", "REJECTED", "NOT_RUN", "RUNTIME_ERROR",
              "NOT_RUN_INVALID_GENERATION"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
SHA1 = re.compile(r"[0-9a-f]{40}\Z")
MAX_REPORT_BYTES = 2_000_000


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def is_sha(value: object) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def is_count(value: object, maximum: int) -> bool:
    return type(value) is int and 0 <= value <= maximum


def read_bounded_json(path: Path) -> tuple[dict, str]:
    require(path.is_file() and not path.is_symlink(), "Report must be a regular, non-symlink file")
    require(0 < path.stat().st_size <= MAX_REPORT_BYTES, "Report file size is invalid")
    raw = path.read_bytes()
    require(len(raw) <= MAX_REPORT_BYTES, "Report exceeds byte limit")
    document = json.loads(raw.decode("utf-8"))
    require(isinstance(document, dict), "Report root must be an object")
    return document, hashlib.sha256(raw).hexdigest()


def check_no_raw_payload(value: object) -> None:
    """Reject extra output/source/diagnostic payload keys, even when nested."""
    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, item in node.items():
                require(isinstance(key, str), "Non-string JSON key")
                norm = key.lower()
                # Flags that indicate no source was saved are safe.
                if any(word in norm for word in ("raw_stdout", "raw_stderr", "source_text",
                                                  "generated_code", "java_source", "stdin_text")):
                    require(norm == "raw_generated_source_or_stdout_saved" and item is False,
                            "Raw source, stdin or diagnostics field forbidden")
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            require("public class Main" not in node and "System.out.print" not in node,
                    "Report contains possible Java source")
    walk(value)


def validate_behavior(value: dict) -> dict:
    require(isinstance(value, dict), "Behavior section missing")
    require(value.get("tasks") == 6 and value.get("total_cases") == 18,
            "Incorrect behavior coverage")
    cases = value.get("cases")
    require(isinstance(cases, list) and len(cases) == 6,
            "Expected exactly six task observations")
    expected_ids = {f"validation-{f}-{i:03}" for f in FAMILIES for i in (2, 3)}
    ids = set()
    counts = {"cases_passed": 0, "tasks_all_pass": 0, "oci_invocations": 0}
    missing = False
    hashes = []
    for task in cases:
        require(isinstance(task, dict), "Invalid task entry")
        identifier, family = task.get("id"), task.get("family")
        require(identifier in expected_ids and identifier not in ids and
                identifier.startswith(f"validation-{family}-"),
                "Unexpected/duplicate task or family")
        ids.add(identifier)
        categories = task.get("case_categories")
        require(isinstance(categories, list) and len(categories) == 3 and
                all(isinstance(x, str) and x in CATEGORIES for x in categories),
                "Invalid case categories")
        require(task.get("cases_total") == 3, "Task coverage mismatch")
        invocations = task.get("cases_oci_invoked")
        passed = task.get("cases_passed")
        require(invocations in (0, 3) and is_count(passed, 3),
                "Invalid invocation / passed count")
        require(passed == categories.count("PASS") and
                (invocations == 0) == all(x == "NOT_RUN_INVALID_GENERATION" for x in categories),
                "Categories contradict observations")
        task_unavailable = any(x in ("NOT_RUN", "RUNTIME_ERROR") for x in categories)
        require(task.get("unavailable") is task_unavailable and
                task.get("task_pass") is (passed == 3 and not task_unavailable) and
                task.get("source_saved") is False,
                "Task evidence flags inconsistent")
        source_sha = task.get("source_sha256")
        if invocations:
            require(is_sha(source_sha), "Generated task missing valid SHA-256")
            require(task.get("generated") == "GENERATED", "Invalid generation status")
        else:
            require(source_sha is None and task.get("generated") in
                    ("INVALID_GENERATED_JAVA", "INSUFFICIENT_CONTEXT"),
                    "Invalid generation has a source hash")
        hashes.append((family, source_sha))
        missing = missing or task_unavailable
        counts["cases_passed"] += passed
        counts["tasks_all_pass"] += int(task.get("task_pass"))
        counts["oci_invocations"] += invocations
    require(ids == expected_ids, "Behavior holdback mismatch")
    for key, total in counts.items():
        require(value.get(key) == total, f"Reported {key} does not match cases")
    require(value.get("unavailable") is missing and
            value.get("source_persisted") is False and
            value.get("java_executed_on_python_host") is False,
            "Behavior safety flags are invalid")
    source_values = [sha for _, sha in hashes if sha is not None]
    within = cross = 0
    for i, (family, sha) in enumerate(hashes):
        if sha is None:
            continue
        for other_family, other_sha in hashes[i+1:]:
            if sha == other_sha:
                if family == other_family:
                    within += 1
                else:
                    cross += 1
    return {"unique_generated_sources": len(set(source_values)),
            "six_distinct_prompts": 6, "cross_family_identical_source_pairs": cross,
            "within_family_identical_source_pairs": within,
            "source_validity_measured": "SHA256_WITHOUT_EXPOSING_CODE",
            "behavioral_semantics_proven": False,
            "cases_passed": counts["cases_passed"],
            "tasks_all_pass": counts["tasks_all_pass"], "unavailable": missing,
            "task_ids": sorted(ids)}


def audit_report(report: dict) -> dict:
    require(isinstance(report, dict), "Missing report")
    check_no_raw_payload(report)
    require(report.get("experiment") == EXPERIMENT and
            report.get("status") == "MATCHED_AB_CONTRASTIVE_VALIDATION_COMPLETE",
            "Wrong experiment or incomplete A/B")
    require(report.get("origin_checkpoint_sha256") == BASE_SHA256 and
            report.get("tokenizer_sha256") == TOKENIZER_SHA256 and
            is_sha(report.get("synthetic_stdio_manifest_sha256")),
            "Unpinned experiment inputs")
    require(report.get("steps_per_arm") == 160 and
            report.get("batch_size") == 4 and
            report.get("seed") == 4201 and
            report.get("learning_rate") == 2e-5,
            "A/B protocol changed")
    require(report.get("stdio_train_rows") == 320 and
            report.get("validation_examples") == 36 and
            report.get("loss_selection_rows") == 24 and
            report.get("behavior_cases_per_model") == 18 and
            report.get("arms") == list(ARMS), "A/B data shape changed")
    expected_ids = [f"validation-{f}-{i:03}" for f in FAMILIES for i in (2, 3)]
    require(report.get("holdback_behavior_task_ids") == expected_ids and
            report.get("holdback_excluded_from_loss_selection") is True,
            "Heldback isolation not recorded")
    schedule = report.get("schedule")
    require(isinstance(schedule, dict) and
            all(schedule.get(k) is True for k in (
                "same_training_examples_both_arms", "unique_training_exposures",
                "same_original_broad_row_positions")) and
            schedule.get("validation_or_test_in_optimizer") is False and
            schedule.get("broad_rows_per_arm") == 320 and
            schedule.get("stdio_rows_per_arm") == 320 and
            schedule.get("sole_intervention") == "STDIO_TRAIN_ORDER_AND_SAME_INDEX_ALGORITHM_PAIRING",
            "Training exposure matching not established")
    require(all(report.get(k) is False for k in (
        "generated_java_executed_on_host", "raw_generated_source_or_stdout_saved",
        "original_test_split_accessed", "public_exercism_data_accessed",
        "synthetic_validation_is_independent_benchmark", "checkpoint_auto_promoted",
        "general_java_capability_established")) and
        all(report.get(k) is True for k in (
            "test_split_untouched", "production_model_unchanged")) and
        report.get("oci_smoke") == "PASS" and
        report.get("model_generated_java_executed") == "OCI_ONLY_BEST_EFFORT" and
        isinstance(report.get("oci_image_id"), str) and
        re.fullmatch(r"sha256:[0-9a-f]{64}", report["oci_image_id"]) is not None,
        "Isolation / nonpromotion evidence incomplete")
    results = {"base_v018": report.get("baseline_behavior")}
    arm_reports = report.get("arms_results")
    require(isinstance(arm_reports, dict) and set(arm_reports) == set(ARMS),
            "Missing A/B arms")
    for arm in ARMS:
        train = arm_reports[arm].get("training")
        require(isinstance(train, dict) and train.get("arm") == arm and
                train.get("steps_completed") == 160 and
                train.get("selected_step") in (0, 80, 160) and
                train.get("test_split_used") is False and
                train.get("source_programs_executed_on_host") is False,
                "Training arm incomplete or untrusted")
        history = train.get("history")
        require(isinstance(history, list) and
                [row.get("step") for row in history] == [0, 80, 160],
                "Training validation checkpoints missing")
        results[arm] = arm_reports[arm].get("behavior")
    computed = {key: validate_behavior(value) for key, value in results.items()}
    reported = report.get("prompt_conditioning_diagnostics")
    require(isinstance(reported, dict) and set(reported) == set(computed),
            "Missing prompt-conditioning diagnostics")
    for key in computed:
        for metric in ("unique_generated_sources", "six_distinct_prompts",
                       "cross_family_identical_source_pairs",
                       "within_family_identical_source_pairs",
                       "source_validity_measured", "behavioral_semantics_proven"):
            require(reported[key].get(metric) == computed[key][metric],
                    f"Incorrect {metric}: {key}")
    base, control, treatment = (computed[k] for k in ("base_v018", *ARMS))
    gain = (treatment["cases_passed"] >= control["cases_passed"] + 2 and
            treatment["cases_passed"] >= base["cases_passed"] + 2 and
            treatment["tasks_all_pass"] >= control["tasks_all_pass"] and
            treatment["unique_generated_sources"] > control["unique_generated_sources"])
    unavailable = any(x["unavailable"] for x in (base, control, treatment))
    expected_gate = ("INCONCLUSIVE_OCI_UNAVAILABLE" if unavailable else
                     "CONTRASTIVE_VALIDATION_SIGNAL_NOT_DEPLOYED" if gain else
                     "NO_VALIDATED_CONTRASTIVE_GAIN")
    gate = report.get("gate")
    require(isinstance(gate, dict) and gate.get("decision") == expected_gate and
            gate.get("improvement_demonstrated") is (gain and not unavailable) and
            gate.get("production_adoption") is False,
            "Selection gate contradicts observations")
    if not unavailable:
        require(gate.get("contrastive_minus_control_correct_cases") ==
                treatment["cases_passed"] - control["cases_passed"] and
                gate.get("contrastive_minus_base_correct_cases") ==
                treatment["cases_passed"] - base["cases_passed"] and
                gate.get("public_benchmark_used_for_selection") is False,
                "Selection gate deltas inconsistent")
    return {"status": "OFFLINE_REPORT_INVARIANTS_PASS",
            "gate": expected_gate,
            "cases_passed": {key: value["cases_passed"] for key, value in computed.items()},
            "unique_source_hashes": {key: value["unique_generated_sources"] for key, value in computed.items()},
            "synthetic_behavior_only": True,
            "independent_benchmark": "NOT_ESTABLISHED",
            "production_promotion": "NOT_AUTHORIZED",
            "github_run_and_artifact_authentication": "NOT_VERIFIED_OFFLINE"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-sha", required=True)
    parser.add_argument("--artifact-id", type=int, required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not is_sha(args.expected_report_sha256):
        parser.error("--expected-report-sha256 must be lowercase SHA-256")
    if not SHA1.fullmatch(args.run_sha) or args.run_id < 1 or args.artifact_id < 1:
        parser.error("Expected real commit SHA, run ID and artifact ID")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", args.artifact_digest):
        parser.error("Expected artifact archive digest")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Refuse to overwrite existing audit evidence")
    document, digest = read_bounded_json(args.report)
    require(digest == args.expected_report_sha256, "Report SHA-256 mismatch")
    summary = audit_report(document)
    summary["source"] = {
        "report_sha256": digest, "run_id_operator_supplied": args.run_id,
        "run_sha_operator_supplied": args.run_sha,
        "artifact_id_operator_supplied": args.artifact_id,
        "artifact_archive_digest_operator_supplied": args.artifact_digest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": summary["status"], "gate": summary["gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
