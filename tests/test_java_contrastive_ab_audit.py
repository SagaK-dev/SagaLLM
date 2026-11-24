"""Offline A/B evidence auditor: tamper, leakage and provenance tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

from scripts import audit_java_contrastive_ab as audit


def make_behavior(unique: int) -> dict:
    hashes = [hashlib.sha256(f"source-{i % unique}".encode()).hexdigest() for i in range(6)]
    tasks = []
    for i, family in enumerate(audit.FAMILIES):
        for variant in (2, 3):
            tasks.append({
                "id": f"validation-{family}-{variant:03}", "family": family,
                "generated": "GENERATED", "cases_total": 3, "cases_oci_invoked": 3,
                "cases_passed": 0,
                "case_categories": ["OUTPUT_MISMATCH_ZERO_EXIT"] * 3,
                "task_pass": False, "unavailable": False, "source_saved": False,
                "source_sha256": hashes[i * 2 + variant - 2],
            })
    return {
        "tasks": 6, "total_cases": 18, "oci_invocations": 18,
        "cases_passed": 0, "tasks_all_pass": 0, "unavailable": False,
        "source_persisted": False, "java_executed_on_python_host": False,
        "cases": tasks,
    }


def make_report() -> dict:
    base, control, treatment = (make_behavior(n) for n in (3, 2, 1))
    def training(name: str) -> dict:
        return {"arm": name, "steps_completed": 160, "selected_step": 160,
                "test_split_used": False, "source_programs_executed_on_host": False,
                "history": [{"step": step, "validation_loss": 1.0}
                            for step in (0, 80, 160)]}
    behavior = {"base_v018": base, audit.ARMS[0]: control, audit.ARMS[1]: treatment}
    metrics = {}
    for name, val in behavior.items():
        m = audit.validate_behavior(val)
        metrics[name] = {key: m[key] for key in (
            "unique_generated_sources", "six_distinct_prompts",
            "cross_family_identical_source_pairs", "within_family_identical_source_pairs",
            "source_validity_measured", "behavioral_semantics_proven")}
    return {
        "experiment": audit.EXPERIMENT,
        "status": "MATCHED_AB_CONTRASTIVE_VALIDATION_COMPLETE",
        "origin_checkpoint_sha256": audit.BASE_SHA256,
        "tokenizer_sha256": audit.TOKENIZER_SHA256,
        "synthetic_stdio_manifest_sha256": "a" * 64,
        "steps_per_arm": 160, "batch_size": 4, "seed": 4201,
        "learning_rate": 2e-5, "stdio_train_rows": 320,
        "validation_examples": 36, "loss_selection_rows": 24,
        "behavior_cases_per_model": 18, "arms": list(audit.ARMS),
        "holdback_behavior_task_ids": [f"validation-{f}-{i:03}" for f in audit.FAMILIES for i in (2, 3)],
        "holdback_excluded_from_loss_selection": True,
        "schedule": {
            "same_training_examples_both_arms": True,
            "unique_training_exposures": True,
            "same_original_broad_row_positions": True,
            "validation_or_test_in_optimizer": False,
            "broad_rows_per_arm": 320, "stdio_rows_per_arm": 320,
            "sole_intervention": "STDIO_TRAIN_ORDER_AND_SAME_INDEX_ALGORITHM_PAIRING",
        },
        "generated_java_executed_on_host": False,
        "raw_generated_source_or_stdout_saved": False,
        "original_test_split_accessed": False,
        "public_exercism_data_accessed": False,
        "synthetic_validation_is_independent_benchmark": False,
        "checkpoint_auto_promoted": False,
        "general_java_capability_established": False,
        "test_split_untouched": True,
        "production_model_unchanged": True,
        "oci_smoke": "PASS", "oci_image_id": "sha256:" + "a" * 64,
        "model_generated_java_executed": "OCI_ONLY_BEST_EFFORT",
        "baseline_behavior": base,
        "arms_results": {
            audit.ARMS[0]: {"training": training(audit.ARMS[0]), "behavior": control},
            audit.ARMS[1]: {"training": training(audit.ARMS[1]), "behavior": treatment},
        },
        "prompt_conditioning_diagnostics": metrics,
        "gate": {"decision": "NO_VALIDATED_CONTRASTIVE_GAIN",
                 "improvement_demonstrated": False,
                 "contrastive_minus_control_correct_cases": 0,
                 "contrastive_minus_base_correct_cases": 0,
                 "production_adoption": False,
                 "public_benchmark_used_for_selection": False},
    }


def test_well_formed_report_is_not_model_proof():
    result = audit.audit_report(make_report())
    assert result["status"] == "OFFLINE_REPORT_INVARIANTS_PASS"
    assert result["gate"] == "NO_VALIDATED_CONTRASTIVE_GAIN"
    assert result["cases_passed"] == dict.fromkeys(("base_v018", *audit.ARMS), 0)
    assert result["unique_source_hashes"]["contrastive_same_variant"] == 1
    assert result["independent_benchmark"] == "NOT_ESTABLISHED"
    assert result["github_run_and_artifact_authentication"] == "NOT_VERIFIED_OFFLINE"


@pytest.mark.parametrize("field,malicious", [
    ("original_test_split_accessed", True),
    ("checkpoint_auto_promoted", True),
    ("holdback_excluded_from_loss_selection", False),
    ("tokenizer_sha256", "b" * 64),
    ("steps_per_arm", 159),
])
def test_protocol_change_fails_closed(field, malicious):
    r = make_report()
    r[field] = malicious
    with pytest.raises(ValueError):
        audit.audit_report(r)


def test_forged_aggregate_or_collision_fails_closed():
    r = make_report()
    r["baseline_behavior"]["cases_passed"] = 1
    with pytest.raises(ValueError, match="cases_passed"):
        audit.audit_report(r)
    r = make_report()
    r["prompt_conditioning_diagnostics"][audit.ARMS[1]]["unique_generated_sources"] = 6
    with pytest.raises(ValueError, match="unique_generated_sources"):
        audit.audit_report(r)


def test_forged_selection_gate_fails_closed():
    r = make_report()
    r["gate"]["improvement_demonstrated"] = True
    r["gate"]["decision"] = "CONTRASTIVE_VALIDATION_SIGNAL_NOT_DEPLOYED"
    with pytest.raises(ValueError, match="Selection gate"):
        audit.audit_report(r)


def test_disallowed_raw_java_or_stderr_fails_closed():
    for field, val in (("source_text", "public class Main {}"),
                       ("raw_stderr", "javac.java:1: expected"),
                       ("model_answer", "public class Main {}")):
        r = make_report()
        r["secret_debug"] = {field: val}
        with pytest.raises(ValueError):
            audit.audit_report(r)


def test_unavailable_never_counts_as_validated_gain():
    r = make_report()
    r["baseline_behavior"]["cases"][0]["case_categories"][0] = "NOT_RUN"
    r["baseline_behavior"]["cases"][0]["unavailable"] = True
    r["baseline_behavior"]["unavailable"] = True
    r["gate"] = {"decision": "INCONCLUSIVE_OCI_UNAVAILABLE",
                 "improvement_demonstrated": False, "production_adoption": False}
    result = audit.audit_report(r)
    assert result["gate"] == "INCONCLUSIVE_OCI_UNAVAILABLE"


def test_cli_rejects_bad_report_checksum_and_writes_no_evidence(tmp_path):
    report = tmp_path / "ab-report.json"
    report.write_text(json.dumps(make_report()), encoding="utf8")
    out = tmp_path / "audit.json"
    kwargs = ["--report", str(report), "--expected-report-sha256", "b" * 64,
              "--run-id", "45167198398", "--run-sha", "40dcce6fe2e4bd2ab83ea70cf5a9cd12368c0312",
              "--artifact-id", "14815096856", "--artifact-digest", "sha256:" + "a" * 64,
              "--output", str(out)]
    with pytest.raises(ValueError, match="Report SHA-256 mismatch"):
        audit.main(kwargs)
    assert not out.exists()
    kwargs[3] = hashlib.sha256(report.read_bytes()).hexdigest()
    assert audit.main(kwargs) == 0
    summary = json.loads(out.read_text(encoding="utf8"))
    assert summary["source"]["run_id_operator_supplied"] == 45167198398
    assert summary["github_run_and_artifact_authentication"] == "NOT_VERIFIED_OFFLINE"
    with pytest.raises(SystemExit):
        audit.main(kwargs)


def test_symlink_and_oversized_payload_rejected(tmp_path):
    data = tmp_path / "large.json"
    data.write_bytes(b"x" * (audit.MAX_REPORT_BYTES + 1))
    with pytest.raises(ValueError, match="size"):
        audit.read_bounded_json(data)
    link = tmp_path / "link.json"
    link.symlink_to(data)
    with pytest.raises(ValueError, match="non-symlink"):
        audit.read_bounded_json(link)
