"""Audit remaining archived experiments and historical exposure declarations offline."""
import argparse
from collections import Counter
import io
import json
from pathlib import Path
import zipfile

from scripts.audit_ranking_auxiliary import LIMIT, TOKENIZER, pin, parse, sha, require
from scripts.audit_replay_activation_scope import BASE, ORACLES, SUITE, hexsha, models
from scripts.audit_parameter_ce_exposure import REPORT_SHA as CE_SHA
from scripts.audit_historical_exposure import audit as archive_audit
from scripts.audit_v018_training import BASE as V017, LAST, DATA
from scripts.audit_java_lineage import ROOT

PRIOR_RUN = 46003548951
PRIOR_ARTIFACT = 15199855239
PRIOR_ZIP = "8f49329e762f9a34960f7748d7d18198054e761aad141686657068ec4a386b44"
INVENTORY_ARTIFACT = 15186012660
INVENTORY_ZIP = "3553b57b58eadfea8ecd36ebedeb8ce72ad8b65136fcfc4ec1a7a58ec1c56cad"
CANDIDATE = "379d57d6d8668995dc34fb310e0c5140041c5fd3e78b77932712785e96b01eae"
STDIO = "0cea8ded8a5c03d15b2b516a86665901266b0b375095579daa8d712661316a13"
BROAD = "181346d5395efdd2ef376730321e1585d453b2b09a822302f6f2d03137e1aa8d"
SPECS = (
    (45167198398, "contrastive-stdio-ab", "1f59f98acafc5b2939f5dc043717beb3b76dd5ff923506064377ca8a7221c468"),
    (45369765716, "m1-conditioning", "191a4699aeabff3a5bd36143aa3379170b0f51c8fcb0fe23aa68f7223e583e07"),
    (45369987553, "m1-conditioning-rerun", "3e19d9f621c8f46011cad1560d596ad4f2f6743124757b93c8783b8012111c92"),
    (45372372592, "m3-alignment-v1-blocked", "ed9e5b9fafa0b0c6c5c193ddb0ae852487346af1f145a680acc5896a5f5675d6"),
)
FAMILIES = ("stdin_min_offset", "stdin_count_odd_offset", "stdin_ascii_lower_suffix")
TASKS = [f"validation-{f}-{i:03d}" for f in FAMILIES for i in (2, 3)]
ARMS = ("control_interleaved", "contrastive_same_variant")
CATEGORIES = {"PASS", "OUTPUT_MISMATCH_ZERO_EXIT", "NONZERO_EXIT_OR_JAVAC_ERROR",
              "TIMEOUT", "OUTPUT_LIMIT", "REJECTED", "NOT_RUN", "RUNTIME_ERROR",
              "NOT_RUN_INVALID_GENERATION"}


def zip_json(path, digest, names):
    raw = pin(path, digest)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries = z.infolist()
        require(len(entries) == len(names) and {m.filename for m in entries} == set(names),
                "Unexpected ZIP members")
        out, hashes = {}, {}
        for m in entries:
            require(0 < m.file_size <= LIMIT, "Unsafe member size")
            with z.open(m) as f:
                body = f.read(LIMIT + 1)
            require(len(body) == m.file_size <= LIMIT, "Member size mismatch")
            out[m.filename], hashes[m.filename] = parse(body), sha(body)
    return out, hashes


def gate(row):
    require(row["actual_exposure_verified"] is False
            and row["training_lineage_verified"] is False
            and row["full_exposure_inventory"] == "NOT_ESTABLISHED"
            and row["inventory_ready_for_independent_evaluation"] is False
            and row["independent_evaluation"] == "NOT_RUN"
            and row["model_selection_permitted"] is False, "Evidence gate mismatch")


def prior(path, inventory_path):
    data, hashes = zip_json(path, PRIOR_ZIP, ["replay-activation-scope.json"])
    scope = data["replay-activation-scope.json"]
    require(scope["status"] == "ARCHIVED_REPLAY_BINDING_ACTIVATION_SCOPE_MATCH"
            and scope["replay_generation_rows_recounted"] == 84
            and scope["binding_pairs_compared"] == 161
            and scope["activation_pairs_compared"] == 133
            and scope["activation_control_checks_recounted"] == 3192
            and scope["new_inventory_entries"] == 0, "Previous scope mismatch")
    gate(scope)
    blobs, inv_hashes = zip_json(inventory_path, INVENTORY_ZIP,
        ["scoped-parameter-ce-inventory.json", "parameter-ce-evidence.json"])
    inventory = blobs["scoped-parameter-ce-inventory.json"]
    ev = blobs["parameter-ce-evidence.json"]
    gate(ev)
    require(ev["status"] == "HISTORICAL_PARAMETER_CE_DECLARATIONS_RECONSTRUCTED"
            and ev["report_sha256"] == CE_SHA
            and ev["inventory_sha256"] == scope["inventory_sha256"] ==
                inv_hashes["scoped-parameter-ce-inventory.json"]
            and len(inventory["files"]) == scope["inventory_entries"] == ev["inventory_entries"] == 15,
            "Inventory chain mismatch")
    return scope, inventory, hashes, inv_hashes


def load_reports(root):
    return [parse(pin(root / f"{run}-{slug}.json", digest, 8 * 1024 * 1024))
            for run, slug, digest in SPECS]


def behavior_counts(report):
    require([r["id"] for r in report["cases"]] == TASKS
            and report["tasks"] == 6 and report["total_cases"] == report["oci_invocations"] == 18
            and report["unavailable"] is False and report["source_persisted"] is False
            and report["java_executed_on_python_host"] is False, "Historical behavior scope mismatch")
    counts = Counter()
    rows = report["cases"]
    for row, family in zip(rows, [f for f in FAMILIES for _ in range(2)]):
        cats = row["case_categories"]
        require(row["family"] == family and row["generated"] == "GENERATED"
                and row["cases_total"] == row["cases_oci_invoked"] == len(cats) == 3
                and set(cats) <= CATEGORIES and row["cases_passed"] == cats.count("PASS")
                and row["task_pass"] is (cats.count("PASS") == 3)
                and row["unavailable"] is False and row["source_saved"] is False
                and hexsha(row["source_sha256"]), "Historical case mismatch")
        counts.update(cats)
    require(report["cases_passed"] == counts["PASS"]
            and report["tasks_all_pass"] == sum(r["task_pass"] for r in rows),
            "Historical total mismatch")
    return {"case_categories": dict(sorted(counts.items())), "cases_passed": counts["PASS"],
            "oci_invocations": report["oci_invocations"]}


def verify_early(ab, m1, rerun, blocked):
    require(ab["status"] == "MATCHED_AB_CONTRASTIVE_VALIDATION_COMPLETE"
            and ab["experiment"] == "JAVA_CONTRASTIVE_STDIO_MATCHED_AB_V1"
            and ab["origin_checkpoint_sha256"] == BASE and ab["tokenizer_sha256"] == TOKENIZER
            and ab["synthetic_stdio_manifest_sha256"] == STDIO
            and ab["contrastive_candidate_sha256"] == CANDIDATE
            and ab["seed"] == 4201 and ab["steps_per_arm"] == 160
            and ab["batch_size"] == 4 and ab["stdio_train_rows"] == 320
            and ab["validation_examples"] == 36 and ab["loss_selection_rows"] == 24
            and ab["holdback_excluded_from_loss_selection"] is True
            and ab["holdback_behavior_task_ids"] == TASKS
            and ab["arms"] == list(ARMS) and set(ab["arms_results"]) == set(ARMS)
            and ab["model_generated_java_executed"] == "OCI_ONLY_BEST_EFFORT"
            and ab["generated_java_executed_on_host"] is False
            and ab["original_test_split_accessed"] is False
            and ab["checkpoint_auto_promoted"] is False
            and ab["general_java_capability_established"] is False, "A/B scope mismatch")
    expected_schedule = {"same_training_examples_both_arms": True, "broad_rows_per_arm": 320,
        "stdio_rows_per_arm": 320, "unique_training_exposures": True,
        "same_original_broad_row_positions": True,
        "sole_intervention": "STDIO_TRAIN_ORDER_AND_SAME_INDEX_ALGORITHM_PAIRING",
        "validation_or_test_in_optimizer": False}
    require(ab["schedule"] == expected_schedule, "A/B schedule mismatch")
    by_model = {"base_v018": ab["baseline_behavior"]}
    for arm in ARMS:
        row = ab["arms_results"][arm]
        training = row["training"]
        require(training["arm"] == arm and training["steps_completed"] == training["selected_step"] == 160
                and [h["step"] for h in training["history"]] == [0, 80, 160]
                and training["test_split_used"] is False, "A/B training mismatch")
        by_model[arm] = row["behavior"]
    counts = {name: behavior_counts(b) for name, b in by_model.items()}
    require(all(b["cases_passed"] == 0 for b in counts.values())
            and ab["gate"]["decision"] == "NO_VALIDATED_CONTRASTIVE_GAIN"
            and ab["gate"]["improvement_demonstrated"] is False
            and ab["gate"]["production_adoption"] is False, "A/B conclusion mismatch")
    for m in (m1, rerun):
        require(m["status"] == "INFERENCE_DIAGNOSTIC_COMPLETE"
                and m["historical_report_sha256"] == SPECS[0][2]
                and m["historical_run_id"] == SPECS[0][0]
                and m["tokenizer_sha256"] == TOKENIZER and m["fixed_task_ids"] == TASKS
                and m["control_checkpoint"] == "NOT_AVAILABLE_IN_PRIOR_ARTIFACT"
                and m["historical_behavior_only"] == counts
                and m["generation_cap"] == 170 and m["model_training"] == "NOT_RUN"
                and m["new_java_compilation"] == m["new_java_execution"] == "NOT_RUN"
                and m["test_split_accessed"] is False and m["production_promotion"] is False
                and m["root_cause"] == m["independent_benchmark"] == "NOT_ESTABLISHED",
                "M1 scope mismatch")
        require([r["id"] for r in m["prompt_metrics"]] == TASKS
                and len({r["prefix_sha256"] for r in m["prompt_metrics"]}) == 6
                and all(hexsha(r["prefix_sha256"]) and type(r["prefix_tokens"]) is int
                        and 0 < r["prefix_tokens"] < 384 and type(r["reference_tokens"]) is int
                        and r["reference_tokens"] > 0 for r in m["prompt_metrics"]),
                "M1 prompt declaration mismatch")
        require(set(m["models"]) == {"base_v018", "contrastive_same_variant"}, "M1 model mismatch")
        for name, cp in (("base_v018", BASE), ("contrastive_same_variant", CANDIDATE)):
            row = m["models"][name]
            observations = row["observations"]
            expected = [x["source_sha256"] for x in by_model[name]["cases"]]
            require(row["checkpoint_sha256"] == cp and row["historical_hash_matches"] == 6
                    and [r["id"] for r in observations] == TASKS
                    and [r["source_sha256"] for r in observations] == expected
                    and all(r["historical_source_sha_matches"] is True and r["stop"] == "EOS"
                            and r["generation_budget"] == 170
                            and r["swapped_prompt_id"] in TASKS and r["swapped_prompt_id"] != r["id"]
                            for r in observations)
                    and row["unique_source_hashes"] == len(set(expected)), "M1 replay declaration mismatch")
    require(m1["prompt_metrics"] == rerun["prompt_metrics"], "M1 rerun prompt drift")
    a = blocked["alignment"]
    require(blocked["status"] == "BLOCKED_ALIGNMENT_ADMISSION"
            and blocked["base_sha256"] == BASE and blocked["tokenizer_sha256"] == TOKENIZER
            and blocked["broad_train_sha256"] == BROAD and blocked["stdio_manifest_sha256"] == STDIO
            and blocked["holdback_sha256"] == SUITE and blocked["optimizer_steps_completed"] == 0
            and blocked["new_inference"] == blocked["new_java_compile"] == blocked["new_java_behavior"] == "NOT_RUN"
            and blocked["production_promotion"] is False
            and blocked["independent_benchmark"] == "NOT_ESTABLISHED"
            and len(a["rows_per_pair_type"]) == 5
            and sum(a["rows_per_pair_type"].values()) == a["admitted_rows"] == 194
            and a["excluded_rows"] == 126 and a["aligned_token_positions"] == 282
            and a["minimum_admitted_rows"] == 160 and a["passed"] is False
            and a["all_five_pair_types_represented"] is False
            and min(a["rows_per_pair_type"].values()) == 0 and hexsha(a["mask_sha256"]),
            "Blocked admission scope mismatch")
    return {"historical_behavior_only": counts, "historical_behavior_cases_recounted": 54,
            "m1_prompt_declarations_compared": 12, "m1_generation_hash_declarations_compared": 24,
            "m1_reports_identical": m1 == rerun, "blocked_alignment": a,
            "blocked_run_produced_new_checkpoint": False,
            "historical_behavior_reexecuted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("experiments/archive"))
    p.add_argument("--prior-zip", type=Path, required=True)
    p.add_argument("--inventory-zip", type=Path, required=True)
    args = p.parse_args()
    scope, inventory, hashes, inv_hashes = prior(args.prior_zip, args.inventory_zip)
    early = verify_early(*load_reports(args.archive / "reports"))
    ledger = archive_audit(args.archive)
    require(ledger["report_count"] == 12, "Archive report coverage changed")
    from scripts.audit_integrated_exposure_scope import integrate
    out = integrate(args.archive, scope, inventory, ledger, early)
    out["prior_evidence"] = {"run_id": PRIOR_RUN, "artifact_id": PRIOR_ARTIFACT,
                             "archive_sha256": PRIOR_ZIP, "member_sha256": hashes}
    out["inventory_evidence"] = {"artifact_id": INVENTORY_ARTIFACT,
                                  "archive_sha256": INVENTORY_ZIP, "member_sha256": inv_hashes}
    print(json.dumps(out, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
