"""Audit three legacy report originals and discover unresolved hosted runs.

No source extraction from model artifacts, model loading, inference, Java or training.
The Actions inventory represents accessible latest-attempt metadata, not exposure proof.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from scripts.archive_experiment_reports import load_manifest
from scripts.audit_ranking_auxiliary import pin, parse, sha, require
from scripts.audit_replay_activation_scope import BASE, hexsha
from scripts.audit_v018_training import BASE as V017, TOKENIZER

LEGACY_REVISION = "fe31880f6cb9bde71e01b22942964cb6df327186"
LEGACY_MANIFEST_BLOB = "75926c7a6cc93a6671e95de50f46ff7aeeaf86ee"
SNAPSHOT_BLOB = "3b7aebc48c782eb5523d621dcdf8b565684d322e"
SPECS = {
    "v018-paired-compile": (45007278383, 14748252897, "5fc921385b9097f053007fdbcdb9bc3b39a7461acbdc1dcf2d19afbd6312b811"),
    "stdio-semantic-sft": (45164032104, 14813711672, "bb3994dfcb6b13f5f94687d281a9a1d615117eecb95647b22cf0021ac79700f7"),
    "public-unreviewed-behavior": (45161749481, 14810574335, "a0b1e779869c332a1c7b343ec870823800da7b9106f341e28d85b3a6529c0653"),
}
STDIO = "0cea8ded8a5c03d15b2b516a86665901266b0b375095579daa8d712661316a13"
SEMANTIC_CANDIDATE = "8d955d6c61fd677bda80934811f62ef5547090d610f41d4f06449a4a7d4ea3e2"
PUBLIC_SUITE = "d1af64a934c304f04b2cfa822280c73cb7da7b64d09e83f62b071b945a244c74"
ANCESTOR_RUNS = {44981937649, 44982940959}
FAMILIES = ("stdin_min_offset", "stdin_count_odd_offset", "stdin_ascii_lower_suffix")
TASKS = [f"validation-{f}-{i:03d}" for f in FAMILIES for i in (0, 1)]

RESEARCH = {
    "java-public-cpu-train", "java-instruction-sft-cpu", "java-context-sft-cpu",
    "java-paired-compile-diagnostics", "java-prompt-validation-experiment",
    "java-identifier-curriculum", "java-identifier-ab-training", "java-broad-mixture-ab",
    "java-public-compile-diagnostic", "java-public-isolated-behavior",
    "java-stdio-semantic-sft", "java-stdio-deterministic", "java-contrastive-stdio-ab",
    "java-m1-conditioning", "java-m3-ranking", "java-m3-frozen-replay",
    "java-m3-parameter-binding", "java-m3-parameter-ce", "java-m3-prompt-signal",
    "java-m3-activation-swap", "java-m3-prompt-bias",
}
INFRASTRUCTURE = {
    "cpu-ci", "archive-experiment-reports", "java-matched-reports", "java-private-intake",
    "java-independent-io-protocol", "java-external-io-intake", "archive-legacy-2026-11-25",
    "audit-historical-exposure", "audit-checkpoint-exposure", "audit-development-data",
    "audit-ancestor-inventory", "audit-development-inventory", "audit-ranking-auxiliary",
    "audit-parameter-ce-exposure", "audit-prompt-probe-scope", "audit-replay-activation-scope",
    "audit-remaining-historical-scope", "audit-legacy-run-discovery",
}
GIT_SHA = re.compile(r"[0-9a-f]{40}\Z")


def read_bounded(path, limit=8 * 1024 * 1024):
    require(path.is_file() and not path.is_symlink(), "Unsafe input")
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    require(0 < len(raw) <= limit, "Input size limit")
    return raw


def blob_pin(path, expected):
    raw = read_bounded(path)
    actual = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    require(actual == expected, "Historical Git blob mismatch")
    return raw


def legacy_reports(root):
    manifest = root / "legacy-2026-11-25.json"
    blob_pin(manifest, LEGACY_MANIFEST_BLOB)
    rows = load_manifest(manifest)
    require(len(rows) == 3 and {r["slug"] for r in rows} == set(SPECS), "Legacy manifest scope")
    result = {}
    for r in rows:
        run, artifact, digest = SPECS[r["slug"]]
        require((r["run_id"], r["artifact_id"], r["report_sha256"]) == (run, artifact, digest),
                "Legacy manifest pin mismatch")
        result[r["slug"]] = parse(pin(root / "reports" / f"{run}-{r['slug']}.json", digest, 8 * 1024 * 1024))
    return rows, result


def compile_scope(d):
    require(d["status"] == "PAIRED_COMPILE_ONLY" and d["baseline_sha256"] == V017
            and d["best_sha256"] == BASE and d["tokenizer_sha256"] == TOKENIZER
            and d["java_executed"] is False and d["root_cause"] == "NOT_ESTABLISHED", "Compile scope")
    rows_by_model = {}
    for name, count in (("base", 2), ("best", 0)):
        m = d[name]
        rows = m["cases"]
        require(m["status"] == "SCORED_COMPILE_ONLY" and m["behavior"] == "NOT_RUN"
                and m["evaluated"] == len(rows) == 8 and len({r["id"] for r in rows}) == 8,
                "Compile row coverage")
        for r in rows:
            require(r["compile"] in {"PASS", "FAIL"} and r["compile_reason"] == "JAVAC"
                    and r["generation_eos_seen"] is True and r["generation_hit_token_cap"] is False
                    and r["generation_stop_reason"] == "EOS" and r["source_persisted"] is False
                    and r["java_behavior"] == "NOT_RUN" and hexsha(r["generated_sha256"])
                    and r["exact"] is False, "Compile row scope")
            if r["compile"] == "PASS":
                require(r["javac_diagnostic_category"] == "NONE" and r["javac_stderr_sha256"] is None,
                        "Unexpected success category")
            else:
                require(r["javac_diagnostic_category"] == "JAVAC_DIAGNOSTIC"
                        and hexsha(r["javac_stderr_sha256"]), "Unexpected failure category")
        require(m["compiled"] == sum(r["compile"] == "PASS" for r in rows) == count
                and m["exact_reference_matches"] == 0, "Compile count mismatch")
        rows_by_model[name] = rows
    before, after = rows_by_model["base"], rows_by_model["best"]
    require([(r["id"], r["family"]) for r in before] == [(r["id"], r["family"]) for r in after],
            "Unmatched compile cases")
    return {"matched_cases": 8, "generation_records_recounted": 16,
            "baseline_compiled": 2, "best_compiled": 0,
            "regressed_ids": [a["id"] for a,b in zip(before,after)
                              if a["compile"] == "PASS" and b["compile"] == "FAIL"],
            "java_behavior": "NOT_RUN", "root_cause": "NOT_ESTABLISHED"}


def public_scope(d):
    require(d["status"] == "EXPLORATORY_PUBLIC_UNREVIEWED_ISOLATED_IO"
            and d["suite_sha256"] == PUBLIC_SUITE and d["original_six_tasks"] == 6
            and d["original_27_cases"] == 27 and d["matched_previous_compiles"] == 4
            and d["v017_checkpoint_sha256"] == V017 and d["v018_checkpoint_sha256"] == BASE
            and d["tokenizer_sha256"] == TOKENIZER and d["human_review_status"] == "REVIEW_REQUIRED"
            and d["public_pretraining_contamination_possible"] is True
            and d["independent_benchmark_approval"] is False and d["model_selection_permitted"] is False
            and d["model_updated"] is False and d["java_execution"] == "OCI_ONLY"
            and d["generated_source_persisted"] is False and d["raw_stdout_or_stderr_persisted"] is False
            and d["isolation"]["security_proof"] is False, "Public scope mismatch")
    result = {}
    for model, count in (("v017", 10), ("v018", 9)):
        m = d[model]
        require(m["status"] == "EVALUATED" and m["compiled_outputs_regenerated"] == len(m["tasks"]) == 2
                and len({r["id"] for r in m["tasks"]}) == 2, "Public model scope")
        total = 0
        for r in m["tasks"]:
            require(r["cases_total"] == len(r["case_statuses"]) == len(r["case_failure_categories"])
                    and r["cases_total"] > 0 and r["cases_passed"] == 0
                    and set(r["case_statuses"]) == {"FAIL"}
                    and set(r["case_failure_categories"]) == {"OUTPUT_MISMATCH_ZERO_EXIT"}
                    and r["task_status"] == "FAIL" and r["generated_source_stored"] is False
                    and r["host_java_execution"] is False and hexsha(r["source_sha256"]), "Public case scope")
            total += r["cases_total"]
        require(total == m["io_cases_attempted"] == m["io_cases_failing"] == count
                and m["io_cases_passed"] == m["tasks_passed_all_cases"] == 0, "Public counts")
        result[model] = {"task_ids": [r["id"] for r in m["tasks"]], "attempted_cases": total, "passed_cases": 0}
    require(sum(r["attempted_cases"] for r in result.values()) == d["attempted_cases_across_models"] == 19,
            "Public total count")
    return {"models": result, "suite_sha256_declared": PUBLIC_SUITE,
            "historical_cases_recounted": 19, "matched_full_suite": False,
            "source_bytes_verified": False, "independent_evaluation": "NOT_RUN", "rights_review": "REVIEW_REQUIRED"}


def stdio_scope(d):
    require(d["status"] == "VALIDATION_STDIO_BEHAVIOR_EVALUATED"
            and d["experiment"] == "JAVA_STDIO_SEMANTIC_SFT_V1"
            and d["origin_checkpoint_sha256"] == BASE and d["tokenizer_sha256"] == TOKENIZER
            and d["stdio_manifest_sha256"] == STDIO
            and d["candidate_checkpoint_sha256"] == SEMANTIC_CANDIDATE
            and d["stdio_train_accepted"] == 320 and d["stdio_validation_accepted"] == 36
            and d["steps"] == d["selected_step"] == 160 and d["batch"] == 4
            and d["broad_exposures"] == d["stdio_exposures"] == 320
            and d["samples_repeated_in_optimizer"] is False and d["model_promotion"] is False
            and d["original_test_split_loaded"] is False and d["public_exercism_tasks_used"] is False
            and d["human_reviewed_independent_benchmark"] is False
            and d["generated_java_executed"] == "BEST_EFFORT_OCI_ONLY"
            and d["gate"]["decision"] == "NO_VALIDATED_BEHAVIOR_GAIN"
            and d["gate"]["validated_gain"] is False and d["gate"]["production_adoption"] is False,
            "STDIO scope mismatch")
    result = {}
    for name in ("baseline_behavior", "selected_behavior"):
        m = d[name]
        require(m["tasks"] == 6 and m["total_cases"] == m["oci_invocations"] == 18
                and [r["id"] for r in m["cases"]] == TASKS and m["cases_passed"] == 0
                and m["tasks_all_pass"] == 0 and m["source_persisted"] is False
                and m["java_executed_on_python_host"] is False and m["unavailable"] is False, "STDIO rows")
        counts = Counter()
        for r in m["cases"]:
            require(r["cases_total"] == r["cases_oci_invoked"] == len(r["case_categories"]) == 3
                    and r["cases_passed"] == 0 and r["task_pass"] is False
                    and set(r["case_categories"]) <= {"NONZERO_EXIT_OR_JAVAC_ERROR", "OUTPUT_MISMATCH_ZERO_EXIT"}
                    and r["source_saved"] is False and r["unavailable"] is False
                    and hexsha(r["source_sha256"]), "STDIO case mismatch")
            counts.update(r["case_categories"])
        require(sum(counts.values()) == 18, "STDIO total mismatch")
        result[name] = dict(sorted(counts.items()))
    return {"historical_cases_recounted": 36, "historical_case_categories": result,
            "candidate_checkpoint_sha256_declared": SEMANTIC_CANDIDATE,
            "parent_checkpoint_sha256_declared": BASE, "selected_step_declared": 160,
            "source_pool_declarations": ["sft/train.jsonl", "stdio/train.jsonl",
                "stdio/validation.jsonl", "derived/stdio_validation_oracles.jsonl"],
            "checkpoint_bytes_verified_here": False, "actual_exposure_verified": False}


def stamp(value):
    require(isinstance(value, str) and value.endswith("Z"), "Invalid timestamp")
    return datetime.fromisoformat(value[:-1] + "+00:00")


def normalize(raw, live=False):
    row = {"id": raw["id"], "attempt": raw["run_attempt" if live else "attempt"],
           "commit_sha": raw["head_sha" if live else "commit_sha"],
           **{k: raw[k] for k in ("path", "event", "status", "conclusion", "created_at", "updated_at")}}
    require(type(row["id"]) is int and row["id"] > 0 and type(row["attempt"]) is int
            and row["attempt"] > 0 and GIT_SHA.fullmatch(row["commit_sha"])
            and all(isinstance(row[k], str) and 0 < len(row[k]) <= 500
                    for k in ("path", "event", "status"))
            and (row["conclusion"] is None or isinstance(row["conclusion"], str))
            and stamp(row["created_at"]) <= stamp(row["updated_at"]), "Invalid run metadata")
    return row


def unique_runs(rows):
    require(len(rows) == len({r["id"] for r in rows}), "Duplicate run ID")
    return {r["id"]: r for r in rows}


def read_snapshot(path):
    raw = blob_pin(path, SNAPSHOT_BLOB)
    d = parse(raw)
    require(d["schema"] == "sagallm-actions-inventory-v1" and d["total_count"] == len(d["runs"]) == 229,
            "Snapshot scope")
    old = unique_runs([normalize(r) for r in d["runs"]])
    require(dict(Counter(r["conclusion"] or r["status"] for r in old.values())) == d["conclusions"],
            "Snapshot conclusion drift")
    return old, sha(raw)


def live_pages(path, cutoff):
    pages = parse(read_bounded(path, 64 * 1024 * 1024))
    require(isinstance(pages, list) and 1 <= len(pages) <= 10, "Pagination bound")
    count = pages[0]["total_count"]
    require(type(count) is int and 0 < count <= 1000
            and all(p["total_count"] == count and len(p["workflow_runs"]) <= 100 for p in pages),
            "Changing/oversized API count")
    rows = [normalize(r, live=True) for p in pages for r in p["workflow_runs"]]
    require(len(rows) == count and all(stamp(r["created_at"]) <= stamp(cutoff) for r in rows),
            "Incomplete or out-of-window page set")
    return unique_runs(rows)


def classify(path):
    name = path.split("@", 1)[0]
    if name.startswith(".github/workflows/") and name.endswith(".yml"):
        stem = name[len(".github/workflows/"):-4]
        if stem in RESEARCH:
            return "RESEARCH_OR_TRAINING_REVIEW"
        if stem in INFRASTRUCTURE:
            return "INFRASTRUCTURE_SCOPE_NOT_CERTIFIED"
    return "UNKNOWN_REVIEW_REQUIRED"


def compare_runs(old, current, archived):
    require(len(archived) == 15 and len({r["run_id"] for r in archived}) == 15, "Archive run coverage")
    archive_by_id = {r["run_id"]: r for r in archived}
    for run, r in archive_by_id.items():
        require(run in old and old[run]["commit_sha"] == r["commit_sha"]
                and old[run]["conclusion"] == "success", "Original archive/run snapshot mismatch")
    missing = sorted(set(old) - set(current))
    added = sorted(set(current) - set(old))
    identity_fields = ("commit_sha", "path", "event", "created_at")
    conflicts = sorted(k for k in set(old) & set(current)
                       if any(old[k][f] != current[k][f] for f in identity_fields))
    changes = sorted(k for k in set(old) & set(current) if old[k] != current[k])
    output = []
    for run, r in sorted(current.items()):
        scope = classify(r["path"])
        if run in archive_by_id:
            evidence = "ARCHIVED_REPORT_PINNED" if r["commit_sha"] == archive_by_id[run]["commit_sha"] else "ARCHIVE_COMMIT_CONFLICT"
        elif run in ANCESTOR_RUNS:
            evidence = "ANCESTOR_ARTIFACT_REFERENCED_NOT_IN_REPORT_ARCHIVE"
        else:
            evidence = "NO_PINNED_REPORT_IN_THIS_SCOPE"
        output.append({**r, "classification": scope, "evidence": evidence,
                       "actual_exposure_verified": False})
    queue = [r["id"] for r in output if r["evidence"] == "NO_PINNED_REPORT_IN_THIS_SCOPE"
             and r["classification"] != "INFRASTRUCTURE_SCOPE_NOT_CERTIFIED"]
    return {"old_count": len(old), "live_count": len(current), "missing_old_run_ids": missing,
            "new_run_ids": added, "changed_metadata_run_ids": changes, "identity_conflicts": conflicts,
            "live_classifications": dict(Counter(r["classification"] for r in output)),
            "unarchived_research_or_unknown_run_ids": queue,
            "archive_run_ids": sorted(archive_by_id), "runs": output,
            "all_old_runs_visible": not missing, "identity_conflicts_absent": not conflicts,
            "all_runs_and_attempts_complete": False,
            "note": "Latest visible attempts at a cutoff; not an atomic or complete historical log. Success is not evidence of training."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("experiments/archive"))
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--live-pages", type=Path, required=True)
    p.add_argument("--cutoff", required=True)
    a = p.parse_args()
    primary = load_manifest(a.archive / "manifest.json")
    require(len(primary) == 12, "Primary archive scope")
    for r in primary:
        pin(a.archive / "reports" / f"{r['run_id']}-{r['slug']}.json", r["report_sha256"], 8 * 1024 * 1024)
    rows, data = legacy_reports(a.legacy)
    old, snapshot_sha = read_snapshot(a.legacy / "actions-inventory-2026-11-25.json")
    current = live_pages(a.live_pages, a.cutoff)
    out = {"schema": "sagallm-legacy-run-discovery-v1", "status": "SCOPED_HOSTED_RUN_DISCOVERY",
           "legacy_revision": LEGACY_REVISION, "historical_snapshot_sha256": snapshot_sha,
           "cutoff_utc": a.cutoff, "archive_report_count": 15,
           "compile_scope": compile_scope(data["v018-paired-compile"]),
           "public_scope": public_scope(data["public-unreviewed-behavior"]),
           "stdio_scope": stdio_scope(data["stdio-semantic-sft"]),
           "run_discovery": compare_runs(old, current, primary + rows),
           "actual_exposure_verified": False, "training_lineage_verified": False,
           "full_exposure_inventory": "NOT_ESTABLISHED", "rights_review": "REVIEW_REQUIRED",
           "independent_evaluation": "NOT_RUN", "model_selection_permitted": False,
           "new_training": False, "new_inference": False, "new_java_execution": False}
    print(json.dumps(out, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
