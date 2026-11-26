"""Index report-declared exposure evidence; never certify completeness."""
import argparse
import json
import re
from pathlib import Path
from scripts.archive_experiment_reports import load_manifest, sha, require, MAX_REPORT
from scripts.audit_v018_training import unique_object

DATA_KEYS = {"data_manifest_sha256", "synthetic_stdio_manifest_sha256",
    "stdio_manifest_sha256", "broad_train_sha256", "holdback_sha256",
    "fresh_suite_sha256", "previous_suite_sha256", "pair_manifest_sha256",
    "regression_oracles_sha256", "manifest_sha256"}
ID_KEYS = {"id", "ids", "task_id", "task_ids", "fixed_task_ids",
    "holdback_behavior_task_ids", "fresh_ids", "regression_ids", "selection_ids"}
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")


def collect(report):
    hashes, ids = set(), set()
    stack = [("", report)]
    while stack:
        key, value = stack.pop()
        if key in DATA_KEYS and isinstance(value, str):
            require(HASH.fullmatch(value), "Invalid declared data digest")
            hashes.add((key, value))
        if key in ID_KEYS:
            values = value if isinstance(value, list) else [value]
            for item in values:
                if isinstance(item, str) and ID.fullmatch(item):
                    ids.add(item)
        if isinstance(value, dict):
            stack.extend(value.items())
        elif isinstance(value, list):
            stack.extend(("", item) for item in value)
    return {"declared_data_hashes": [{"field": k, "sha256": v} for k,v in sorted(hashes)],
            "reported_ids": sorted(ids), "reported_id_count": len(ids)}


def audit(root):
    manifest = root / "manifest.json"
    rows = load_manifest(manifest)
    reports = []
    for row in rows:
        path = root / "reports" / f'{row["run_id"]}-{row["slug"]}.json'
        require(path.is_file() and not path.is_symlink() and
                0 < path.stat().st_size <= MAX_REPORT, "Missing or oversized report")
        raw = path.read_bytes()
        require(sha(raw) == row["report_sha256"], "Report SHA mismatch")
        report = json.loads(raw, object_pairs_hook=unique_object)
        require(isinstance(report, dict), "Invalid report")
        reports.append({**row, "reported_status": report.get("status", "NOT_RECORDED"),
                        **collect(report), "source_bytes_verified": False,
                        "actual_exposure_verified": False})
    return {"schema": "sagallm-historical-exposure-ledger-v1",
            "archive_manifest_sha256": sha(manifest.read_bytes()),
            "report_count": len(reports), "reports": reports,
            "historical_corpus_reconstruction_runs": [45770654618, 45772294170],
            "missing_evidence": ["complete optimizer sample order per checkpoint",
                "raw bytes for all development probes and selection suites",
                "unarchived runs and local experiments",
                "verified checkpoint-specific exposure inventory", "rights review"],
            "full_exposure_inventory": "NOT_ESTABLISHED",
            "independent_evaluation": "NOT_RUN", "model_selection_permitted": False}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("experiments/archive"))
    args = p.parse_args()
    print(json.dumps(audit(args.archive), sort_keys=True, indent=2))
