"""Preserve and verify exact recovered report bytes, independently of expiry.

Only metadata JSON already screened in the pinned recovery artifact is admitted.
No source, weights, logs or model execution.
"""
import argparse
from pathlib import Path
import json
import zipfile
from collections import Counter

from scripts.audit_ranking_auxiliary import pin, parse, sha, require
from scripts.audit_unarchived_artifact_triage import read_evidence, gate
from scripts.audit_recovered_report_ledger import read_recovery, validate
from scripts.audit_recover_unarchived_reports import PRIOR_ZIP as TRIAGE_ZIP, screen, ALLOWED
from scripts.audit_recovered_checkpoint_bytes import RECOVERY_ZIP

ROOT = Path("experiments/archive/recovered-2026-11-28")
MANIFEST_SHA = "8d5181f65d442a3aa283c3de606fc51fe2e73c0eae1a4005e6d63b8662d2a251"
MANIFEST_BYTES = 31365
RECOVERY_ARTIFACT = 15240845377
RECOVERY_RUN = 46088835743
EXPECTED_COUNT = 35
EXPECTED_BYTES = 426121


def manifest_files(m):
    gate(m)
    require(m["schema"] == "sagallm-recovered-historical-reports-v1"
            and m["status"] == "BOUNDED_RECOVERY_ATTEMPT_COMPLETE"
            and m["candidate_count"] == m["zip_verified_count"] == len(m["artifacts"]) == 33
            and m["all_candidate_archives_verified"] is True and m["all_reports_recovered"] is False
            and m["preserved_report_count"] == EXPECTED_COUNT and m["preserved_bytes"] == EXPECTED_BYTES
            and m["new_training"] is False and m["new_inference"] is False
            and m["new_java_execution"] is False, "Pinned manifest scope mismatch")
    require(m["prior_evidence"]["archive_sha256"] == TRIAGE_ZIP, "Triage pin mismatch")
    require(len({r["artifact_id"] for r in m["artifacts"]}) == 33, "Duplicate original artifact")
    out = {}
    for r in m["artifacts"]:
        require(r["archive_bytes_verified"] is True and r["status"] == "ZIP_VERIFIED"
                and [x["member"] for x in r["report_members"]] == list(ALLOWED.get(r["artifact_name"], ())),
                "Original archive or report scope mismatch")
        for i, row in enumerate(r["report_members"]):
            filename = f"{r['run_id']}-{r['artifact_id']}-{i}.json"
            require(row["status"] == "METADATA_SCREEN_PASSED_NOT_RIGHTS_REVIEW"
                    and row["preserved_file"] == filename and filename not in out
                    and type(row["bytes"]) is int and row["bytes"] > 0, "Report member gate mismatch")
            out[filename] = row
    require(len(out) == EXPECTED_COUNT and sum(r["bytes"] for r in out.values()) == EXPECTED_BYTES,
            "Original count/bytes mismatch")
    return out


def verify_local(root):
    require(root.is_dir() and not root.is_symlink(), "Missing archive directory")
    require({p.name for p in root.iterdir()} == {"recovery-manifest.json", "reports"}, "Unexpected archive root members")
    raw = pin(root / "recovery-manifest.json", MANIFEST_SHA)
    require(len(raw) == MANIFEST_BYTES, "Manifest size mismatch")
    m = parse(raw)
    wanted = manifest_files(m)
    reports = root / "reports"
    require(reports.is_dir() and not reports.is_symlink()
            and {p.name for p in reports.iterdir()} == set(wanted), "Original file coverage mismatch")
    for name, row in sorted(wanted.items()):
        body = pin(reports / name, row["report_sha256"], row["bytes"])
        require(len(body) == row["bytes"] and screen(body) == row["status"], "Original report mismatch")
    return {"status": "EXACT_ORIGINALS_VERIFIED_LOCALLY", "manifest_sha256": MANIFEST_SHA,
            "recovery_artifact_id": RECOVERY_ARTIFACT, "recovery_run_id": RECOVERY_RUN,
            "recovery_zip_sha256": RECOVERY_ZIP, "original_report_instances": len(wanted),
            "distinct_report_digests": len({r["report_sha256"] for r in wanted.values()}),
            "preserved_bytes": sum(r["bytes"] for r in wanted.values()),
            "requires_actions_artifact_for_local_verification": False,
            "actual_exposure_verified": False, "training_lineage_verified": False,
            "full_exposure_inventory": "NOT_ESTABLISHED", "rights_review": "REVIEW_REQUIRED",
            "independent_evaluation": "NOT_RUN", "model_selection_permitted": False}


def stage(root, recovery_zip, triage_zip):
    require(not root.exists() and not root.is_symlink(), "Refuse overwrite")
    manifest, blobs = read_recovery(recovery_zip, RECOVERY_ZIP)
    td, _ = read_evidence(triage_zip, TRIAGE_ZIP, ["unarchived-artifact-triage.json"])
    validate(manifest, blobs, td["unarchived-artifact-triage.json"])
    with zipfile.ZipFile(recovery_zip) as z:
        raw = z.read("recovery-manifest.json")
    require(len(raw) == MANIFEST_BYTES and sha(raw) == MANIFEST_SHA, "Original manifest digest mismatch")
    wanted = manifest_files(manifest)
    require(set(blobs) == {"reports/" + n for n in wanted}, "Exact ZIP membership mismatch")
    root.mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "recovery-manifest.json").write_bytes(raw)
    for name in sorted(wanted):
        with (root / "reports" / name).open("xb") as f:
            f.write(blobs["reports/" + name])
    return verify_local(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--recovery-zip", type=Path)
    p.add_argument("--triage-zip", type=Path)
    p.add_argument("--stage", action="store_true")
    a = p.parse_args()
    if a.stage:
        require(a.recovery_zip and a.triage_zip, "Stage requires exact evidence ZIPs")
        result = stage(a.root, a.recovery_zip, a.triage_zip)
    else:
        result = verify_local(a.root)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
