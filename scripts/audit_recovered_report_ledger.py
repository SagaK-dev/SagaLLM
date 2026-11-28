"""Verify recovered report originals and index declared checkpoint evidence.

Hashes are statements in historical reports, not verified checkpoint bytes.
No execution or checkpoint loading.
"""
import argparse
from collections import defaultdict, Counter
import io
import json
from pathlib import PurePosixPath, Path
import re
import zipfile

from scripts.audit_ranking_auxiliary import pin, parse, sha, require
from scripts.audit_unarchived_artifact_triage import read_evidence, gate, hexsha
from scripts.audit_recover_unarchived_reports import (
    PRIOR_ZIP as TRIAGE_ZIP, ALLOWED, MAX_REPORT, MAX_TOTAL_REPORT_BYTES, MAX_ARTIFACTS,
    screen, queue, PRIOR_ARTIFACT as TRIAGE_ARTIFACT, PRIOR_RUN as TRIAGE_RUN,
)

# Field names are explicit; generated-source hashes and optimizer schedule hashes
# must never be mistaken for model checkpoint identities.
CHECKPOINT_FIELDS = {
    "checkpoint_sha256": "UNQUALIFIED_CHECKPOINT_DECLARATION",
    "best_checkpoint_sha256": "BEST",
    "last_checkpoint_sha256": "LAST",
    "checkpoint_sha256_base": "PARENT",
    "base_checkpoint_sha256": "PARENT",
    "origin_checkpoint_sha256": "PARENT",
    "candidate_checkpoint_sha256": "CANDIDATE",
    "contrastive_candidate_sha256": "CANDIDATE",
    "baseline_sha256": "BASELINE_REFERENCE",
    "best_sha256": "BEST_REFERENCE",
    "v017_checkpoint_sha256": "BASELINE_REFERENCE",
    "v018_checkpoint_sha256": "BASELINE_REFERENCE",
    "base_sha256": "BASE_REFERENCE",
}
DATA_FIELDS = {
    "data_manifest_sha256", "dataset_manifest_sha256", "synthetic_stdio_manifest_sha256",
    "stdio_manifest_sha256", "broad_train_sha256", "train_sha256",
    "holdback_sha256", "fresh_suite_sha256", "previous_suite_sha256",
    "suite_sha256", "pair_manifest_sha256", "tokenizer_sha256",
}
SAFE_KEY = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
MAX_EVIDENCE = 40 * 1024 * 1024


def read_recovery(path, digest):
    raw = pin(path, digest, MAX_EVIDENCE)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries = z.infolist()
        names = [m.filename for m in entries]
        require(len(names) == len(set(names)) and 1 <= len(names) <= MAX_ARTIFACTS * 3 + 1,
                "Unexpected recovery ZIP members")
        require("recovery-manifest.json" in names, "Missing recovery manifest")
        blobs = {}
        total = 0
        for m in entries:
            name = m.filename
            require(name == "recovery-manifest.json" or re.fullmatch(r"reports/[0-9]+-[0-9]+-[0-9]+\.json", name),
                    "Unexpected recovery member name")
            require(not m.is_dir() and 0 < m.file_size <= MAX_REPORT, "Recovery member size")
            total += m.file_size
            require(total <= MAX_TOTAL_REPORT_BYTES + MAX_REPORT, "Recovery total size")
            with z.open(m) as f:
                body = f.read(MAX_REPORT + 1)
            require(len(body) == m.file_size <= MAX_REPORT, "Recovery member length")
            blobs[name] = body
    manifest = parse(blobs.pop("recovery-manifest.json"))
    return manifest, blobs


def validate(manifest, blobs, triage):
    gate(manifest)
    require(manifest["schema"] == "sagallm-recovered-historical-reports-v1"
            and manifest["status"] == "BOUNDED_RECOVERY_ATTEMPT_COMPLETE"
            and manifest["all_reports_recovered"] is False
            and manifest["new_training"] is False and manifest["new_inference"] is False
            and manifest["new_java_execution"] is False, "Recovery gates mismatch")
    prior = manifest["prior_evidence"]
    require(prior["archive_sha256"] == TRIAGE_ZIP and prior["run_id"] == TRIAGE_RUN
            and prior["artifact_id"] == TRIAGE_ARTIFACT, "Recovery prior mismatch")
    candidates = queue(triage)
    records = manifest["artifacts"]
    require(len(records) == len(candidates) == manifest["candidate_count"], "Recovery candidate coverage")
    seen, result = set(), []
    verified = 0
    for rec, cand in zip(records, candidates):
        require(rec["run_id"] == cand["run_id"] and rec["artifact_id"] == cand["artifact_id"],
                "Recovery identity mismatch")
        if rec["archive_bytes_verified"] is False:
            require(rec["status"] in {"RECOVERY_BLOCKED", "EXPIRED_SINCE_TRIAGE"}
                    and not rec.get("report_members"), "Blocked archive inconsistency")
            continue
        require(rec["archive_bytes_verified"] is True and rec["status"] == "ZIP_VERIFIED"
                and rec["archive_sha256"] == cand["api_zip_sha256"]
                and rec["commit_sha"] == cand["commit_sha"] and rec["artifact_name"] == cand["name"],
                "Verified archive pin mismatch")
        verified += 1
        wanted = ALLOWED.get(cand["name"], ())
        members = rec["report_members"]
        require([m["member"] for m in members] == list(wanted), "Report allowlist drift")
        rec_bytes = 0
        for index, m in enumerate(members):
            if m["status"] != "METADATA_SCREEN_PASSED_NOT_RIGHTS_REVIEW":
                require("preserved_file" not in m, "Blocked member preserved")
                continue
            expected = f"{cand['run_id']}-{cand['artifact_id']}-{index}.json"
            key = "reports/" + expected
            require(m["preserved_file"] == expected and key not in seen and key in blobs,
                    "Preserved member identity mismatch")
            raw = blobs[key]
            require(len(raw) == m["bytes"] and sha(raw) == m["report_sha256"]
                    and screen(raw) == m["status"], "Preserved report mismatch")
            seen.add(key)
            rec_bytes += len(raw)
            result.append({"run_id": cand["run_id"], "artifact_id": cand["artifact_id"],
                "commit_sha": cand["commit_sha"], "archive_sha256": cand["api_zip_sha256"],
                "artifact_name": cand["name"], "member": m["member"], "report_sha256": m["report_sha256"],
                "preserved_file": key, "bytes": len(raw), "data": parse(raw)})
        require(rec_bytes == rec["preserved_bytes"], "Archive preservation count mismatch")
    require(seen == set(blobs) and len(result) == manifest["preserved_report_count"]
            and sum(r["bytes"] for r in result) == manifest["preserved_bytes"]
            and verified == manifest["zip_verified_count"]
            and manifest["all_candidate_archives_verified"] is (verified == len(records)),
            "Recovery aggregate mismatch")
    require(dict(Counter(r["status"] for r in records)) == manifest["artifact_status_counts"]
            and dict(Counter(m["status"] for r in records for m in r.get("report_members", [])))
            == manifest["report_member_status_counts"], "Recovery status aggregate mismatch")
    return result


def declarations(d):
    models, sources = [], []
    stack = [((), d)]
    while stack:
        path, value = stack.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                # Do not publish arbitrary historical keys as field paths.
                token = key if SAFE_KEY.fullmatch(key) else "<redacted-key>"
                child = path + (token,)
                if key in CHECKPOINT_FIELDS and hexsha(item):
                    models.append({"field_path": list(child), "sha256": item, "role": CHECKPOINT_FIELDS[key]})
                if key in DATA_FIELDS and hexsha(item):
                    sources.append({"field_path": list(child), "sha256": item, "field": key})
                if isinstance(item, (dict, list)):
                    stack.append((child, item))
        elif isinstance(value, list):
            stack.extend((path + (str(i),), item) for i, item in enumerate(value)
                         if isinstance(item, (dict, list)))
    return sorted(models, key=lambda r: (r["field_path"], r["sha256"])), sorted(sources, key=lambda r: r["field_path"])


def ledger(records, triage):
    combined = triage["combined_scope"]
    gate(combined)
    known = {r["checkpoint_sha256"] for r in combined["checkpoint_declarations"]}
    require(len(known) == combined["checkpoint_count"] == 18, "Known checkpoint scope mismatch")
    unique = defaultdict(list)
    claims = defaultdict(list)
    source_pins = defaultdict(list)
    for rec in records:
        meta = {k: v for k, v in rec.items() if k != "data"}
        unique[rec["report_sha256"]].append(meta)
        model_rows, data_rows = declarations(rec["data"])
        for row in model_rows:
            claims[row["sha256"]].append({"run_id": rec["run_id"], "artifact_id": rec["artifact_id"],
                "report_sha256": rec["report_sha256"], **row})
        for row in data_rows:
            source_pins[row["sha256"]].append({"run_id": rec["run_id"], "artifact_id": rec["artifact_id"],
                "report_sha256": rec["report_sha256"], **row})
    # All hashes not yet in the prior map are review candidates, not new trained models.
    return {"schema": "sagallm-recovered-report-ledger-v1",
            "status": "RECOVERED_ORIGINALS_REVERIFIED_AND_INDEXED",
            "preserved_report_instances": len(records), "distinct_report_sha256_count": len(unique),
            "duplicate_report_instances": len(records) - len(unique),
            "report_groups": [{"report_sha256": h, "instances": v} for h,v in sorted(unique.items())],
            "known_checkpoint_declaration_count": len(known),
            "checkpoint_hash_claims": [{"sha256": h, "known_in_prior_scope": h in known,
                    "claims": v, "checkpoint_bytes_verified_here": False} for h,v in sorted(claims.items())],
            "unmapped_checkpoint_hash_claims": sorted(set(claims) - known),
            "source_hash_declarations": [{"sha256": h, "claims": v,
                    "source_bytes_verified_here": False} for h,v in sorted(source_pins.items())],
            "no_new_checkpoint_certification": True, "actual_exposure_verified": False,
            "training_lineage_verified": False, "full_exposure_inventory": "NOT_ESTABLISHED",
            "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recovery-zip", type=Path, required=True)
    p.add_argument("--recovery-sha256", required=True)
    p.add_argument("--triage-zip", type=Path, required=True)
    a = p.parse_args()
    d, hashes = read_evidence(a.triage_zip, TRIAGE_ZIP, ["unarchived-artifact-triage.json"])
    triage = d["unarchived-artifact-triage.json"]
    manifest, blobs = read_recovery(a.recovery_zip, a.recovery_sha256)
    rows = validate(manifest, blobs, triage)
    out = ledger(rows, triage)
    out["recovery_archive_sha256"] = a.recovery_sha256
    out["triage_archive_sha256"] = TRIAGE_ZIP
    print(json.dumps(out, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
