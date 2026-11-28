"""Verify historical checkpoint member bytes without deserialization or training."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from scripts.audit_ranking_auxiliary import require
from scripts.audit_unarchived_artifact_triage import read_evidence, gate, stamp, hexsha
from scripts.audit_recover_unarchived_reports import bounded_download, queue, PRIOR_ZIP
from scripts.audit_recovered_report_ledger import read_recovery, validate, ledger

RECOVERY_ZIP = "25a0d7cbacdead46d18c7261287a601bf7a2c114c8ff9a5a6ebf22a0e8ba3040"
LEDGER_ZIP = "97d59becda044e504f9d8076282cac382a311bff8194829c1c16fea105872b0b"
RECOVERY_ARTIFACT = 15240845377
LEDGER_ARTIFACT = 15240695936
MAX_MEMBER = 96 * 1024 * 1024
MEMBERS = {
    "sagallm-v016-pinned-java-cpu": {
        "java-model/java-best.pt": "BEST",
        "java-model/java-last.pt": "LAST"},
    "sagallm-v017-java-instruction-sft-cpu": {
        "java-sft-model/java-sft-best.pt": "BEST",
        "java-sft-model/java-sft-last.pt": "LAST"},
    "sagallm-v018-java-context-sft-cpu": {
        "java-context-model/java-sft-best.pt": "BEST",
        "java-context-model/java-sft-last.pt": "LAST"},
    "sagallm-broad-mixture-low-lr-validation-ab": {
        "broad-mix-ab-evidence/canonical-validation-best.pt": "ARM_BEST",
        "broad-mix-ab-evidence/identifier_augmented-validation-best.pt": "ARM_BEST"},
    "sagallm-java-identifier-ab-validation": {
        "identifier-ab-evidence/canonical-validation-best.pt": "ARM_BEST",
        "identifier-ab-evidence/identifier_augmented-validation-best.pt": "ARM_BEST"},
    "sagallm-contrastive-stdio-matched-validation-ab": {
        "contrastive-stdio-ab-evidence/contrastive-validation-candidate.pt": "CANDIDATE"},
    "sagallm-m3-ranking-cpu-research": {
        f"{seed}-{arm}.pt": "ARM_BEST"
        for seed in (4201, 4202, 4203) for arm in ("control", "ranking")},
}
# Only these role types can be used for a conservative same-artifact output match.
OUTPUT_ROLES = {
    "BEST": {"BEST"}, "LAST": {"LAST"}, "CANDIDATE": {"CANDIDATE"},
    "ARM_BEST": {"BEST", "UNQUALIFIED_CHECKPOINT_DECLARATION"},
}


def hash_member(z, member):
    require(not member.is_dir() and 0 < member.file_size <= MAX_MEMBER
            and not member.flag_bits & 1, "Unsafe checkpoint member")
    h = hashlib.sha256()
    total = 0
    with z.open(member) as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            total += len(chunk)
            require(total <= member.file_size <= MAX_MEMBER, "Checkpoint member overflow")
            h.update(chunk)
    require(total == member.file_size, "Checkpoint byte count mismatch")
    return h.hexdigest(), total


def verify_archive(path, candidate, claims):
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size == candidate["size_in_bytes"], "Archive size mismatch")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    require(h.hexdigest() == candidate["api_zip_sha256"], "Archive digest mismatch")
    wanted = MEMBERS[candidate["name"]]
    result = []
    with zipfile.ZipFile(path) as z:
        rows = z.infolist()
        require(len(rows) <= 256 and len({m.filename for m in rows}) == len(rows),
                "Unexpected archive directory")
        names = {m.filename for m in rows}
        for name, role in wanted.items():
            if name not in names:
                result.append({"member": name, "role": role, "status": "MEMBER_NOT_VISIBLE",
                               "checkpoint_bytes_verified": False})
                continue
            digest, size = hash_member(z, z.getinfo(name))
            matched = claims.get(digest, [])
            same_artifact = [r for r in matched if r["artifact_id"] == candidate["artifact_id"]]
            compatible = [r for r in same_artifact if r["role"] in OUTPUT_ROLES[role]]
            if role == "ARM_BEST":
                # A digest match alone cannot establish a named training arm.
                status = "MATCHED_SAME_ARTIFACT_OUTPUT_CLAIM_ARM_UNVERIFIED" if compatible else (
                    "MATCHED_OTHER_DECLARATION" if matched else "UNDECLARED_MEMBER_DIGEST")
            else:
                status = ("MATCHED_SAME_ARTIFACT_OUTPUT_CLAIM" if compatible else
                          "MATCHED_OTHER_DECLARATION" if matched else "UNDECLARED_MEMBER_DIGEST")
            result.append({"member": name, "role": role, "sha256": digest, "bytes": size,
                           "checkpoint_bytes_verified": True, "status": status,
                           "same_artifact_claims": same_artifact,
                           "compatible_same_artifact_output_claims": compatible,
                           "claim_count": len(matched), "checkpoint_deserialized": False,
                           "training_lineage_verified": False})
        return {"run_id": candidate["run_id"], "artifact_id": candidate["artifact_id"],
                "archive_sha256": candidate["api_zip_sha256"], "archive_bytes_verified": True,
                "status": "ARCHIVE_VERIFIED", "members": result,
                "unlisted_pt_members": sum(n.endswith(".pt") and n not in wanted for n in names)}


def inspect(candidate, claims, observed):
    base = {"run_id": candidate["run_id"], "artifact_id": candidate["artifact_id"],
            "archive_bytes_verified": False, "members": []}
    if stamp(candidate["expires_at"]) <= stamp(observed):
        return {**base, "status": "EXPIRED"}
    try:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"archive.zip"
            bounded_download(candidate, path)
            return verify_archive(path, candidate, claims)
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as e:
        return {**base, "status": "BLOCKED", "error_type": type(e).__name__}


def reconcile(triage, indexed, observed, getter=inspect):
    gate(indexed)
    claims = {r["sha256"]: r["claims"] for r in indexed["checkpoint_hash_claims"]}
    require(len(claims) == len(indexed["checkpoint_hash_claims"]) and
            all(hexsha(h) for h in claims), "Claim digest mismatch")
    candidates = [a for a in queue(triage) if a["name"] in MEMBERS]
    require(candidates, "No matching checkpoint archives")
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda a: getter(a, claims, observed), candidates))
    require([r["artifact_id"] for r in rows] == [a["artifact_id"] for a in candidates],
            "Checkpoint archive queue mismatch")
    all_members = [m for r in rows for m in r["members"]]
    verified = [m for m in all_members if m["checkpoint_bytes_verified"] is True]
    observed_hashes = {m["sha256"] for m in verified}
    expected = set(indexed["unmapped_checkpoint_hash_claims"])
    return {"schema": "sagallm-recovered-checkpoint-byte-scope-v1",
            "status": "BOUNDED_CHECKPOINT_BYTE_AUDIT_COMPLETE",
            "candidate_archive_count": len(candidates),
            "archive_status_counts": dict(Counter(r["status"] for r in rows)),
            "member_status_counts": dict(Counter(m["status"] for m in all_members)),
            "verified_member_count": len(verified), "distinct_verified_member_hash_count": len(observed_hashes),
            "unmapped_claims_matched_to_member_bytes": sorted(expected & observed_hashes),
            "unmapped_claims_without_matching_member_bytes": sorted(expected - observed_hashes),
            "member_hashes_without_indexed_claim": sorted(observed_hashes - set(claims)),
            "archives": rows, "checkpoint_deserialized": False, "new_training": False,
            "new_inference": False, "new_java_execution": False,
            "actual_exposure_verified": False, "training_lineage_verified": False,
            "full_exposure_inventory": "NOT_ESTABLISHED", "rights_review": "REVIEW_REQUIRED",
            "independent_evaluation": "NOT_RUN", "model_selection_permitted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("triage", "recovery", "ledger"):
        p.add_argument("--" + name + "-zip", type=Path, required=True)
    p.add_argument("--observed-at", required=True)
    a = p.parse_args()
    td, _ = read_evidence(a.triage_zip, PRIOR_ZIP, ["unarchived-artifact-triage.json"])
    triage = td["unarchived-artifact-triage.json"]
    ld, _ = read_evidence(a.ledger_zip, LEDGER_ZIP, ["recovered-report-ledger.json"])
    indexed = ld["recovered-report-ledger.json"]
    manifest, blobs = read_recovery(a.recovery_zip, RECOVERY_ZIP)
    rebuilt = ledger(validate(manifest, blobs, triage), triage)
    rebuilt["recovery_archive_sha256"] = RECOVERY_ZIP
    rebuilt["triage_archive_sha256"] = PRIOR_ZIP
    require(rebuilt == indexed, "Recovered report ledger drift")
    result = reconcile(triage, indexed, a.observed_at)
    result["recovery_evidence"] = {"artifact_id": RECOVERY_ARTIFACT, "archive_sha256": RECOVERY_ZIP}
    result["ledger_evidence"] = {"artifact_id": LEDGER_ARTIFACT, "archive_sha256": LEDGER_ZIP}
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
