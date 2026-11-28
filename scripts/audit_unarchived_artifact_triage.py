"""Join audited scope and inspect metadata for unarchived hosted-run artifacts.

Never download candidate archives, load model state, execute Java, or train.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import subprocess
import zipfile

from scripts.audit_ranking_auxiliary import LIMIT, pin, parse, sha, require
from scripts.audit_legacy_run_discovery import SPECS, SEMANTIC_CANDIDATE, PUBLIC_SUITE, BASE, stamp, hexsha

DISCOVERY_RUN = 46087589108
DISCOVERY_ARTIFACT = 15239323736
DISCOVERY_ZIP = "3c7cbf0b6b2974f792eb6a87db36e466bf5ab45ebe255563342479d16f8c2ee9"
INTEGRATED_RUN = 46087393977
INTEGRATED_ARTIFACT = 15240220228
INTEGRATED_ZIP = "7694851ed7a074b7171ef3278e756ddb48fd79e8110b9878000f7fe16052c396"
MAX_CANDIDATES = 150
MAX_ARCHIVE = 128 * 1024 * 1024


def read_evidence(path, digest, names):
    raw = pin(path, digest)
    out, hashes = {}, {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries = z.infolist()
        require(len(entries) == len(names) and {m.filename for m in entries} == set(names),
                "Unexpected evidence archive members")
        for m in entries:
            require(0 < m.file_size <= LIMIT, "Evidence member size limit")
            with z.open(m) as f:
                body = f.read(LIMIT + 1)
            require(len(body) == m.file_size <= LIMIT, "Evidence member length mismatch")
            out[m.filename], hashes[m.filename] = parse(body), sha(body)
    return out, hashes


def gate(row):
    require(row["actual_exposure_verified"] is False and row["training_lineage_verified"] is False
            and row["full_exposure_inventory"] == "NOT_ESTABLISHED"
            and row["independent_evaluation"] == "NOT_RUN"
            and row["model_selection_permitted"] is False, "Evidence scope gate mismatch")


def join_scope(primary, discovery):
    gate(primary)
    gate(discovery)
    require(primary["status"] == "ARCHIVED_REPORT_SCOPE_INTEGRATED_NOT_COMPLETE"
            and primary["archive_report_count"] == 12
            and primary["checkpoint_count"] == 17 and primary["inventory_entries"] == 15
            and primary["inventory_ready_for_independent_evaluation"] is False
            and discovery["status"] == "SCOPED_HOSTED_RUN_DISCOVERY"
            and discovery["archive_report_count"] == 15, "Scope count mismatch")
    coverage = primary["archive_report_scope"]
    old_ids = {r["run_id"] for r in coverage}
    ids = discovery["run_discovery"]["archive_run_ids"]
    require(len(coverage) == len(old_ids) == 12 and len(ids) == len(set(ids)) == 15
            and set(ids) == old_ids | {x[0] for x in SPECS.values()}, "Report coverage mismatch")
    records = primary["checkpoint_declarations"]
    require(len(records) == len({r["checkpoint_sha256"] for r in records}) == 17
            and len({r["model"] for r in records}) == 17
            and all(hexsha(r["checkpoint_sha256"]) and r["actual_exposure_verified"] is False
                    and r["checkpoint_bytes_verified_here"] is False and r["inventory_ready"] is False
                    for r in records), "Primary checkpoint gate mismatch")
    require(len(primary["unresolved_checkpoint_declarations"]) == 1
            and primary["unresolved_checkpoint_declarations"][0]["checkpoint_sha256"] is None,
            "Unresolved control declaration mismatch")
    semantic = discovery["stdio_scope"]
    require(semantic["candidate_checkpoint_sha256_declared"] == SEMANTIC_CANDIDATE
            and semantic["parent_checkpoint_sha256_declared"] == BASE
            and semantic["selected_step_declared"] == 160
            and semantic["actual_exposure_verified"] is False
            and semantic["checkpoint_bytes_verified_here"] is False
            and SEMANTIC_CANDIDATE not in {r["checkpoint_sha256"] for r in records},
            "Semantic candidate mismatch")
    same_pool = next(r for r in records if r["model"] == "contrastive-stdio-ab-candidate")
    require(same_pool["parent_checkpoint_sha256_declared"] == BASE
            and set(semantic["source_pool_declarations"]) ==
                {"sft/train.jsonl", "stdio/train.jsonl", "stdio/validation.jsonl",
                 "derived/stdio_validation_oracles.jsonl"}
            and set(semantic["source_pool_declarations"]) <= {s["path"] for s in same_pool["sources"]}
            and all(hexsha(s["sha256"]) for s in same_pool["sources"]), "Source pool declaration mismatch")
    extra = {"model": "stdio-semantic-sft-candidate",
             "checkpoint_sha256": SEMANTIC_CANDIDATE,
             "parent_checkpoint_sha256_declared": BASE,
             "selected_or_last_step_declared": 160, "source_report_run_id": SPECS["stdio-semantic-sft"][0],
             "source_assignment_basis": "CONSERVATIVE_SAME_DECLARED_POOLS_NOT_SAME_SAMPLE_ORDER",
             "sources": same_pool["sources"],
             "actual_exposure_verified": False, "checkpoint_bytes_verified_here": False,
             "inventory_ready": False}
    public = discovery["public_scope"]
    require(public["suite_sha256_declared"] == PUBLIC_SUITE and public["source_bytes_verified"] is False
            and public["matched_full_suite"] is False and public["independent_evaluation"] == "NOT_RUN",
            "Public suite gate mismatch")
    return {"archive_report_count": 15, "archive_report_run_ids": sorted(ids),
            "legacy_report_pins": [{"slug": slug, "run_id": r, "artifact_id": a, "report_sha256": h}
                                   for slug, (r, a, h) in sorted(SPECS.items())],
            "checkpoint_count": 18, "checkpoint_declarations": records + [extra],
            "unresolved_checkpoint_declarations": primary["unresolved_checkpoint_declarations"],
            "inventory_entries": 15, "inventory_sha256": primary["inventory_sha256"],
            "new_inventory_entries": 0,
            "missing_source_declarations": [
                {"source": "public-unreviewed-behavior-suite", "sha256_declared": PUBLIC_SUITE,
                 "role": "development_probe_context", "source_bytes_verified": False,
                 "rights_review": "REVIEW_REQUIRED", "independence": "NOT_ESTABLISHED"}],
            "actual_exposure_verified": False, "training_lineage_verified": False,
            "inventory_ready_for_independent_evaluation": False,
            "full_exposure_inventory": "NOT_ESTABLISHED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False}


def review_queue(discovery):
    d = discovery["run_discovery"]
    rows = d["runs"]
    require(len(rows) == d["live_count"] == len({r["id"] for r in rows}), "Duplicate or incomplete run list")
    queue = [r for r in rows if r["evidence"] == "NO_PINNED_REPORT_IN_THIS_SCOPE"
             and r["classification"] != "INFRASTRUCTURE_SCOPE_NOT_CERTIFIED"]
    require([r["id"] for r in queue] == d["unarchived_research_or_unknown_run_ids"]
            and 0 < len(queue) <= MAX_CANDIDATES
            and all(type(r["id"]) is int and r["id"] > 0 and r["actual_exposure_verified"] is False
                    for r in queue), "Review queue mismatch or limit")
    return queue


def artifact_row(a, run, observed):
    aid = a["id"]
    size = a["size_in_bytes"]
    name = a["name"]
    expired = a["expired"]
    expires = a["expires_at"]
    require(type(aid) is int and aid > 0 and type(size) is int and size >= 0
            and isinstance(name, str) and 0 < len(name) <= 500 and type(expired) is bool
            and stamp(a["created_at"]) <= stamp(expires), "Artifact metadata shape")
    digest = a.get("digest")
    digest_ok = isinstance(digest, str) and digest.startswith("sha256:") and hexsha(digest[7:])
    link = a.get("workflow_run")
    binding = ("MISSING" if link is None else "MATCH" if
               link.get("id") == run["id"] and link.get("head_sha") == run["commit_sha"] else "CONFLICT")
    blockers = []
    if binding != "MATCH":
        blockers.append("RUN_COMMIT_BINDING_" + binding)
    if not digest_ok:
        blockers.append("SHA256_NOT_RECORDED")
    if expired or stamp(expires) <= stamp(observed):
        blockers.append("EXPIRED_OR_PAST_EXPIRY")
    if not 0 < size <= MAX_ARCHIVE:
        blockers.append("SIZE_OUTSIDE_CURRENT_ARCHIVE_LIMIT")
    return {"artifact_id": aid, "name": name, "size_in_bytes": size,
            "api_zip_sha256": digest[7:] if digest_ok else None,
            "expired_api": expired, "expires_at": expires, "run_binding": binding,
            "status": "METADATA_READY_FOR_BYTE_VERIFICATION" if not blockers else "BLOCKED_METADATA",
            "blockers": blockers, "archive_bytes_verified": False, "member_report_sha256": None}


def artifact_pages(pages, run, observed):
    require(isinstance(pages, list) and 1 <= len(pages) <= 10, "Artifact page bound")
    total = pages[0]["total_count"]
    require(type(total) is int and 0 <= total <= 1000
            and all(p["total_count"] == total and len(p["artifacts"]) <= 100 for p in pages),
            "Artifact pagination count mismatch")
    rows = [artifact_row(a, run, observed) for p in pages for a in p["artifacts"]]
    require(len(rows) == total == len({r["artifact_id"] for r in rows}), "Duplicate or missing artifacts")
    return {"run_id": run["id"], "commit_sha": run["commit_sha"], "workflow_path": run["path"],
            "run_status_at_discovery": run["status"], "run_conclusion_at_discovery": run["conclusion"],
            "artifacts": sorted(rows, key=lambda x: x["artifact_id"]),
            "status": ("NO_ARTIFACT_VISIBLE" if not rows else "ARTIFACT_METADATA_REVIEWED"),
            "actual_exposure_verified": False}


def fetch(run, observed):
    endpoint = f"repos/SagaK-dev/SagaLLM/actions/runs/{run['id']}/artifacts"
    try:
        result = subprocess.run(["gh", "api", "--paginate", "--slurp", "-X", "GET", endpoint,
                                 "-f", "per_page=100"], check=True, timeout=45,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise ValueError(f"Artifact metadata retrieval failed for run {run['id']}") from None
    require(0 < len(result.stdout) <= 8 * 1024 * 1024, "Artifact API response size")
    return artifact_pages(parse(result.stdout), run, observed)


def triage(discovery, observed, getter=fetch):
    queue = review_queue(discovery)
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(lambda row: getter(row, observed), queue))
    require([r["run_id"] for r in result] == [r["id"] for r in queue], "Result queue mismatch")
    ids = [a["artifact_id"] for r in result for a in r["artifacts"]]
    require(len(ids) == len(set(ids)), "Artifact repeated across runs")
    ready = sorted(({"run_id": r["run_id"], "commit_sha": r["commit_sha"], **a}
                    for r in result for a in r["artifacts"]
                    if a["status"] == "METADATA_READY_FOR_BYTE_VERIFICATION"),
                   key=lambda r: (stamp(r["expires_at"]), r["run_id"], r["artifact_id"]))
    return {"run_count": len(result), "artifact_count": len(ids),
            "runs_without_visible_artifacts": [r["run_id"] for r in result if not r["artifacts"]],
            "artifact_status_counts": dict(Counter(a["status"] for r in result for a in r["artifacts"])),
            "priority_by_expiry": ready, "runs": result,
            "archive_bytes_verified": False, "actual_exposure_verified": False,
            "note": "API metadata at inspection only; missing artifacts do not prove no training."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--discovery-zip", type=Path, required=True)
    p.add_argument("--integrated-zip", type=Path, required=True)
    p.add_argument("--observed-at", required=True)
    args = p.parse_args()
    d, dh = read_evidence(args.discovery_zip, DISCOVERY_ZIP, ["legacy-run-discovery.json"])
    i, ih = read_evidence(args.integrated_zip, INTEGRATED_ZIP, ["integrated-exposure-scope.json",
        "early-ab-reconstruction.json", "early-m1-reconstruction.json", "early-blocked-reconstruction.json"])
    primary, discovery = i["integrated-exposure-scope.json"], d["legacy-run-discovery.json"]
    for mode in ("ab", "m1", "blocked"):
        row = i[f"early-{mode}-reconstruction.json"]
        require(row["mode"] == mode and row["status"] == "RECONSTRUCTED_SCOPED_INPUT_METADATA"
                and row["model_loaded"] is False and row["actual_exposure_verified"] is False,
                "Historical reconstruction gate mismatch")
    combined = join_scope(primary, discovery)
    result = triage(discovery, args.observed_at)
    print(json.dumps({"schema": "sagallm-unarchived-artifact-triage-v1",
        "status": "SCOPED_ARTIFACT_METADATA_TRIAGE", "observed_at_utc": args.observed_at,
        "discovery_evidence": {"run_id": DISCOVERY_RUN, "artifact_id": DISCOVERY_ARTIFACT,
                               "archive_sha256": DISCOVERY_ZIP, "member_sha256": dh},
        "integration_evidence": {"run_id": INTEGRATED_RUN, "artifact_id": INTEGRATED_ARTIFACT,
                                 "archive_sha256": INTEGRATED_ZIP, "member_sha256": ih},
        "combined_scope": combined, "triage": result,
        "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
        "actual_exposure_verified": False, "training_lineage_verified": False,
        "full_exposure_inventory": "NOT_ESTABLISHED", "model_selection_permitted": False,
        "new_training": False, "new_inference": False, "new_java_execution": False},
        sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
