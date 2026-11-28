"""Reconcile recovered checkpoint outputs, parents, and trial/selected-step scope.

Only read pinned report and byte-audit evidence. No tensor equality or actual
optimizer exposure is inferred from serialized checkpoint digests.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.audit_unarchived_artifact_triage import read_evidence, gate, hexsha
from scripts.audit_recovered_report_ledger import read_recovery, validate
from scripts.audit_recovered_checkpoint_bytes import RECOVERY_ZIP, LEDGER_ZIP, MEMBERS
from scripts.audit_recover_unarchived_reports import PRIOR_ZIP as TRIAGE_ZIP
from scripts.audit_v018_training import BEST, BASE, DATA, TOKENIZER
from scripts.audit_java_lineage import ROOT
from scripts.audit_ranking_auxiliary import require

BYTE_RUN = 46089074949
BYTE_ARTIFACT = 15239793469
BYTE_ZIP = "b0b54398be4613aa1b5a29b5be905277f8b197f505a709961bd97c3c74c867bf"
PUBLIC = "9e11217e9bd47d836a680a39e0820cd02f159a62e8e9f2c4cd864dd64ae7bf11"
IDENTIFIER = "ad18e0a328f00db2bca204c3911afd66307099e3d41ba0f0d97c964463d3c2d9"
AB = {
    "sagallm-java-identifier-ab-validation": ("TRAIN_VALIDATION_AB_COMPLETE", "base_checkpoint_sha256", "candidate_sha256", 60),
    "sagallm-broad-mixture-low-lr-validation-ab": ("JAVA_BROAD_MIXTURE_LOW_LR_AB_V1", "origin_model_sha256", "candidate_corpus_sha256", 80),
}
TRAIN = {
    "sagallm-v016-pinned-java-cpu": (None, "dataset_manifest_sha256", PUBLIC, 950),
    "sagallm-v017-java-instruction-sft-cpu": (ROOT, "data_manifest_sha256", DATA, 650),
    "sagallm-v018-java-context-sft-cpu": (BASE, "data_manifest_sha256", DATA, 400),
}


def declared_output(report, name, member):
    require(report["tokenizer_sha256"] == TOKENIZER, "Tokenizer drift")
    if name in TRAIN:
        parent, field, corpus, steps = TRAIN[name]
        require(report["status"] == "TRAINED_CPU" and report[field] == corpus
                and report["optimizer_steps_completed"] == steps
                and report["test_used_for_model_selection"] is False, "Training report scope mismatch")
        if parent is None:
            require(report["random_initialization"] is True and report["external_pretrained_weights"] is False,
                    "Root declaration mismatch")
        else:
            require(report["checkpoint_sha256_base"] == parent, "Parent declaration mismatch")
        role = MEMBERS[name][member]
        field = "best_checkpoint_sha256" if role == "BEST" else "last_checkpoint_sha256"
        selected = report["best_checkpoint_step"] if role == "BEST" else steps
        require(type(selected) is int and 0 < selected <= steps, "Selected step mismatch")
        return {"sha256": report[field], "parent_sha256": parent, "selected_step": selected,
                "trial_steps_completed": steps, "role": role, "arm": None,
                "corpus_manifest_sha256": corpus, "identifier_candidate_sha256": None,
                "field_path": [field], "selection_scope": "REPORTED_POSITIVE_STEP_OUTPUT"}
    require(name in AB, "Unreviewed checkpoint artifact class")
    status, parent_field, candidate_field, steps = AB[name]
    require(report["status"] == status and report[parent_field] == BEST
            and report["canonical_manifest_sha256"] == DATA and report[candidate_field] == IDENTIFIER
            and report["steps_per_arm"] == steps and report["decision"]["production_adoption"] is False,
            "A/B report scope mismatch")
    if name == "sagallm-java-identifier-ab-validation":
        require(report["selection_uses_only_validation"] is True and report["test_used_for_selection"] is False
                and report["test_split_evaluated"] is False and report["generated_java_executed"] is False,
                "Identifier trial gate mismatch")
    else:
        require(report["selection_uses_validation_only"] is True and report["test_split_read_or_scored"] is False
                and report["Java_programs_executed"] is False and report["production_model_unchanged"] is True,
                "Broad mixture trial gate mismatch")
    arm = member.rsplit("/", 1)[1].removesuffix("-validation-best.pt")
    require(arm in ("canonical", "identifier_augmented")
            and set(report["results"]) == {"canonical", "identifier_augmented"}, "Arm naming mismatch")
    row = report["results"][arm]
    require(row["arm"] == arm and row["optimizer_steps_completed"] == steps
            and row["selected_step"] == 0 and row["initial_validation_loss"] == row["best_validation_loss"]
            and all(h["validation_response_loss"] >= row["initial_validation_loss"] for h in row["history"]),
            "Step-zero selection mismatch")
    return {"sha256": row["checkpoint_sha256"], "parent_sha256": BEST, "selected_step": 0,
            "trial_steps_completed": steps, "role": "ARM_BEST", "arm": arm,
            "corpus_manifest_sha256": DATA, "identifier_candidate_sha256": IDENTIFIER,
            "field_path": ["results", arm, "checkpoint_sha256"],
            "selection_scope": "REPORTED_STEP_ZERO_SELECTION_NOT_TENSOR_EQUALITY"}


def acyclic(parents):
    for node in parents:
        seen = set()
        while node is not None:
            require(node in parents and node not in seen, "Unknown parent or cyclic declaration")
            seen.add(node)
            node = parents[node]


def reconcile(records, scope, byte):
    gate(scope)
    gate(byte)
    require(byte["status"] == "BOUNDED_CHECKPOINT_BYTE_AUDIT_COMPLETE"
            and byte["recovery_evidence"]["archive_sha256"] == RECOVERY_ZIP
            and byte["ledger_evidence"]["archive_sha256"] == LEDGER_ZIP
            and byte["checkpoint_deserialized"] is False
            and byte["candidate_archive_count"] == 17 and byte["verified_member_count"] == 34
            and byte["distinct_verified_member_hash_count"] == 18
            and byte["unmapped_claims_without_matching_member_bytes"] == []
            and byte["member_hashes_without_indexed_claim"] == [], "Byte audit scope mismatch")
    known = {r["checkpoint_sha256"]: r for r in scope["checkpoint_declarations"]}
    require(len(known) == scope["checkpoint_count"] == 18, "Prior scope count")
    reports = {r["artifact_id"]: r for r in records if r["artifact_name"] in TRAIN or r["artifact_name"] in AB}
    require(len(reports) == 17, "Training report coverage mismatch")
    grouped = defaultdict(list)
    for a in byte["archives"]:
        rec = reports[a["artifact_id"]]
        name = rec["artifact_name"]
        require(a["run_id"] == rec["run_id"] and a["archive_sha256"] == rec["archive_sha256"]
                and a["archive_bytes_verified"] is True
                and {m["member"] for m in a["members"]} == set(MEMBERS[name]), "Archive/member identity mismatch")
        for m in a["members"]:
            d = declared_output(rec["data"], name, m["member"])
            require(hexsha(d["sha256"]) and d["sha256"] == m["sha256"]
                    and m["role"] == d["role"] and m["checkpoint_bytes_verified"] is True
                    and m["checkpoint_deserialized"] is False, "Output declaration/byte mismatch")
            claims = m["compatible_same_artifact_output_claims"]
            require(any(c["field_path"] == d["field_path"] and c["report_sha256"] == rec["report_sha256"]
                        and c["artifact_id"] == a["artifact_id"] for c in claims),
                    "Exact output field not found")
            grouped[d["sha256"]].append({"declaration": d, "run_id": rec["run_id"],
                "artifact_id": a["artifact_id"], "report_sha256": rec["report_sha256"],
                "archive_sha256": rec["archive_sha256"], "member": m["member"], "bytes": m["bytes"]})
    require(len(grouped) == 18 and sum(len(v) for v in grouped.values()) == 34
            and set(grouped)-set(known) == set(byte["unmapped_claims_matched_to_member_bytes"]),
            "Recovered output identity coverage mismatch")
    parents = {k: v["parent_checkpoint_sha256_declared"] for k,v in known.items()}
    nodes = []
    for digest, claims in sorted(grouped.items()):
        d = claims[0]["declaration"]
        require(all(c["declaration"] == d for c in claims), "Conflicting declarations for identical bytes")
        parent = d["parent_sha256"]
        if digest in parents:
            require(parents[digest] == parent and known[digest]["selected_or_last_step_declared"] == d["selected_step"],
                    "Previously mapped declaration conflict")
        parents[digest] = parent
        nodes.append({**d, "previously_mapped": digest in known,
                      "checkpoint_member_bytes_verified": True, "instances": [
                          {k:v for k,v in c.items() if k != "declaration"} for c in claims],
                      "parent_link_scope": "REPORT_DECLARATION_ONLY",
                      "actual_exposure_verified": False, "tensor_equality_verified": False})
    acyclic(parents)
    require(len(parents) == 34, "Merged checkpoint count mismatch")
    zero = [n for n in nodes if n["selected_step"] == 0]
    require(len(zero) == 6 and sum(len(n["instances"]) for n in zero) == 8,
            "Step-zero output coverage mismatch")
    return {"schema": "sagallm-recovered-semantic-lineage-v1",
            "status": "DECLARED_PARENT_AND_OUTPUT_ROLES_RECONCILED",
            "prior_checkpoint_count": len(known), "combined_checkpoint_hash_count": len(parents),
            "new_checkpoint_hash_count": len(set(grouped)-set(known)),
            "recovered_checkpoint_hash_count": len(nodes), "recovered_member_count": 34,
            "step_zero_distinct_hashes": len(zero), "step_zero_member_instances": 8,
            "positive_step_distinct_hashes": len(nodes)-len(zero),
            "prior_scope_checkpoint_declarations": list(known.values()),
            "recovered_outputs": nodes, "declared_parent_graph_acyclic": True,
            "parent_links": [{"checkpoint_sha256": h, "parent_sha256": p} for h,p in sorted(parents.items())],
            "unresolved_checkpoint_declarations": scope["unresolved_checkpoint_declarations"],
            "missing_source_declarations": scope["missing_source_declarations"] + [
                {"source": "identifier-train-candidates", "sha256_declared": IDENTIFIER,
                 "role": "TRIAL_TRAIN_INPUT_NOT_PROVEN_SELECTED_CHECKPOINT_EXPOSURE",
                 "source_bytes_verified_here": False, "rights_review": "REVIEW_REQUIRED"}],
            "inventory_entries": scope["inventory_entries"], "inventory_sha256": scope["inventory_sha256"],
            "new_inventory_entries": 0, "checkpoint_deserialized": False,
            "actual_exposure_verified": False, "training_lineage_verified": False,
            "full_exposure_inventory": "NOT_ESTABLISHED", "inventory_ready_for_independent_evaluation": False,
            "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("triage", "recovery", "byte"):
        p.add_argument("--"+name+"-zip", type=Path, required=True)
    a = p.parse_args()
    td, _ = read_evidence(a.triage_zip, TRIAGE_ZIP, ["unarchived-artifact-triage.json"])
    triage = td["unarchived-artifact-triage.json"]
    bd, _ = read_evidence(a.byte_zip, BYTE_ZIP, ["recovered-checkpoint-byte-scope.json"])
    manifest, blobs = read_recovery(a.recovery_zip, RECOVERY_ZIP)
    out = reconcile(validate(manifest, blobs, triage), triage["combined_scope"], bd["recovered-checkpoint-byte-scope.json"])
    out["byte_evidence"] = {"run_id": BYTE_RUN, "artifact_id": BYTE_ARTIFACT, "archive_sha256": BYTE_ZIP}
    print(json.dumps(out, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
