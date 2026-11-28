"""Integrate source pins for reconstructed identifier trial inputs, not exposure."""
import argparse
import json
from pathlib import Path

from scripts.audit_ranking_auxiliary import pin, parse, sha, canonical, require
from scripts.audit_unarchived_artifact_triage import read_evidence, gate, hexsha
from scripts.audit_identifier_trial_inputs import REVISIONS, ORIGINALS, CANDIDATE, MANIFEST, DATA, TOKENIZER

LINEAGE_ARTIFACT=15246425930
LINEAGE_ZIP="49df23428f59e87b965bc6138e19c2269147446c5c3742858f9ef54313c3521b"
INVENTORY_ARTIFACT=15186012660
INVENTORY_ZIP="3553b57b58eadfea8ecd36ebedeb8ce72ad8b65136fcfc4ec1a7a58ec1c56cad"


def integrate(reports, lineage, inventory, inventory_hash):
    gate(lineage)
    require(lineage["status"]=="DECLARED_PARENT_AND_OUTPUT_ROLES_RECONCILED"
            and lineage["combined_checkpoint_hash_count"]==34
            and lineage["step_zero_distinct_hashes"]==6
            and lineage["step_zero_member_instances"]==8
            and lineage["inventory_entries"]==15 and lineage["inventory_sha256"]==inventory_hash,
            "Prior lineage inventory mismatch")
    require(inventory["schema"]=="sagallm-private-java-exposure-v1"
            and len(inventory["files"])==15 and len({r["path"] for r in inventory["files"]})==15,
            "Prior inventory scope")
    require(len(reports)==4 and {r["revision"] for r in reports}==set(REVISIONS),"Trial coverage mismatch")
    for r in reports:
        mode,run,artifact,digest=REVISIONS[r["revision"]]
        require(r["schema"]=="sagallm-identifier-trial-input-v1"
                and r["status"]=="RECONSTRUCTED_TRIAL_INPUT_SCOPE"
                and (r["mode"],r["run_id"],r["artifact_id"],r["report_sha256"])==(mode,run,artifact,digest)
                and r["source_archives"]=={str(k):v for k,v in ORIGINALS.items()}
                and r["candidate_sha256"]==CANDIDATE and r["candidate_bytes"]==102959
                and r["candidate_manifest_sha256"]==MANIFEST and r["paired_rows"]==120
                and r["train_families"]==10 and r["canonical_manifest_sha256"]==DATA
                and r["tokenizer_sha256"]==TOKENIZER and r["selected_step"]==0
                and r["source_bytes_verified_here"] is True
                and r["selected_checkpoint_exposure_certified"] is False
                and all(r[k] is False for k in ("model_loaded","new_training","new_inference",
                         "java_compiled","java_executed","actual_exposure_verified","training_lineage_verified",
                         "historical_schedule_digest_recorded","model_selection_permitted"))
                and hexsha(r["control_paired_sha256"]) and hexsha(r["derived_schedule_sha256"])
                and hexsha(r["derived_selection_ids_sha256"]), "Trial reconstruction gate mismatch")
    control={r["control_paired_sha256"] for r in reports}
    require(len(control)==1 and len({r["control_paired_bytes"] for r in reports})==1
            and len({r["derived_selection_ids_sha256"] for r in reports})==1,
            "Reconstructed source/selection drift")
    for mode in ("identifier","broad"):
        subset=[r for r in reports if r["mode"]==mode]
        require(len(subset)==2 and len({r["derived_schedule_sha256"] for r in subset})==1
                and subset[0]["trial_counts"]==subset[1]["trial_counts"], "Rerun schedule drift")
    new=[
        {"group":"train","path":"identifier/train-identifier-variants.jsonl","sha256":CANDIDATE,"format":"chat-jsonl"},
        {"group":"train","path":"identifier/control-paired.jsonl","sha256":next(iter(control)),"format":"chat-jsonl"},
    ]
    require(not {r["path"] for r in new}&{r["path"] for r in inventory["files"]},"Duplicate inventory path")
    combined={"schema":inventory["schema"],"files":inventory["files"]+new}
    raw=canonical(combined)
    out={"schema":"sagallm-identifier-trial-input-scope-v1",
         "status":"SCOPED_IDENTIFIER_TRIAL_INPUTS_RECONSTRUCTED",
         "inventory_sha256":sha(raw),"prior_inventory_sha256":inventory_hash,
         "inventory_entries":17,"new_inventory_entries":2,"new_sources":new,
         "source_archive_count":3,"trial_report_count":4,"trial_reconstructions":reports,
         "scope":"TRIAL_INPUT_SCREENING_ONLY_NOT_SELECTED_CHECKPOINT_EXPOSURE",
         "selected_checkpoint_exposure_certified":False,"prior_source_bytes_rechecked_here":False,
         "actual_exposure_verified":False,"training_lineage_verified":False,
         "full_exposure_inventory":"NOT_ESTABLISHED","inventory_ready_for_independent_evaluation":False,
         "rights_review":"REVIEW_REQUIRED","independent_evaluation":"NOT_RUN","model_selection_permitted":False}
    return raw,out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("lineage-zip","inventory-zip","reconstructions","inventory-output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    ld,_=read_evidence(a.lineage_zip,LINEAGE_ZIP,["recovered-semantic-lineage.json"])
    iv,hashes=read_evidence(a.inventory_zip,INVENTORY_ZIP,
                             ["scoped-parameter-ce-inventory.json","parameter-ce-evidence.json"])
    ev=iv["parameter-ce-evidence.json"]
    gate(ev)
    ih=hashes["scoped-parameter-ce-inventory.json"]
    require(ev["inventory_sha256"]==ih and ev["inventory_entries"]==15,"Prior inventory evidence mismatch")
    paths=sorted(a.reconstructions.glob("*.json"))
    require(len(paths)==4,"Unexpected reconstruction files")
    reports=[parse(path.read_bytes()) for path in paths]
    raw,out=integrate(reports,ld["recovered-semantic-lineage.json"],iv["scoped-parameter-ce-inventory.json"],ih)
    out["lineage_evidence"]={"artifact_id":LINEAGE_ARTIFACT,"archive_sha256":LINEAGE_ZIP}
    out["inventory_evidence"]={"artifact_id":INVENTORY_ARTIFACT,"archive_sha256":INVENTORY_ZIP}
    with a.inventory_output.open("xb") as f:
        f.write(raw)
    print(json.dumps(out,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
