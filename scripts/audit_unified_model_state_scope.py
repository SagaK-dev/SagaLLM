"""Integrate pinned historical declarations, tensor aliases and development inputs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.audit_unarchived_artifact_triage import read_evidence, gate, hexsha
from scripts.audit_ranking_auxiliary import pin, parse, sha, canonical, require
from scripts.audit_identifier_trial_scope import LINEAGE_ZIP, REVISIONS
from scripts.audit_v018_training import BEST, BASE, TOKENIZER
from scripts.audit_legacy_run_discovery import PUBLIC_SUITE

TENSOR_ZIP="33fbe94779b9a4b6ae43c29a1756f83127c0370f8392081095da37bdef40de7a"
TRIAL_ZIP="afe8fba96ac0764333007f96eb74565a6bd65cad02d268b288e4d46aefd41e98"
ORIGINAL_MANIFEST="8d5181f65d442a3aa283c3de606fc51fe2e73c0eae1a4005e6d63b8662d2a251"
PROMPTS_BLOB="39c821286244c16024a1506a4ebf2b3792ebe003"
PUBLIC_REV="0d1fbd6fb22dc6b9b272a1ed6f886e9338ffcb6e"
ORIGINAL_REV="04b16fbb3a4901123d2891827b38beb6f492b4f4"
PUBLIC_REPORTS={
 "45159337002-14809802712-0.json":"675faedd65720d714d94358945da597acd2f301f32742e74cb9a0383c9fb11f0",
 "45159800650-14809758426-0.json":"675faedd65720d714d94358945da597acd2f301f32742e74cb9a0383c9fb11f0",
 "45161286649-14809854631-0.json":"247e5899ea2b3f58293bccab648364f8a1fd3363c7c1003383d65a674f987d51",
 "45161722607-14812505339-0.json":"a0b1e779869c332a1c7b343ec870823800da7b9106f341e28d85b3a6529c0653",
}


def public_scope(suite, script, reports):
    require(sha(suite)==PUBLIC_SUITE, "Public suite digest mismatch")
    require(hashlib.sha1(b"blob "+str(len(script)).encode()+b"\0"+script).hexdigest()==PROMPTS_BLOB,
            "Historical script blob mismatch")
    tree=ast.parse(script.decode("utf8"))
    assignments=[n for n in tree.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=="PROMPTS" for t in n.targets)]
    require(len(assignments)==1,"Historical prompt assignment mismatch")
    prompts=ast.literal_eval(assignments[0].value)
    d=parse(suite)
    require(d["origin"]=="EXTERNAL_SOURCE_INTAKE_UNREVIEWED" and d["review_status"]=="PENDING_HUMAN_REVIEW"
            and d["license_id"]==d["train_overlap_review"]=="NOT_REVIEWED" and d["reviewers"]==[],
            "Public review gate drift")
    tasks=d["tasks"]
    ids=[t["id"] for t in tasks]
    require(len(ids)==len(set(ids))==6 and ids==list(prompts)
            and sum(len(t["cases"]) for t in tasks)==27,"Public suite coverage mismatch")
    rows=[]
    for t in tasks:
        require(set(t)=={"id","family","prompt","cases"},"Unexpected public task")
        def add(field,value):
            require(isinstance(value,str),"Nontext public field")
            rows.append({"id":t["id"],"family":t["family"],"field":field,"text":value})
        add("intake_prompt",t["prompt"])
        add("historical_abridged_task",prompts[t["id"]])
        for i,c in enumerate(t["cases"]):
            require(set(c)=={"stdin","stdout"},"Public case shape")
            for k in ("stdin","stdout"):
                add(f"case_{i}_{k}",c[k])
    require(len(rows)==66,"Public field count")
    require(set(reports)==set(PUBLIC_REPORTS),"Public report coverage")
    data={}
    for name,raw in reports.items():
        require(sha(raw)==PUBLIC_REPORTS[name],"Public report pin mismatch")
        r=parse(raw)
        require(r["suite_sha256"]==PUBLIC_SUITE and r["tokenizer_sha256"]==TOKENIZER
                and r["v017_checkpoint_sha256"]==BASE and r["v018_checkpoint_sha256"]==BEST
                and r["model_selection_permitted"] is False,"Public report gate")
        data[name]=r
    compiles=[r for r in data.values() if r["status"]=="EXPLORATORY_PUBLIC_COMPILE_ONLY_UNREVIEWED"]
    behavior=[r for r in data.values() if r["status"]=="EXPLORATORY_PUBLIC_UNREVIEWED_ISOLATED_IO"]
    require(len(compiles)==len(behavior)==2 and compiles[0]==compiles[1],"Public run scope")
    comp=compiles[0]
    require(comp["human_reviews"]==0 and comp["source_license_review"]=="PENDING"
            and comp["behavioral_status"]=="NOT_RUN" and comp["java_executed"] is False,"Compile scope")
    for model in ("v017","v018"):
        c=comp[model]
        require([x["id"] for x in c["cases"]]==ids and c["compiled"]==2 and c["failed"]==4
                and c["evaluated"]==6 and c["unavailable_or_skipped"]==0,"Compile totals")
        require(all(x["prompt_sha256"]==sha(prompts[x["id"]].encode()) and x["status"] in ("PASS","FAIL")
                    and x["java_executed"] is False and x["source_saved"] is False
                    for x in c["cases"]) and sum(x["status"]=="PASS" for x in c["cases"])==2,
                "Historical abridged prompt mismatch")
        compiled={x["id"]:x["output_sha256"] for x in c["cases"] if x["status"]=="PASS"}
        for b in behavior:
            require(b["independent_benchmark_approval"] is False and b["model_updated"] is False
                    and b["human_review_status"]=="REVIEW_REQUIRED" and b["original_27_cases"]==27
                    and b["attempted_cases_across_models"]==19,"Behavior scope")
            m=b[model]
            require({x["id"]:x["source_sha256"] for x in m["tasks"]}==compiled
                    and m["io_cases_passed"]==0 and m["io_cases_attempted"]==sum(x["cases_total"] for x in m["tasks"])
                    and all(x["cases_total"]==len(next(t["cases"] for t in tasks if t["id"]==x["id"]))
                            ==len(x["case_statuses"]) and set(x["case_statuses"])=={"FAIL"}
                            and x["host_java_execution"] is False for x in m["tasks"]),
                    "Behavior/compile/source binding mismatch")
    raw=b"".join(canonical(r) for r in rows)
    return {"group":"development","path":"derived/public_unreviewed_io.jsonl","sha256":sha(raw),"format":"text-jsonl"}, {
        "source_sha256":PUBLIC_SUITE,"prompt_script_git_blob":PROMPTS_BLOB,"source_revision":PUBLIC_REV,
        "task_count":6,"oracle_cases":27,"screening_fields":66,
        "historical_compile_instances":24,"historical_behavior_case_instances":38,
        "distinct_compile_report_digests":1,"distinct_behavior_report_digests":2,
        "source_bytes_verified_here":True,"oracle_correctness_independently_verified":False,
        "source_rights_verified":False,"actual_model_inputs_replayed":False,
        "independent_evaluation":"NOT_RUN","rights_review":"REVIEW_REQUIRED",
        "provenance_url":d["provenance_url"]}


def unify(lineage,tensor,trial,inventory,inv_sha,public_entry,public):
    for x in (lineage,tensor,trial):
        gate(x)
    require(lineage["combined_checkpoint_hash_count"]==34 and lineage["step_zero_distinct_hashes"]==6
            and lineage["declared_parent_graph_acyclic"] is True,"Lineage coverage")
    require(tensor["lineage_evidence"]["archive_sha256"]==LINEAGE_ZIP
            and tensor["parent_checkpoint_sha256"]==BEST and tensor["member_count"]==8
            and tensor["all_selected_step_zero_tensors_match_parent"] is True
            and tensor["tensor_and_config_equal_member_count"]==8
            and tensor["tensor_and_config_equal_distinct_hashes"]==6,"Tensor scope")
    expected={(i["artifact_id"],i["member"]):(n["sha256"],n["arm"],i["run_id"])
              for n in lineage["recovered_outputs"] if n["selected_step"]==0 for i in n["instances"]}
    rows=tensor["results"]
    require(len(expected)==len(rows)==len({(r["artifact_id"],r["member"]) for r in rows})==8,
            "Tensor member coverage")
    for r in rows:
        require(expected[(r["artifact_id"],r["member"])]==(r["checkpoint_sha256"],r["arm"],r["run_id"])
                and r["embedded_parent_sha256"]==BEST and r["embedded_selected_step"]==0
                and r["config_equal"] is True and r["all_tensor_digests_match"] is True
                and r["different_tensor_names"]==[]
                and r["parent_tensor_index_sha256"]==r["candidate_tensor_index_sha256"]
                and r["parent_tensor_count"]==r["candidate_tensor_count"]
                and r["parent_tensor_bytes"]==r["candidate_tensor_bytes"],"Tensor result binding")
    require(len({r["parent_tensor_index_sha256"] for r in rows})==1,"Parent tensor index drift")
    aliases={r["checkpoint_sha256"]:BEST for r in rows}
    all_hashes={r["checkpoint_sha256"] for r in lineage["parent_links"]}
    require(len(aliases)==6 and set(aliases)<=all_hashes and BEST in all_hashes
            and BEST not in aliases and len(all_hashes)==34,"Alias graph scope")
    require(trial["lineage_evidence"]["archive_sha256"]==LINEAGE_ZIP
            and trial["prior_inventory_sha256"]==lineage["inventory_sha256"]
            and trial["inventory_sha256"]==inv_sha and trial["inventory_entries"]==17
            and trial["selected_checkpoint_exposure_certified"] is False
            and len(inventory["files"])==len({r["path"] for r in inventory["files"]})==17
            and inventory["files"][-2:]==trial["new_sources"],"Trial inventory binding")
    require(public_entry["path"] not in {r["path"] for r in inventory["files"]},"Duplicate public path")
    merged={"schema":inventory["schema"],"files":inventory["files"]+[public_entry]}
    return merged,{"schema":"sagallm-unified-model-state-scope-v1",
        "status":"SCOPED_MODEL_STATE_AND_DEVELOPMENT_INPUTS_INTEGRATED",
        "serialized_checkpoint_hash_count":34,
        "known_tensor_alias_count":6,
        "model_state_equivalence_classes_upper_bound":28,
        "exact_distinct_model_state_count":"NOT_ESTABLISHED",
        "known_aliases":[{"checkpoint_sha256":h,"equivalent_parent_sha256":p,
                           "scope":"MODEL_TENSORS_AND_CONFIG_ONLY"} for h,p in sorted(aliases.items())],
        "declared_parent_links":lineage["parent_links"],
        "unresolved_checkpoint_declarations":lineage["unresolved_checkpoint_declarations"],
        "inventory_entries":18,"inventory_sha256":sha(canonical(merged)),
        "public_development_scope":public,"trial_source_scope":trial["scope"],
        "trial_sources_not_selected_checkpoint_updates":True,
        "prior_source_bytes_rechecked_here":False,"actual_exposure_verified":False,
        "training_lineage_verified":False,"full_exposure_inventory":"NOT_ESTABLISHED",
        "inventory_ready_for_independent_evaluation":False,"rights_review":"REVIEW_REQUIRED",
        "independent_evaluation":"NOT_RUN","model_selection_permitted":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("lineage-zip","tensor-zip","trial-zip","originals","suite","prompt-script","inventory-output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    ld,_=read_evidence(a.lineage_zip,LINEAGE_ZIP,["recovered-semantic-lineage.json"])
    td,_=read_evidence(a.tensor_zip,TENSOR_ZIP,["step-zero-tensor-equivalence.json"])
    names=["identifier-trial-input-scope.json","scoped-identifier-inventory.json"]+[
        f"identifier-trial-reconstructions/{r}.json" for r in REVISIONS]
    iv,hashes=read_evidence(a.trial_zip,TRIAL_ZIP,names)
    trial=iv["identifier-trial-input-scope.json"]
    require({r["revision"]:r for r in trial["trial_reconstructions"]}==
            {r:iv[f"identifier-trial-reconstructions/{r}.json"] for r in REVISIONS},"Trial report mismatch")
    m=parse(pin(a.originals/"recovery-manifest.json",ORIGINAL_MANIFEST))
    members={x["preserved_file"]:x for r in m["artifacts"] for x in r["report_members"] if "preserved_file" in x}
    require(len(members)==35,"Durable original manifest coverage")
    reports={name:pin(a.originals/"reports"/name,digest) for name,digest in PUBLIC_REPORTS.items()}
    require(all(members[n]["report_sha256"]==sha(v) for n,v in reports.items()),"Durable report binding")
    require(a.prompt_script.stat().st_size<=200000,"Prompt source too large")
    e,public=public_scope(pin(a.suite,PUBLIC_SUITE),a.prompt_script.read_bytes(),reports)
    merged,out=unify(ld["recovered-semantic-lineage.json"],td["step-zero-tensor-equivalence.json"],
                     trial,iv["scoped-identifier-inventory.json"],hashes["scoped-identifier-inventory.json"],e,public)
    out["durable_original_manifest_sha256"]=ORIGINAL_MANIFEST
    with a.inventory_output.open("xb") as f:
        f.write(canonical(merged))
    print(json.dumps(out,sort_keys=True,indent=2))


if __name__=="__main__":
    main()
