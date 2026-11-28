"""Pre-registered validation-only canonical vs template-capped SFT comparison.

No arm adoption, TEST scoring, generated Java, or independent evaluation.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from unittest.mock import patch

import torch
from sagallm import java_instruction_sft as sft
from sagallm.java_diverse_corpus import build_java_diverse_corpus
from sagallm.exposure_trace import canonical, digest
from sagallm.java_template_subset import MODE, MEMBERSHIP, COUNT, build_template_subset
from scripts.audit_sft_prompt_collisions import audit as collision_audit
from scripts.run_pinned_traced_context_pilot import materialize_parent, DATA, PARENT_ARTIFACT, PARENT_ZIP
from scripts.verify_integrated_sft_trace import audit

SEEDS=(717,719,727)
ARMS=(("canonical",2.175e-5),(MODE,2.175e-5))
STEPS=48
def spec(arm,lr):
    return {"kind":"constant","peak_lr":lr,"warmup_steps":0,"min_lr_ratio":1.0}

BATCH=4
CONTEXT=384


def require(ok,msg):
    if not ok:raise ValueError(msg)


def scope_guard(report,proof):
    require(report["evaluation_scope"]=="VALIDATION_ONLY_DEVELOPMENT"
            and report["heldout_test_status"]=="NOT_RUN"
            and report["test_bytes_structurally_verified"] is True
            and report["original_split_counts"]["test"] is None
            and report["test_used_for_model_selection"] is False
            and all(report[k] is None for k in ("heldout_test_response_loss_before_sft",
                "heldout_test_response_loss_best_sft","heldout_test_response_loss_last_sft"))
            and report["compile_only_base"]["status"]==report["compile_only_best"]["status"]=="NOT_RUN"
            and report["java_code_executed"] is False
            and report["optimizer_steps_completed"]==proof["steps_rechecked"]==STEPS
            and proof["row_exposures_rechecked"]==STEPS*BATCH
            and proof["source_and_optimizer_declarations"]=="DECLARATIONS_CONSISTENT"
            and proof["learning_rate_schedule_status"]=="RECORDED_STEP_RATES_MATCH",
            "Validation-only acceptance gate mismatch")


def schedule_digest(path):
    events=[json.loads(x) for x in path.read_bytes().splitlines()]
    intents=[{k:v for k,v in x["payload"].items() if k!="optimizer_lr"} for x in events if x["kind"]=="step_intent"]
    require(len(intents)==STEPS and sum(len(i["rows"]) for i in intents)==STEPS*BATCH,
            "Unexpected schedule scope")
    return digest(canonical(intents))


def aggregate(rows):
    require(len(rows)==len(SEEDS)*len(ARMS),"Incomplete pre-registered sweep")
    require({(r["seed"],r["arm"]) for r in rows}=={(s,a) for s in SEEDS for a,_ in ARMS},
            "Duplicate or unexpected arm")
    for seed in SEEDS:
        subset=[r for r in rows if r["seed"]==seed]
        require(len({r["schedule_sha256"] for r in subset})==2
                and len({r["initial_validation_loss"] for r in subset})==1
                and len({r["validation_split_sha256"] for r in subset})==1,
                "Unexpected identical schedule or unmatched evaluation")
    table=[]
    for arm,lr in ARMS:
        subset=[r for r in rows if r["arm"]==arm]
        require(all(r["lr"]==lr and r["lr_schedule"]==spec(arm,lr) and r["test_status"]=="NOT_RUN" for r in subset),
                "Arm setting mismatch")
        table.append({"arm":arm,"lr":lr,"seed_count":len(subset),
            "mean_best_validation_loss":sum(r["best_validation_loss"] for r in subset)/len(subset),
            "mean_last_validation_loss":sum(r["last_validation_loss"] for r in subset)/len(subset),
            "mean_best_minus_baseline":sum(r["best_validation_loss"]-r["initial_validation_loss"] for r in subset)/len(subset),
            "positive_selected_step_count":sum(r["best_step"]>0 for r in subset)})
    return table


def run(parent_zip,output):
    output=Path(output)
    output.mkdir()
    rows=[]
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);materialize_parent(parent_zip,root)
        dataset=root/"data";build_java_diverse_corpus(dataset)
        require(sft.sha_file(dataset/"manifest.json")==DATA,"Canonical corpus pin mismatch")
        plan=collision_audit(dataset)["candidate_plan"]
        require(plan["membership_sha256"]==MEMBERSHIP,"Candidate plan drift")
        capped=root/"subset";subset=build_template_subset(dataset,capped,{r["row_sha256"] for r in plan["candidate_rows"]})
        full=json.loads((dataset/"manifest.json").read_bytes())
        require(all(full["files"][s]==subset["files"][s] for s in ("validation","test")),"Heldout manifest drift")
        base=root/"base.pt";tok=root/"tokenizer.json"
        original_read=sft.read_split
        for seed in SEEDS:
            for arm,lr in ARMS:
                dest=root/f"{seed}-{arm}"
                ds=dataset if arm=="canonical" else capped
                manifest=full if arm=="canonical" else subset
                split_calls=[]
                def guarded_read(root,split,*args,**kwargs):
                    split_calls.append(split)
                    require(split in ("train","validation"),"TEST tokenization forbidden")
                    return original_read(root,split,*args,**kwargs)
                with patch.object(sft,"read_split",side_effect=guarded_read), patch.object(
                    sft,"heldout_compilation",side_effect=AssertionError("Compile forbidden")):
                    report=sft.train_instruction_sft(base,tok,ds,dest,steps=STEPS,batch_size=BATCH,
                        eval_every=8,eval_count=30,test_count=30,compile_cases=0,lr=lr,seed=seed,
                        max_seconds=240,require_pinned=True,target_context=CONTEXT,
                        pinned_checkpoint_sha256=sft.V017_SFT_BEST_SHA256,
                        record_exposure=True,validation_only=True,lr_schedule=spec(arm,lr),dataset_mode=arm)
                require(split_calls==["train","validation"],"Unexpected training split access")
                exp=report["optimizer_exposure"]["trace"]
                proof=audit(dest,report_sha256=sft.sha_file(dest/"train-report.json"),
                    trace_sha256=exp["trace_sha256"],tip_sha256=exp["tip_sha256"],
                    dataset=ds,base=base,tokenizer=tok)
                scope_guard(report,proof)
                expected=17080 if arm=="canonical" else COUNT
                require(report["original_split_counts"]["train"]["accepted"]==expected and report["original_split_counts"]["train"]["total"]==expected,"Train admission mismatch")
                require(arm=="canonical" or proof["dataset_membership_status"]=="PINNED_TEMPLATE_SUBSET_MEMBERSHIP_VERIFIED","Subset audit missing")
                history=report["history"]
                require([r["step"] for r in history]==list(range(8,STEPS+1,8)),"Evaluation cadence mismatch")
                row={"seed":seed,"arm":arm,"lr":lr,"lr_schedule":spec(arm,lr),
                     "schedule_sha256":schedule_digest(dest/"optimizer-exposure.jsonl"),
                     "train_rows":expected,"train_manifest_sha256":report["data_manifest_sha256"],
                     "validation_split_sha256":manifest["files"]["validation"]["sha256"],
                     "distinct_train_rows_rechecked":proof["distinct_train_rows_rechecked"],
                     "initial_validation_loss":report["initial_validation_response_loss"],
                     "best_validation_loss":report["best_validation_response_loss"],
                     "last_validation_loss":history[-1]["validation_response_loss"],
                     "best_step":report["best_checkpoint_step"],"test_status":"NOT_RUN",
                     "report_sha256":proof["report_sha256"],"trace_sha256":proof["trace_sha256"],
                     "loss_targets_rechecked":proof["loss_targets_rechecked"],
                     "best_checkpoint_sha256":report["best_checkpoint_sha256"],
                     "last_checkpoint_sha256":report["last_checkpoint_sha256"],
                     "split_encoder_calls":split_calls}
                require(all(math.isfinite(row[k]) for k in ("initial_validation_loss","best_validation_loss","last_validation_loss")),
                        "Nonfinite validation metric")
                rows.append(row)
                evidence=output/f"{seed}-{arm}";evidence.mkdir()
                for name in ("train-report.json","optimizer-exposure.jsonl"):
                    (evidence/name).write_bytes((dest/name).read_bytes())
                (evidence/"offline-audit.json").write_text(json.dumps(proof,sort_keys=True,indent=2)+"\n")
                print(json.dumps(row,sort_keys=True),flush=True)
        (output/"dataset-manifest.json").write_bytes((dataset/"manifest.json").read_bytes())
        (output/"subset-manifest.json").write_bytes((capped/"manifest.json").read_bytes())
    table=aggregate(rows)
    pairs=[]
    for seed in SEEDS:
        a,b=[r for r in rows if r["seed"]==seed]
        pairs.append({"seed":seed,"subset_minus_full_best_loss":b["best_validation_loss"]-a["best_validation_loss"],"subset_minus_full_last_loss":b["last_validation_loss"]-a["last_validation_loss"]})
    result={"schema":"sagallm-template-capped-sft-comparison-v1","status":"TEMPLATE_CAPPED_SFT_COMPARISON_COMPLETE",
            "paired_differences":pairs,"subset_membership_sha256":MEMBERSHIP,"same_validation_bytes":True,
            "parent_artifact_id":PARENT_ARTIFACT,"parent_zip_sha256":PARENT_ZIP,"dataset_manifest_sha256":DATA,
            "seeds":list(SEEDS),"arms":[{"arm":a,"lr":lr,"lr_schedule":spec(a,lr)} for a,lr in ARMS],
            "steps_per_arm":STEPS,"batch_size":BATCH,"context":CONTEXT,
            "optimizer_updates":len(rows)*STEPS,"row_exposures_rechecked":len(rows)*STEPS*BATCH,
            "matched_schedule_within_seed":False,"equal_update_budget":True,"equal_target_token_budget":False,"runs":rows,"descriptive_validation_summary":table,
            "execution_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "python":sys.version.split()[0],"torch":str(torch.__version__),"platform":platform.platform(),
            "test_bytes_structurally_verified":True,"test_tokenization_or_scoring":False,
            "test_status":"NOT_RUN","java_compiled":False,"java_executed":False,
            "cross_arm_model_selection_performed":False,"checkpoint_binaries_published":False,
            "independent_evaluation":"NOT_RUN","rights_review":"REVIEW_REQUIRED","model_selection_permitted":False}
    (output/"subset-comparison-summary.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ("status","optimizer_updates","row_exposures_rechecked","descriptive_validation_summary","paired_differences")},sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-zip",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.parent_zip,a.output)
