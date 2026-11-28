"""Offline audit of a completed opt-in Java SFT run.

Requires caller-supplied report, trace and tip SHA-256 pins. Rebuilds the
deterministic TRAIN schedule, verifies saved checkpoint bytes, and checks
reported selection history. Does not deserialize checkpoints or run a model.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from sagallm.exposure_trace import MAX_BYTES, batch_record, digest, is_sha, require, verify_trace, _unique
from sagallm.exposure_source_binding import bounded, replay_source_binding, MAX_CHECKPOINT_BYTES
from sagallm.java_instruction_sft import read_split, interleaved_order, pack, PROMPT_VERSION
from sagallm.tokenization import load_tokenizer

MAX_REPORT_BYTES=2*1024*1024


def declared_selection(report):
    """Validate selection arithmetic, not the measured loss values."""
    history=report["history"]
    completed=report["optimizer_steps_completed"]
    best=report["initial_validation_response_loss"]
    require(type(best) in (int,float) and math.isfinite(best) and best>=0,
            "Invalid initial validation loss")
    step=0; seen=0; previous=0
    for row in history:
        s=row["step"]; loss=row["validation_response_loss"]
        require(type(s) is int and previous<s<=completed and type(loss) in (int,float)
                and math.isfinite(loss) and loss>=0,"Invalid selection history")
        if loss<best-0.001:
            best=loss;step=s;seen=row["distinct_train_examples_seen"]
        previous=s
    require(step==report["best_checkpoint_step"] and best==report["best_validation_response_loss"]
            and seen==report["best_checkpoint_train_examples_seen"],"Declared best selection drift")
    return step


def audit(output,*,report_sha256,trace_sha256,tip_sha256,dataset,base,tokenizer):
    require(all(is_sha(x) for x in (report_sha256,trace_sha256,tip_sha256)),"External digest pins required")
    root=Path(output)
    require(root.is_dir() and not root.is_symlink(),"Unsafe output directory")
    report=json.loads(bounded(root/"train-report.json",report_sha256,MAX_REPORT_BYTES),object_pairs_hook=_unique)
    require(report["status"]=="TRAINED_CPU" and report["phase"]=="JAVA_INSTRUCTION_SFT"
            and report["prompt_version"]==PROMPT_VERSION and report["test_used_for_model_selection"] is False
            and report["java_code_executed"] is False and report["java_behavior_test"]=="NOT_RUN"
            and report["java_coding_ability"]=="NOT_ESTABLISHED","SFT report gate mismatch")
    exp=report["optimizer_exposure"]
    require(exp["status"]=="LOCAL_TRACE_AND_TRAIN_SOURCE_VERIFIED"
            and exp["journal_file"]=="optimizer-exposure.jsonl"
            and exp["independent_attestation"] is False and exp["historical_exposure_verified"] is False,
            "Unexpected trace declaration")
    journal=root/"optimizer-exposure.jsonl"
    v=verify_trace(journal,expected_sha256=trace_sha256,expected_tip=tip_sha256)
    require(v==exp["trace"] and v["status"]=="FINALIZED_LOCAL_TRACE","Report trace mismatch")
    events=[json.loads(x,object_pairs_hook=_unique) for x in
            bounded(journal,trace_sha256,MAX_BYTES).splitlines()]
    h=events[0]["payload"]
    completed=v["completed_optimizer_returns"]
    require(completed==report["optimizer_steps_completed"]<=report["optimizer_steps_requested"]
            ==h["max_steps"] and report["termination"] in ("STEP_LIMIT","TIME_BUDGET")
            and (report["termination"]!="STEP_LIMIT" or completed==h["max_steps"]),
            "Completed step/termination mismatch")
    require(h["base_checkpoint_sha256"]==report["checkpoint_sha256_base"]
            and h["tokenizer_sha256"]==report["tokenizer_sha256"]
            and h["data_manifest_sha256"]==report["data_manifest_sha256"],"Report input pins mismatch")
    context=report["model_context_length"]
    replay=replay_source_binding(journal,expected_trace_sha256=trace_sha256,expected_tip=tip_sha256,
        dataset_dir=dataset,tokenizer_path=tokenizer,base_checkpoint=base,context=context)
    require(replay==exp["source_binding"],"Source replay report mismatch")
    # Compare the preserved tokenizer as well as caller-supplied input tokenizer.
    bounded(root/"tokenizer.json",h["tokenizer_sha256"],8*1024*1024)
    original=report["original_model_context_length"]
    require(type(original) is int and 1<=original<=context,"Invalid original context")
    tok=load_tokenizer(tokenizer)
    rows,counts=read_split(Path(dataset),"train",tok,context,original_context=original)
    require(counts==report["original_split_counts"]["train"] and len(rows)>=h["batch_size"],
            "TRAIN admission count mismatch")
    order=interleaved_order(rows,h["seed"]);seen=set();step=0;best_seen=0;targets=0
    best_step=declared_selection(report)
    for event in events:
        if event["kind"]!="step_intent":
            continue
        pay=event["payload"];step+=1
        indices=[order[((step-1)*h["batch_size"]+j)%len(order)] for j in range(h["batch_size"])]
        chosen=[rows[i] for i in indices];x,y=pack(chosen)
        record=batch_record(chosen,x,y)
        require(pay["step"]==step and record=={k:z for k,z in pay.items() if k!="step"},
                "Deterministic batch schedule mismatch")
        seen.update(indices);targets+=record["loss_targets"]
        if step==best_step:
            best_seen=len(seen)
    require(step==completed and len(seen)==report["distinct_training_examples_seen"]
            and best_seen==report["best_checkpoint_train_examples_seen"]
            and targets==replay["loss_targets_rechecked"],"Schedule totals mismatch")
    checkpoint_rows=[]
    for role,expected_step in (("best",best_step),("last",completed)):
        binding=v["checkpoints"][role]
        expected=report[role+"_checkpoint_sha256"]
        require(binding["checkpoint_sha256"]==expected
                and binding["completed_optimizer_returns"]==expected_step,"Checkpoint binding mismatch")
        raw=bounded(root/f"java-sft-{role}.pt",expected,MAX_CHECKPOINT_BYTES)
        checkpoint_rows.append({"role":role,"sha256":expected,"bytes":len(raw),
                                "step_declared":expected_step,"journal_prefix_sha256":binding["prefix_sha256"]})
        del raw
    return {"schema":"sagallm-integrated-sft-offline-audit-v1",
            "status":"PINNED_REPORT_TRACE_SCHEDULE_AND_CHECKPOINT_BYTES_MATCH",
            "report_sha256":report_sha256,"trace_sha256":trace_sha256,"tip_sha256":tip_sha256,
            "steps_rechecked":step,"row_exposures_rechecked":step*h["batch_size"],
            "distinct_train_rows_rechecked":len(seen),"loss_targets_rechecked":targets,
            "best_step_declared":best_step,"checkpoint_bindings":checkpoint_rows,
            "selection_arithmetic_rechecked":True,"validation_losses_recomputed":False,
            "checkpoint_deserialized":False,"embedded_metadata_independently_read":False,
            "validation_or_test_rows_read":False,"actual_training_process_attested":False,
            "historical_exposure_verified":False,"gradient_influence_verified":False,
            "independent_evaluation":"NOT_RUN","model_selection_permitted":False}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("output","dataset","base","tokenizer"):
        p.add_argument("--"+k,type=Path,required=True)
    for k in ("report-sha256","trace-sha256","tip-sha256"):
        p.add_argument("--"+k,required=True)
    print(json.dumps(audit(**vars(p.parse_args())),sort_keys=True,indent=2))
