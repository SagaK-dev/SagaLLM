"""Bounded prospective v0.17 -> 384 context pilot with integrated trace.

This is a new development run, not a historical rerun or model promotion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import zipfile

import torch
from sagallm.java_diverse_corpus import build_java_diverse_corpus
from sagallm.java_instruction_sft import (train_instruction_sft,sha_file,
    V017_SFT_BEST_SHA256,V016_TOKENIZER_SHA256)
from scripts.verify_integrated_sft_trace import audit

PARENT_ARTIFACT=14738368209
PARENT_RUN=44982940959
PARENT_ZIP="c25fa1833b64a620540d93067ce2c91a8c3916589dd5b565c8892661f9373581"
PARENT_SIZE=38138498
DATA="1e9c0ff933c5096cf2d633e81ec0c90498006fab45397a28a4c073fbf181a23a"
STEPS=12
BATCH=4
SEED=717
LR=4e-5
CONTEXT=384


def require(ok,msg):
    if not ok:
        raise ValueError(msg)


def materialize_parent(archive,root):
    p=Path(archive)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size==PARENT_SIZE
            and sha_file(p)==PARENT_ZIP,"Parent archive pin mismatch")
    with zipfile.ZipFile(p) as z:
        entries=z.infolist()
        require(0<len(entries)<=64 and len({m.filename for m in entries})==len(entries),
                "Unsafe parent ZIP directory")
        for name,dest,digest,limit in (
            ("java-sft-model/java-sft-best.pt","base.pt",V017_SFT_BEST_SHA256,90*1024*1024),
            ("java-sft-model/tokenizer.json","tokenizer.json",V016_TOKENIZER_SHA256,8*1024*1024)):
            m=z.getinfo(name)
            require(not m.is_dir() and not m.flag_bits&1 and 0<m.file_size<=limit,
                    "Unsafe parent member")
            with z.open(m) as f:
                raw=f.read(limit+1)
            require(len(raw)==m.file_size and hashlib.sha256(raw).hexdigest()==digest,
                    "Parent member digest mismatch")
            with (root/dest).open("xb") as f:
                f.write(raw)


def run(archive,output):
    output=Path(output)
    require(not output.exists() and not output.is_symlink(),"Refuse overwrite")
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        materialize_parent(archive,root)
        dataset=root/"data";build_java_diverse_corpus(dataset)
        require(sha_file(dataset/"manifest.json")==DATA,"Canonical corpus pin mismatch")
        base=root/"base.pt";tok=root/"tokenizer.json"
        report=train_instruction_sft(base,tok,dataset,output,steps=STEPS,batch_size=BATCH,
            eval_every=4,eval_count=30,test_count=30,compile_cases=0,lr=LR,seed=SEED,
            max_seconds=180,require_pinned=True,target_context=CONTEXT,
            pinned_checkpoint_sha256=V017_SFT_BEST_SHA256,record_exposure=True)
        exp=report["optimizer_exposure"]
        proof=audit(output,report_sha256=sha_file(output/"train-report.json"),
            trace_sha256=exp["trace"]["trace_sha256"],tip_sha256=exp["trace"]["tip_sha256"],
            dataset=dataset,base=base,tokenizer=tok)
        require(report["optimizer_steps_completed"]==STEPS and report["termination"]=="STEP_LIMIT"
                and proof["steps_rechecked"]==STEPS and proof["row_exposures_rechecked"]==STEPS*BATCH
                and report["model_context_length"]==CONTEXT and report["compile_only_base"]["status"]=="NOT_RUN"
                and report["compile_only_best"]["status"]=="NOT_RUN","Pilot acceptance scope mismatch")
        (output/"dataset-manifest.json").write_bytes((dataset/"manifest.json").read_bytes())
        (output/"offline-audit.json").write_text(json.dumps(proof,sort_keys=True,indent=2)+"\n")
        summary={"schema":"sagallm-prospective-traced-context-pilot-v1",
            "status":"PINNED_PARENT_TRACE_PILOT_COMPLETE","parent_run_id":PARENT_RUN,
            "parent_artifact_id":PARENT_ARTIFACT,"parent_archive_sha256":PARENT_ZIP,
            "parent_checkpoint_sha256":V017_SFT_BEST_SHA256,"tokenizer_sha256":V016_TOKENIZER_SHA256,
            "data_manifest_sha256":DATA,"steps":STEPS,"batch_size":BATCH,"seed":SEED,"lr":LR,
            "context":CONTEXT,"optimizer_returns":proof["steps_rechecked"],
            "row_exposures_rechecked":proof["row_exposures_rechecked"],
            "best_step":report["best_checkpoint_step"],
            "initial_validation_loss":report["initial_validation_response_loss"],
            "best_validation_loss":report["best_validation_response_loss"],
            "heldout_base_loss":report["heldout_test_response_loss_before_sft"],
            "heldout_best_loss":report["heldout_test_response_loss_best_sft"],
            "heldout_last_loss":report["heldout_test_response_loss_last_sft"],
            "report_sha256":proof["report_sha256"],"trace_sha256":proof["trace_sha256"],
            "tip_sha256":proof["tip_sha256"],"best_checkpoint_sha256":report["best_checkpoint_sha256"],
            "last_checkpoint_sha256":report["last_checkpoint_sha256"],
            "git_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
            "python_version":sys.version.split()[0],"torch_version":str(torch.__version__),
            "platform":platform.platform(),"historical_reproduction":False,"new_training":True,
            "java_compiled":False,"java_executed":False,"independent_evaluation":"NOT_RUN",
            "model_selection_permitted":False,"production_adopted":False,
            "historical_exposure_verified":False,"rights_review":"REVIEW_REQUIRED"}
        (output/"pilot-summary.json").write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
        print(json.dumps(summary,sort_keys=True),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-zip",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();run(a.parent_zip,a.output)
