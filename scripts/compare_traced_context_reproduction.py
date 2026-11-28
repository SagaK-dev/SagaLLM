"""Recheck the original pilot and compare an exact-revision rerun.

A matching model state is not a claim of universal determinism or independence.
Serialized checkpoint identity, model tensors, optimizer tensors and trace
payloads are separate measurements.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import torch
from sagallm.exposure_trace import canonical, digest, file_sha, verify_trace, require

REV="a089929068b5b64ac69bfb3a7d916b1864b8a299"
EVIDENCE_ZIP="103201aae7eb8f6f1484c91c001a82dc43244d7cbaa9ab4ddcdd239eca7d056f"
WEIGHTS_ZIP="f027874de13707df48567bae374ad8278d968160cb8580299278ae06119c6773"
EVIDENCE={
 "optimizer-exposure.jsonl":(28631,"6a2ccd2622c4fb2c4df1b838ea1a22fc3ca344e98fd3b98d292b9e35b8cfec05"),
 "offline-audit.json":(1524,"9b329e24c0ce163e7c9faccd38512de780e1a9c5a4b3d384c083b1988dc91d3d"),
 "train-report.json":(5601,"0254485bb2c9ceee06227405689edd0476a7200deda6e8711789c4f06ae7c86e"),
 "dataset-manifest.json":(2586,"1e9c0ff933c5096cf2d633e81ec0c90498006fab45397a28a4c073fbf181a23a"),
 "pilot-summary.json":(1872,"481e4d324a0e57abe345be96661b54ec51ac953b57a08d334dfd77dfb00d7034"),
}
WEIGHTS={
 "java-sft-best.pt":(20718372,"729b0fbe346c38c819cb70d1373a4c1cce9fbbd8ce29eadf790ff233c9e73ccc"),
 "java-sft-last.pt":(20718372,"dacb86e7833e1857c1a55ce9724ee5a3406f28493f6681317cad8964c8288def"),
 "tokenizer.json":(57342,"186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe"),
}
MAX_CHECKPOINT=90*1024*1024

def unpack_pinned(path,zip_sha,size,members,dest):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size==size
            and file_sha(path)==zip_sha,"Original ZIP pin mismatch")
    dest.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as z:
        require(len(z.infolist())==len(members) and set(z.namelist())==set(members),"Unexpected ZIP members")
        for name,(n,h) in members.items():
            m=z.getinfo(name)
            require(m.file_size==n and not m.is_dir() and not m.flag_bits&1,"Member shape mismatch")
            with z.open(m) as f: raw=f.read(n+1)
            require(len(raw)==n and digest(raw)==h,"Original member mismatch")
            with (dest/name).open("xb") as f: f.write(raw)

def read_run(root):
    report=json.loads((root/"train-report.json").read_bytes())
    summary=json.loads((root/"pilot-summary.json").read_bytes())
    audit=json.loads((root/"offline-audit.json").read_bytes())
    trace=root/"optimizer-exposure.jsonl"
    require(summary["status"]=="PINNED_PARENT_TRACE_PILOT_COMPLETE" and summary["git_revision"]==REV
            and summary["steps"]==summary["optimizer_returns"]==12 and summary["row_exposures_rechecked"]==48
            and summary["seed"]==717 and summary["batch_size"]==4 and summary["lr"]==4e-5
            and summary["context"]==384 and summary["independent_evaluation"]=="NOT_RUN"
            and summary["production_adopted"] is False and summary["model_selection_permitted"] is False,
            "Pilot scope mismatch")
    require(report["optimizer_steps_completed"]==12 and report["test_used_for_model_selection"] is False
            and report["java_code_executed"] is False and report["compile_only_base"]["status"]=="NOT_RUN"
            and report["compile_only_best"]["status"]=="NOT_RUN"
            and file_sha(root/"train-report.json")==summary["report_sha256"]==audit["report_sha256"]
            and file_sha(root/"dataset-manifest.json")==summary["data_manifest_sha256"]==EVIDENCE["dataset-manifest.json"][1]
            and file_sha(root/"tokenizer.json")==summary["tokenizer_sha256"]==WEIGHTS["tokenizer.json"][1],
            "Pilot report identity mismatch")
    checked=verify_trace(trace,expected_sha256=summary["trace_sha256"],expected_tip=summary["tip_sha256"])
    require(checked==report["optimizer_exposure"]["trace"] and checked["status"]=="FINALIZED_LOCAL_TRACE"
            and checked["completed_optimizer_returns"]==12 and audit["steps_rechecked"]==12
            and audit["row_exposures_rechecked"]==48 and audit["trace_sha256"]==summary["trace_sha256"],
            "Trace/audit mismatch")
    for role,step in (("best",report["best_checkpoint_step"]),("last",12)):
        path=root/f"java-sft-{role}.pt"
        require(0<path.stat().st_size<=MAX_CHECKPOINT and file_sha(path)==summary[role+"_checkpoint_sha256"]
                ==report[role+"_checkpoint_sha256"]==checked["checkpoints"][role]["checkpoint_sha256"]
                and checked["checkpoints"][role]["completed_optimizer_returns"]==step,"Checkpoint binding mismatch")
    events=[json.loads(l) for l in trace.read_bytes().splitlines()]
    return report,summary,checked,events

def state_fingerprint(value):
    """Type-tagged recursive digest, including bit-exact tensor contents."""
    def encode(v):
        if isinstance(v,torch.Tensor):
            require(v.device.type=="cpu" and v.layout==torch.strided and not v.is_quantized
                    and v.numel()<=8_000_000,"Unsupported tensor")
            raw=v.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            return ["tensor",str(v.dtype),list(v.shape),digest(raw)]
        if isinstance(v,dict):
            require(all(type(k) in (str,int) for k in v),"Unexpected mapping key")
            return ["dict",[[type(k).__name__,k,encode(v[k])] for k in sorted(v,key=lambda k:(type(k).__name__,k))]]
        if isinstance(v,(list,tuple)):
            return [type(v).__name__,[encode(x) for x in v]]
        require(v is None or type(v) in (str,int,float,bool),"Unexpected state value")
        require(type(v) is not float or math.isfinite(v),"Nonfinite state scalar")
        return [type(v).__name__,v]
    return digest(canonical(encode(value)))

def compare_checkpoint(a,b,role,ar,br):
    out=[]
    for root,report in ((a,ar),(b,br)):
        p=root/f"java-sft-{role}.pt"
        require(p.is_file() and not p.is_symlink() and 0<p.stat().st_size<=MAX_CHECKPOINT
                and file_sha(p)==report[role+"_checkpoint_sha256"],"Checkpoint changed")
        require(torch.serialization.get_unsafe_globals_in_checkpoint(p)==[],"Unsupported checkpoint globals")
        state=torch.load(p,map_location="cpu",weights_only=True)
        require(type(state) is dict and isinstance(state["model"],dict),"Checkpoint schema mismatch")
        binding=report["optimizer_exposure"]["trace"]["checkpoints"][role]
        require(state["step"]==binding["completed_optimizer_returns"]
                and state["train_state"]["exposure_trace"]=={k:binding[k] for k in
                    ("schema","completed_optimizer_returns","prefix_sha256")},"Embedded trace metadata drift")
        opt_key="optimizer" if "optimizer" in state else "optimizer_state_dict"
        require(opt_key in state and isinstance(state[opt_key],dict),"Missing optimizer state")
        out.append({"file_sha256":file_sha(p),"step":state["step"],
                    "model_sha256":state_fingerprint(state["model"]),
                    "config_sha256":state_fingerprint(state["config"]),
                    "optimizer_sha256":state_fingerprint(state[opt_key])})
        del state
    return {"role":role,"original":out[0],"rerun":out[1],
            **{k+"_equal":out[0][k]==out[1][k] for k in out[0]}}

def compare(a,b):
    ar,asum,at,ae=read_run(a);br,bsum,bt,be=read_run(b)
    for k in ("parent_run_id","parent_artifact_id","parent_archive_sha256","parent_checkpoint_sha256","tokenizer_sha256",
              "data_manifest_sha256","steps","batch_size","seed","lr","context","git_revision","python_version","torch_version"):
        require(asum[k]==bsum[k],"Rerun fixed setting mismatch: "+k)
    intents=lambda events:[e["payload"] for e in events if e["kind"]=="step_intent"]
    require(len(intents(ae))==len(intents(be))==12,"Intent count mismatch")
    keys=("initial_validation_response_loss","best_validation_response_loss","heldout_test_response_loss_before_sft",
          "heldout_test_response_loss_best_sft","heldout_test_response_loss_last_sft")
    metrics={}
    for k in keys:
        x,y=ar[k],br[k]
        require(math.isfinite(x) and math.isfinite(y),"Nonfinite metric")
        metrics[k]={"original":x,"rerun":y,"absolute_difference":abs(x-y),"exact_equal":x==y}
    rows=[compare_checkpoint(a,b,r,ar,br) for r in ("best","last")]
    same_batches=intents(ae)==intents(be)
    same_model=all(r["model_sha256_equal"] and r["config_sha256_equal"] and r["step_equal"] for r in rows)
    same_opt=all(r["optimizer_sha256_equal"] for r in rows)
    return {"schema":"sagallm-traced-context-reproduction-v1","status":"BOUNDED_RERUN_COMPARISON_COMPLETE",
            "original_run_id":46104477318,"original_revision":REV,
            "original_evidence_artifact_id":15247550137,"original_checkpoint_artifact_id":15247155692,
            "original_zip_pins_verified":True,"steps_per_run":12,"row_exposures_per_run":48,
            "identical_ordered_batch_payloads":same_batches,"model_state_and_config_equal":same_model,
            "optimizer_state_equal":same_opt,"exact_serialized_checkpoint_equality":all(r["file_sha256_equal"] for r in rows),
            "trace_file_equal":at["trace_sha256"]==bt["trace_sha256"],
            "history_exact_equal":ar["history"]==br["history"],"metrics":metrics,"checkpoints":rows,
            "best_step_equal":ar["best_checkpoint_step"]==br["best_checkpoint_step"],
            "historical_full_environment_available":False,"universal_reproducibility_established":False,
            "independent_evaluation":"NOT_RUN","rights_review":"REVIEW_REQUIRED","model_selection_permitted":False}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("evidence-zip","weights-zip","original","rerun"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    unpack_pinned(a.evidence_zip,EVIDENCE_ZIP,16203,EVIDENCE,a.original)
    unpack_pinned(a.weights_zip,WEIGHTS_ZIP,38246766,WEIGHTS,a.original)
    print(json.dumps(compare(a.original,a.rerun),sort_keys=True,indent=2))
