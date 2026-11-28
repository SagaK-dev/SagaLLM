"""Cross-check old and new validation sweeps without loading model weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

from sagallm.sft_lr_schedule import step_lr

ZIP="19ad506f4e9f9b3b3654d0c44081638e2296fef5a6623c507f0c852f9b081ada"
OLD_ZIP="d9d5528c4068c65b9440c09e00d9ac1da6b688c2c6ca8490c6d40bfe45d46fd4"
NEW_ARMS=("constant_low","constant_reference","warmup_cosine")
SEEDS=(717,719)

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()

def load(path,size,pin,arms,summary):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size==size
            and sha(path.read_bytes())==pin,"Sweep ZIP mismatch")
    expected={"dataset-manifest.json","environment-freeze.txt",summary}
    expected|={f"{seed}-{arm}/{name}" for seed in SEEDS for arm in arms for name in
               ("train-report.json","optimizer-exposure.jsonl","offline-audit.json")}
    with zipfile.ZipFile(path) as z:
        require(len(z.infolist())==len(expected) and set(z.namelist())==expected,"Sweep member coverage")
        raw={}
        for n in expected:
            i=z.getinfo(n)
            require(not i.is_dir() and not i.flag_bits&1 and 0<i.file_size<=100000,"Unsafe member")
            raw[n]=z.read(n)
    return raw

def inspect(raw,summary,steps,arms):
    d=json.loads(raw[summary]);result={}
    require(d["optimizer_updates"]==6*steps and d["row_exposures_rechecked"]==24*steps
            and d["test_status"]=="NOT_RUN" and d["cross_arm_model_selection_performed"] is False
            and d["dataset_manifest_sha256"]==sha(raw["dataset-manifest.json"]),"Sweep scope mismatch")
    for row in d["runs"]:
        key=(row["seed"],row["arm"]);p=f"{key[0]}-{key[1]}/"
        require(key not in result and key[0] in SEEDS and key[1] in arms,"Unexpected arm")
        r=json.loads(raw[p+"train-report.json"]);a=json.loads(raw[p+"offline-audit.json"])
        require(row["report_sha256"]==a["report_sha256"]==sha(raw[p+"train-report.json"])
                and row["trace_sha256"]==a["trace_sha256"]==sha(raw[p+"optimizer-exposure.jsonl"])
                and r["optimizer_steps_completed"]==a["steps_rechecked"]==steps
                and a["row_exposures_rechecked"]==4*steps and r["heldout_test_status"]=="NOT_RUN"
                and r["optimizer_spec"]["lr"]==row["lr"] and r["optimizer_spec"]["seed"]==key[0]
                and r["optimizer_spec"]["batch_size"]==4
                and r["best_validation_response_loss"]==row["best_validation_loss"],"Arm scope mismatch")
        tip="0"*64;events=[json.loads(x) for x in raw[p+"optimizer-exposure.jsonl"].splitlines()]
        for n,e in enumerate(events):
            body={k:v for k,v in e.items() if k!="event_sha256"}
            require(e["seq"]==n and e["prev_sha256"]==tip and sha(canonical(body))==e["event_sha256"],"Trace chain mismatch")
            tip=e["event_sha256"]
        h=events[0]["payload"]
        require(tip==a["tip_sha256"] and events[-1]["kind"]=="final"
                and events[-1]["payload"]["completed_optimizer_returns"]==steps and h["max_steps"]==steps
                and h["seed"]==key[0] and h["batch_size"]==4,"Trace finalization")
        intents=[e["payload"] for e in events if e["kind"]=="step_intent"]
        returned=[e["payload"]["step"] for e in events if e["kind"]=="optimizer_returned"]
        require([i["step"] for i in intents]==returned==list(range(1,steps+1)),"Step coverage")
        batches=[{k:v for k,v in i.items() if k!="optimizer_lr"} for i in intents]
        require(sha(canonical(batches))==row["schedule_sha256"] and sum(len(i["rows"]) for i in batches)==steps*4,
                "Input schedule mismatch")
        if steps==24:
            spec=h["lr_schedule"]
            require(spec==r["optimizer_spec"]["lr_schedule"]==row["lr_schedule"]
                    and a["learning_rate_schedule_status"]=="RECORDED_STEP_RATES_MATCH","Recorded schedule mismatch")
            rates=[i["optimizer_lr"] for i in intents]
            require(rates==[step_lr(spec,steps,s) for s in returned],"Recorded rate mismatch")
        else:
            require("lr_schedule" not in h and all("optimizer_lr" not in i for i in intents),"Unexpected legacy schedule")
            rates=None
        result[key]={"history":r["history"],"batches":batches,"rates":rates,"row":row,"header":h}
    require(set(result)=={(s,a) for s in SEEDS for a in arms},"Incomplete arm coverage")
    return result

def compare(old,new):
    pairs=[];rates=[]
    for seed in SEEDS:
        for a,b in (("low","constant_low"),("reference","constant_reference")):
            x,y=old[seed,a],new[seed,b]
            require(all(x["header"][k]==y["header"][k] for k in ("base_checkpoint_sha256","tokenizer_sha256",
                       "data_manifest_sha256","train_split_sha256","seed","batch_size")),"Parent/input identity mismatch")
            diffs=[]
            for l,r in zip(x["history"],y["history"]):
                require(l["step"]==r["step"],"Evaluation step mismatch")
                diffs.append({"step":l["step"],"train_absolute_difference":abs(l["last_train_response_loss"]-r["last_train_response_loss"]),
                              "validation_absolute_difference":abs(l["validation_response_loss"]-r["validation_response_loss"])})
            require(len(diffs)==3,"Prefix eval coverage")
            pairs.append({"seed":seed,"old_arm":a,"new_arm":b,"first_12_input_batches_equal":x["batches"]==y["batches"][:12],
                          "first_12_history_exact_equal":x["history"]==y["history"][:3],"differences":diffs})
        for arm in NEW_ARMS:
            row=new[seed,arm];lr=row["rates"]
            rates.append({"seed":seed,"arm":arm,"sum_step_rates":math.fsum(lr),"mean_step_rate":math.fsum(lr)/len(lr),
                          "best_step":row["row"]["best_step"],"best_validation_loss":row["row"]["best_validation_loss"],
                          "last_validation_loss":row["row"]["last_validation_loss"]})
    return {"schema":"sagallm-sweep-prefix-comparison-v1","status":"PINNED_SWEEP_PREFIX_COMPARISON_COMPLETE",
            "old_artifact_id":15247392388,"new_artifact_id":15254809203,"pair_count":len(pairs),"pairs":pairs,
            "matching_prefix_input_pair_count":sum(p["first_12_input_batches_equal"] for p in pairs),
            "exact_history_pair_count":sum(p["first_12_history_exact_equal"] for p in pairs),
            "maximum_train_loss_absolute_difference":max(d["train_absolute_difference"] for p in pairs for d in p["differences"]),
            "maximum_validation_loss_absolute_difference":max(d["validation_absolute_difference"] for p in pairs for d in p["differences"]),
            "rate_exposure":rates,"equal_cumulative_rate_shape_comparison_established":False,
            "checkpoint_tensor_comparison_performed":False,"new_training":False,"independent_evaluation":"NOT_RUN",
            "rights_review":"REVIEW_REQUIRED","model_selection_permitted":False}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old",type=Path,required=True);p.add_argument("--new",type=Path,required=True)
    a=p.parse_args()
    old=load(a.old,87946,OLD_ZIP,("low","reference","high"),"sweep-summary.json")
    new=load(a.new,150625,ZIP,NEW_ARMS,"schedule-sweep-summary.json")
    require(old["dataset-manifest.json"]==new["dataset-manifest.json"],"Dataset drift")
    result=compare(inspect(old,"sweep-summary.json",12,("low","reference","high")),
                   inspect(new,"schedule-sweep-summary.json",24,NEW_ARMS))
    result["new_member_sha256"]={k:sha(v) for k,v in sorted(new.items())}
    print(json.dumps(result,sort_keys=True,indent=2))
