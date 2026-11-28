"""Preserve exact metadata from completed reproduction and LR sweep runs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path("experiments/archive/prospective-2026-11-28")
ARCHIVES={
 "reproduction":(46105248933,15247001728,19306,"0047b411672deba1c1c929685469125f51156c1858dbe3d2b7280f00c64d7ea1"),
 "lr-sweep":(46105411718,15247392388,87946,"d9d5528c4068c65b9440c09e00d9ac1da6b688c2c6ca8490c6d40bfe45d46fd4"),
}
PINS={
 "reproduction/dataset-manifest.json":"1e9c0ff933c5096cf2d633e81ec0c90498006fab45397a28a4c073fbf181a23a",
 "reproduction/environment.json":"4a5e1052a59a8f81aeda64c0bd93fea5ace666603fc60b2a12b0711ff3a105ac",
 "reproduction/environment-freeze.txt":"a58a75f6e5f77a8d1124f63cee361dc47cedfba17958c19417d863c44d1c76ab",
 "reproduction/offline-audit.json":"1f5e359d7c2dac614367e44435bebf29bca079601919548e208865310cb6b8d4",
 "reproduction/optimizer-exposure.jsonl":"6a2ccd2622c4fb2c4df1b838ea1a22fc3ca344e98fd3b98d292b9e35b8cfec05",
 "reproduction/pilot-summary.json":"d163fa98c04eb5d0976fd135b161c6355d49f5c4453f8b3007899f49eafcd441",
 "reproduction/reproduction-report.json":"90261c56e4a4b24a0e7d56c21e4b7ba5adc9e2d607fff470067affa95985bf19",
 "reproduction/train-report.json":"88f40790ef6f0ee1f00e1c07f25ecc1b144640734621036bd8fcd8da6b3753e1",
 "lr-sweep/dataset-manifest.json":"1e9c0ff933c5096cf2d633e81ec0c90498006fab45397a28a4c073fbf181a23a",
 "lr-sweep/environment-freeze.txt":"a58a75f6e5f77a8d1124f63cee361dc47cedfba17958c19417d863c44d1c76ab",
 "lr-sweep/sweep-summary.json":"1fe63bf681ca6d6c5d71efba466e01e5b11974b8cf25964fd7cc24a4bf5caea7",
}
RUN_PINS={
 "717-high":("bed921117a3655ae8030c372f1971c739ce78d8243b3f817c82b0a704c12b553","7c88f8d62dc8b8b8210aee30c144c5dfbb304e5a5c3b979d51573eeff4608875","5cc1bfc22fe702c79e75183e29ec787fe9523bb4e6381a0f4b16e96e6c928a46"),
 "717-low":("9b72caab5e1463373f94e49861bd850e822273000237e99654170ff664ca5923","8afd9205cee74be318aa259bb2fd485bb911aae116cd7ed37eeb3e5cfc7c7e18","1aa145400fd3462a4e31a639aa0eeda3755663b8a57e72bc05845b2f336b4dd8"),
 "717-reference":("1f21b99834c61aa3537152d488c17a6428847cab147e1a02d8621e3eefc49b4a","70dac0a19ab27207556d387f8e5ba8bc06b211f52cb0474a5ea81453329103a3","60c8e44a05be0f992a887d827c67ce06f8d1ecef7dc5ef05bd9253de082ed0e4"),
 "719-high":("0461f0f7c12cdcc218f0453dec83002f40140c087409351cdc1110c55959b053","d96e4d9a5c2d906f4a922d76e1fc02318f830c97950cccd76a57ffeb40845765","ae51137cb35e427ef5404002e2128f719d98267efe176f5c92ba9a020dfa44a3"),
 "719-low":("27e106345be75a0ba681ae55ee76fe8e6908fac240aa14dc68bed9423b20c9ae","dce99cc47f13184a498051a738ff762ff6256f59a68186219cd9bf54ca7c262b","3a064bbcb2d5a273bc9eaf9c74a03f0c4c379eef07dd80f48edd9bf5efca1f27"),
 "719-reference":("f070ef663f25d99ccfdfaebc3ace09d1b6c27f654272bb98301aa805ead68cd7","859e4fce2243c9f4ec580318398484b59d71324dae5c79ea0fcf4ab8b924e3a4","b0aba49b02e1ef848a6c0b6c1214e13dc6c234108d5e093f0dba70f988587dff"),
}
for arm,hs in RUN_PINS.items():
    for name,h in zip(("train-report.json","optimizer-exposure.jsonl","offline-audit.json"),hs):
        PINS[f"lr-sweep/{arm}/{name}"]=h

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def manifest(raw):
    return {"schema":"sagallm-prospective-evidence-archive-v1",
        "archives":{k:{"run_id":v[0],"artifact_id":v[1],"size":v[2],"zip_sha256":v[3]} for k,v in ARCHIVES.items()},
        "members":{k:{"sha256":PINS[k],"bytes":len(raw[k])} for k in sorted(PINS)},
        "checkpoint_binaries_preserved":False,"rights_review":"REVIEW_REQUIRED","independent_evaluation":"NOT_RUN"}

def reconcile(raw):
    require(set(raw)==set(PINS) and all(sha(raw[k])==h for k,h in PINS.items()),"Member identity mismatch")
    def obj(n):return json.loads(raw[n])
    rep=obj("reproduction/reproduction-report.json")
    require(rep["status"]=="BOUNDED_RERUN_COMPARISON_COMPLETE" and
            all(rep[k] is True for k in ("identical_ordered_batch_payloads","model_state_and_config_equal",
             "optimizer_state_equal","exact_serialized_checkpoint_equality","trace_file_equal","history_exact_equal","best_step_equal"))
            and rep["universal_reproducibility_established"] is False
            and rep["model_selection_permitted"] is False,"Reproduction scope mismatch")
    rerun=obj("reproduction/pilot-summary.json")
    require(rerun["report_sha256"]==PINS["reproduction/train-report.json"]
            and rerun["trace_sha256"]==PINS["reproduction/optimizer-exposure.jsonl"],"Rerun binding")
    sweep=obj("lr-sweep/sweep-summary.json")
    require(sweep["status"]=="MATCHED_VALIDATION_ONLY_SWEEP_COMPLETE" and sweep["optimizer_updates"]==72
            and sweep["row_exposures_rechecked"]==288 and sweep["test_status"]=="NOT_RUN"
            and sweep["cross_arm_model_selection_performed"] is False,"Sweep scope mismatch")
    schedules={}
    for row in sweep["runs"]:
        arm=f'{row["seed"]}-{row["arm"]}'
        prefix=f"lr-sweep/{arm}/"
        r=obj(prefix+"train-report.json");a=obj(prefix+"offline-audit.json")
        events=[json.loads(line) for line in raw[prefix+"optimizer-exposure.jsonl"].splitlines()]
        require(row["report_sha256"]==a["report_sha256"]==PINS[prefix+"train-report.json"]
                and row["trace_sha256"]==a["trace_sha256"]==PINS[prefix+"optimizer-exposure.jsonl"]
                and r["heldout_test_status"]=="NOT_RUN" and r["optimizer_steps_completed"]==a["steps_rechecked"]==12
                and a["row_exposures_rechecked"]==48 and r["best_checkpoint_step"]==row["best_step"]
                and r["best_validation_response_loss"]==row["best_validation_loss"],"Arm report binding")
        tip="0"*64
        for i,e in enumerate(events):
            # Original v1 canonical event hash chain; verify independently of torch.
            body={k:v for k,v in e.items() if k!="event_sha256"}
            require(e["seq"]==i and e["prev_sha256"]==tip and sha(canonical(body))==e["event_sha256"],"Trace chain mismatch")
            tip=e["event_sha256"]
        require(tip==a["tip_sha256"],"Trace tip mismatch")
        intents=[e["payload"] for e in events if e["kind"]=="step_intent"]
        schedule=sha(canonical(intents))
        require(len(intents)==12 and sum(len(i["rows"]) for i in intents)==48
                and schedule==row["schedule_sha256"],"Input schedule mismatch")
        schedules.setdefault(row["seed"],set()).add(schedule)
    require(len(sweep["runs"])==6 and len(schedules)==2 and all(len(s)==1 for s in schedules.values()),"Sweep coverage")
    return {"status":"DURABLE_REPRODUCTION_AND_SWEEP_METADATA_VERIFIED","original_members":len(raw),
            "original_bytes":sum(map(len,raw.values())),"reproduction_comparison_verified_from_original_report":True,
            "sweep_arms":6,"sweep_optimizer_returns_reported":72,"sweep_row_exposures_reported":288,
            "matched_input_schedule_rechecked":True,"checkpoint_bytes_rechecked_here":False,
            "independent_evaluation":"NOT_RUN","rights_review":"REVIEW_REQUIRED","model_selection_permitted":False}

def stage(root,archives):
    require(not root.exists() and not root.is_symlink(),"Refuse overwrite")
    raw={}
    for key,path in archives.items():
        _,_,size,h=ARCHIVES[key]
        require(path.is_file() and not path.is_symlink() and path.stat().st_size==size and sha(path.read_bytes())==h,"ZIP identity mismatch")
        expected={k.removeprefix(key+"/") for k in PINS if k.startswith(key+"/")}
        with zipfile.ZipFile(path) as z:
            require(len(z.infolist())==len(expected) and set(z.namelist())==expected,"Archive member coverage")
            for n in expected:
                m=z.getinfo(n);require(not m.is_dir() and not m.flag_bits&1 and 0<m.file_size<=100000,"Unsafe member")
                raw[key+"/"+n]=z.read(n)
    report=reconcile(raw)
    root.mkdir(parents=True)
    for name,b in raw.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    (root/"archive-manifest.json").write_bytes(canonical(manifest(raw))+b"\n")
    return report

def verify(root):
    require(root.is_dir() and not root.is_symlink(),"Unsafe archive root")
    paths={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(paths==set(PINS)|{"archive-manifest.json"},"Unexpected archive files")
    raw={}
    for n in PINS:
        p=root/n
        require(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()) and p.stat().st_size<=100000,"Unsafe member path")
        raw[n]=p.read_bytes()
    require((root/"archive-manifest.json").read_bytes()==canonical(manifest(raw))+b"\n","Archive manifest mismatch")
    return reconcile(raw)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT);p.add_argument("--reproduction",type=Path);p.add_argument("--lr-sweep",type=Path)
    a=p.parse_args()
    require((a.reproduction is None)==(a.lr_sweep is None),"Both original ZIPs required")
    print(json.dumps(stage(a.root,{"reproduction":a.reproduction,"lr-sweep":a.lr_sweep}) if a.reproduction else verify(a.root),sort_keys=True))
