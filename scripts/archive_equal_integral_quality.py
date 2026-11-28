"""Pin and preserve exact equal-integral and canonical quality metadata."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile
from collections import Counter

ROOT=Path("experiments/archive/equal-integral-and-corpus-quality")
ZIPS={"equal-integral":(15256371982,172761,"de50615e791627f144387ef3867109340b0dce72224740c2164f69d0ae5ef624"),
      "quality":(15257010429,4731,"ef0df3b6c8388be55f50d5b067315a2824e8bdb898f2ddf84179e23129d548f2"),
      "collisions":(15257180292,136418,"53ad3831380218c723d4c1f05e73185babfd026d974ebb332727f1f5bbc39a80")}
PINS={
"equal-integral/717-constant_matched/checkpoint-tensor-index.json":"42116bffb082454d4cae59416bd39e23aaf7a76b6ee2377af10a143d68168b87",
"equal-integral/717-constant_matched/offline-audit.json":"68d99f3ebb873dab60f0003fb59a633e5850fb37a7b438c4d368815af3ee09e0",
"equal-integral/717-constant_matched/optimizer-exposure.jsonl":"75c644a1ba592897c4d81d6eda635f0e269a2a6de4c75c9b8b556dc5ae2227c5",
"equal-integral/717-constant_matched/train-report.json":"0b9742813d3f827bdb281a406d0888ce51ad732998461479902c313bc5cb68c4",
"equal-integral/717-warmup_cosine/checkpoint-tensor-index.json":"6d0ddd6653408ef9f72106ee8be54e1ad8dee218dd82129c09eb6f02123c7512",
"equal-integral/717-warmup_cosine/offline-audit.json":"26c7cc13e73cb62df8ba29113db657cadf300b984ab1501c7e7e15e83e336049",
"equal-integral/717-warmup_cosine/optimizer-exposure.jsonl":"c2d1bd941f836834e8e1b736c2d616db67b69665b563e8ef4d0393d963c0e036",
"equal-integral/717-warmup_cosine/train-report.json":"f6d548486fba323c658ad65304b387c44db73458620147e6a0033770e54b2ae8",
"equal-integral/719-constant_matched/checkpoint-tensor-index.json":"065cb939ba3f9a5eebc37bb2576485931fe00a8abdd88838110e2349161606fe",
"equal-integral/719-constant_matched/offline-audit.json":"4103e6f0d5caa58f4861eb630d0c247f1c9a85d35ed7b540ecd98b13ddde6d94",
"equal-integral/719-constant_matched/optimizer-exposure.jsonl":"80ac9fbd6e40deb7caa4b16ee17c8c7851f8f02c61f167d1a1b91efb7e7196ed",
"equal-integral/719-constant_matched/train-report.json":"f1d968614fbacb108696707ccd5a5745c15e7ff835be02f5d1d20d71b12d501c",
"equal-integral/719-warmup_cosine/checkpoint-tensor-index.json":"8b8ee49017dc8904ba4a0a78566388e26be4f203824a94629a696f03112b3495",
"equal-integral/719-warmup_cosine/offline-audit.json":"42422892c58cda7956d20a5b921fd56dcaae18cb3f036917fdb9662a44a274ba",
"equal-integral/719-warmup_cosine/optimizer-exposure.jsonl":"688f2712b97755f88c433b1e4a6e53c4ea74c43b0e946354948cda4936c74f63",
"equal-integral/719-warmup_cosine/train-report.json":"10f778b3f9a2b6d7291c8277644c22ae67e935b0ed61d3ec647390055dc22f1c",
"equal-integral/727-constant_matched/checkpoint-tensor-index.json":"30a17d4c9978078aeb8012c39f6cd9bec8642d38ea3cffe2bb18ab0c49be8040",
"equal-integral/727-constant_matched/offline-audit.json":"2ed8f989f7dd79e9c3d47b58b47e8e2c2b84fd77a16a861562008adb70485684",
"equal-integral/727-constant_matched/optimizer-exposure.jsonl":"81489196471a02b3d201e9f11318358bda3b7719caa14d12eaaf1d459b0eb92c",
"equal-integral/727-constant_matched/train-report.json":"66284632346a94c6a9fc2dd4cc5e9db43d1bd9b10563afeb546cce23f00abbb8",
"equal-integral/727-warmup_cosine/checkpoint-tensor-index.json":"0eda77ceb2e4ed28f90228aab6ef46d78263cd79cfb329634fd9480dd2f0827a",
"equal-integral/727-warmup_cosine/offline-audit.json":"b9676be6af6858fbf181e2dec8ae6664635266fb6ecfc2269329f5a225ffbc5d",
"equal-integral/727-warmup_cosine/optimizer-exposure.jsonl":"61f8d52f7fa28d918d4d805bbe9faf99c6f17c98f9b971fc7d3226a31ad2c6eb",
"equal-integral/727-warmup_cosine/train-report.json":"c5f4dfd5c6a8fc1142f0a3e431580a1f62647627e87b8a4bb793898ef9d85fae",
"equal-integral/dataset-manifest.json":"1e9c0ff933c5096cf2d633e81ec0c90498006fab45397a28a4c073fbf181a23a",
"equal-integral/environment-freeze.txt":"b4e9957445e5288c791fa85b733451e5217fc724d9188e9474c8de4902a3953b",
"equal-integral/equal-integral-summary.json":"8fddd0dbb5104ad2dbc240eaa3b6541888f91d17c7aa219ca748132dfa891a96",
"quality/canonical-quality.json":"6810535b181f96f2ec949e64220a0ee032f84000bfb7f4f35f6b072cf5c6e9db",
"collisions/prompt-collision-review.json":"474bfc40d76aab67dd23771a2214e673a99c0c8845231a338880439069e03397"}
def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(o):return json.dumps(o,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode()
def verify(raw):
    require(set(raw)==set(PINS) and all(sha(v)==PINS[k] for k,v in raw.items()),"Member pin mismatch")
    def obj(n):return json.loads(raw[n])
    s=obj("equal-integral/equal-integral-summary.json")
    require(s["status"]=="EQUAL_INTEGRAL_LR_SWEEP_COMPLETE" and s["optimizer_updates"]==144
            and s["row_exposures_rechecked"]==576 and s["test_status"]=="NOT_RUN"
            and s["model_selection_permitted"] is False,"Sweep scope mismatch")
    seen=set();indices=[];paired={}
    for r in s["runs"]:
        key=(r["seed"],r["arm"])
        require(key not in seen and math.isclose(r["sum_step_rates"],.000522,rel_tol=0,abs_tol=1e-15),"Arm drift")
        seen.add(key);p=f"equal-integral/{key[0]}-{key[1]}/"
        report=obj(p+"train-report.json");audit=obj(p+"offline-audit.json")
        require(r["report_sha256"]==audit["report_sha256"]==PINS[p+"train-report.json"]
                and r["trace_sha256"]==audit["trace_sha256"]==PINS[p+"optimizer-exposure.jsonl"]
                and audit["steps_rechecked"]==24 and audit["row_exposures_rechecked"]==96
                and report["heldout_test_status"]=="NOT_RUN","Report/trace scope mismatch")
        events=[json.loads(line) for line in raw[p+"optimizer-exposure.jsonl"].splitlines()]
        tip="0"*64
        for n,e in enumerate(events):
            require(e["seq"]==n and e["prev_sha256"]==tip and sha(canonical({k:v for k,v in e.items() if k!="event_sha256"}))==e["event_sha256"],"Trace chain mismatch")
            tip=e["event_sha256"]
        require(tip==audit["tip_sha256"],"Trace tip mismatch")
        intents=[e["payload"] for e in events if e["kind"]=="step_intent"]
        require(len(intents)==24 and math.isclose(math.fsum(i["optimizer_lr"] for i in intents),.000522,rel_tol=0,abs_tol=1e-15),"Actual recorded LR mismatch")
        h=sha(canonical([{k:v for k,v in i.items() if k!="optimizer_lr"} for i in intents]))
        require(h==r["schedule_sha256"],"Batch schedule mismatch")
        paired.setdefault(key[0],set()).add(h)
        t=obj(p+"checkpoint-tensor-index.json")
        require(set(t)=={"best","last"},"Role mismatch")
        for role in ("best","last"):
            x=t[role];tensors=x["tensors"]
            require(x["checkpoint_sha256"]==r[role+"_checkpoint_sha256"]==report[role+"_checkpoint_sha256"]
                    and x["step"]==(r["best_step"] if role=="best" else 24)
                    and x["tensor_count"]==len(tensors)==39 and x["tensor_bytes"]==8461056
                    and x["tensor_bytes"]==sum(v["bytes"] for v in tensors.values())
                    and x["model_index_sha256"]==sha(canonical(tensors)),"Tensor metadata binding mismatch")
            indices.append(x["model_index_sha256"])
    require(seen=={(seed,arm) for seed in (717,719,727) for arm in ("constant_matched","warmup_cosine")}
            and all(len(x)==1 for x in paired.values()),"Matched arm coverage")
    q=obj("quality/canonical-quality.json");c=obj("collisions/prompt-collision-review.json");p=c["candidate_plan"]
    require(q["exact_response_count"]==17536 and q["distinct_normalized_signatures"]==111
            and c["collision_groups"]==28 and p["candidate_count"]==1448
            and p["membership_sha256"]==sha(canonical(sorted(r["row_sha256"] for r in p["candidate_rows"])))
            =="8ddb2ff98d0b4f1d6a83557bc168151c7d32ccdd535f8dd50a32da96cebf8064"
            and Counter(r["language"] for r in p["candidate_rows"])=={"ja":724,"en":724},"Dataset quality metadata mismatch")
    return {"status":"DURABLE_EQUAL_INTEGRAL_AND_QUALITY_VERIFIED","original_members":len(raw),"original_bytes":sum(map(len,raw.values())),
            "tensor_metadata_roles":len(indices),"distinct_model_index_hashes":len(set(indices)),
            "tensor_bytes_reopened":False,"train_membership_count":1448,"rights_review":"REVIEW_REQUIRED",
            "independent_evaluation":"NOT_RUN","model_selection_permitted":False}
def manifest(raw):
    return canonical({"schema":"sagallm-durable-equal-integral-quality-v1","archives":ZIPS,
                      "members":{k:{"sha256":PINS[k],"bytes":len(v)} for k,v in raw.items()},
                      "checkpoint_binaries_preserved":False})+b"\n"
def stage(root,paths):
    require(not root.exists() and not root.is_symlink(),"Refuse overwrite")
    raw={}
    for k,p in paths.items():
        _,size,pin=ZIPS[k]
        require(p.is_file() and not p.is_symlink() and p.stat().st_size==size and sha(p.read_bytes())==pin,"ZIP pin mismatch")
        expected={n.removeprefix(k+"/") for n in PINS if n.startswith(k+"/")}
        with zipfile.ZipFile(p) as z:
            require(len(z.infolist())==len(expected) and set(z.namelist())==expected,"ZIP member mismatch")
            for n in expected:
                info=z.getinfo(n);require(not info.is_dir() and not info.flag_bits&1 and 0<info.file_size<=1000000,"Unsafe member")
                raw[k+"/"+n]=z.read(n)
    result=verify(raw);root.mkdir(parents=True)
    for n,b in raw.items():
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    (root/"archive-manifest.json").write_bytes(manifest(raw))
    return result
def local(root):
    require(root.is_dir() and not root.is_symlink(),"Unsafe root")
    require({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}==set(PINS)|{"archive-manifest.json"},"File scope mismatch")
    raw={}
    for n in PINS:
        p=root/n;require(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()) and 0<p.stat().st_size<=1000000,"Unsafe file")
        raw[n]=p.read_bytes()
    require((root/"archive-manifest.json").read_bytes()==manifest(raw),"Manifest mismatch")
    return verify(raw)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT)
    for k in ZIPS:p.add_argument("--"+k,type=Path)
    a=p.parse_args();paths={k:getattr(a,k.replace("-","_")) for k in ZIPS}
    require(all(paths.values()) or not any(paths.values()),"All ZIPs required")
    print(json.dumps(stage(a.root,paths) if all(paths.values()) else local(a.root),sort_keys=True))
