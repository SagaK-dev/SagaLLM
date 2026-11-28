"""Preserve exact schedule sweep and prefix-comparison metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path("experiments/archive/schedule-sweep-46124885819")
ZIP_PINS={"sweep":(15254809203,150625,"19ad506f4e9f9b3b3654d0c44081638e2296fef5a6623c507f0c852f9b081ada"),
          "prefix":(15255591135,2166,"bb3dbbc1e908d78526c657643ef2800ee1434b4bb6b529f73afd475e11d7f917")}
PREFIX_SHA="12d8532f39832e7a9d332e114ce17cfae8792d0b22797d279c90a9b83c69224e"
EXPECTED={"dataset-manifest.json","environment-freeze.txt","schedule-sweep-summary.json"}|{f"{s}-{a}/{n}" for s in (717,719) for a in ("constant_low","constant_reference","warmup_cosine") for n in ("train-report.json","optimizer-exposure.jsonl","offline-audit.json")}

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(obj):return (json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)+"\n").encode()
def verify_members(raw):
    require(set(raw)=={"prefix/prefix-comparison.json"}|{"sweep/"+n for n in EXPECTED},"Member scope mismatch")
    p=raw["prefix/prefix-comparison.json"]
    require(sha(p)==PREFIX_SHA,"Prefix report identity mismatch")
    d=json.loads(p)
    require(d["status"]=="PINNED_SWEEP_PREFIX_COMPARISON_COMPLETE" and d["new_artifact_id"]==15254809203
            and d["matching_prefix_input_pair_count"]==4 and d["exact_history_pair_count"]==0
            and d["checkpoint_tensor_comparison_performed"] is False and d["model_selection_permitted"] is False,
            "Prefix scope mismatch")
    require(d["new_member_sha256"]=={n:sha(raw["sweep/"+n]) for n in EXPECTED},"Sweep member pins mismatch")
    s=json.loads(raw["sweep/schedule-sweep-summary.json"])
    require(s["status"]=="MATCHED_LR_SCHEDULE_SWEEP_COMPLETE" and s["optimizer_updates"]==144
            and s["row_exposures_rechecked"]==576 and s["test_status"]=="NOT_RUN"
            and s["cross_arm_model_selection_performed"] is False and s["model_selection_permitted"] is False,
            "Sweep scope mismatch")
    return {"status":"DURABLE_SCHEDULE_SWEEP_PREFIX_VERIFIED","members":len(raw),
            "original_bytes":sum(map(len,raw.values())),"optimizer_updates_reported":144,
            "exposures_reported":576,"matched_prefix_pairs":4,"exact_history_pairs":0,
            "checkpoint_binaries_preserved":False,"independent_evaluation":"NOT_RUN","rights_review":"REVIEW_REQUIRED"}
def manifest(raw):
    return canonical({"schema":"sagallm-durable-schedule-sweep-v1","archives":{k:{"artifact_id":v[0],"bytes":v[1],"sha256":v[2]} for k,v in ZIP_PINS.items()},
        "members":{k:{"bytes":len(v),"sha256":sha(v)} for k,v in raw.items()},"checkpoint_binaries_preserved":False})
def stage(root,paths):
    require(not root.exists() and not root.is_symlink(),"Refuse overwrite")
    raw={}
    for k,p in paths.items():
        _,size,pin=ZIP_PINS[k]
        require(p.is_file() and not p.is_symlink() and p.stat().st_size==size and sha(p.read_bytes())==pin,"ZIP identity mismatch")
        expected=EXPECTED if k=="sweep" else {"prefix-comparison.json"}
        with zipfile.ZipFile(p) as z:
            require(len(z.infolist())==len(expected) and set(z.namelist())==expected,"ZIP scope mismatch")
            for n in expected:
                i=z.getinfo(n);require(not i.is_dir() and not i.flag_bits&1 and 0<i.file_size<=100000,"Unsafe member")
                raw[k+"/"+n]=z.read(n)
    result=verify_members(raw)
    root.mkdir(parents=True)
    for n,b in raw.items():
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    (root/"archive-manifest.json").write_bytes(manifest(raw))
    return result
def verify(root):
    require(root.is_dir() and not root.is_symlink(),"Unsafe root")
    names={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    expected={"prefix/prefix-comparison.json"}|{"sweep/"+n for n in EXPECTED}
    require(names==expected|{"archive-manifest.json"},"Unexpected files")
    raw={}
    for n in expected:
        p=root/n;require(not p.is_symlink() and p.resolve().is_relative_to(root.resolve()) and 0<p.stat().st_size<=100000,"Unsafe member")
        raw[n]=p.read_bytes()
    require((root/"archive-manifest.json").read_bytes()==manifest(raw),"Manifest mismatch")
    return verify_members(raw)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--sweep",type=Path);p.add_argument("--prefix",type=Path)
    a=p.parse_args();require((a.sweep is None)==(a.prefix is None),"Both ZIPs required")
    print(json.dumps(stage(a.root,{"sweep":a.sweep,"prefix":a.prefix}) if a.sweep else verify(a.root),sort_keys=True))
