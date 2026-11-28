"""Preserve exact metadata-only originals of the successful traced context pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path("experiments/archive/traced-context-pilot-46104477318")
ZIP_SHA="103201aae7eb8f6f1484c91c001a82dc43244d7cbaa9ab4ddcdd239eca7d056f"
MEMBERS={
 "optimizer-exposure.jsonl":(28631,"6a2ccd2622c4fb2c4df1b838ea1a22fc3ca344e98fd3b98d292b9e35b8cfec05"),
 "offline-audit.json":(1524,"9b329e24c0ce163e7c9faccd38512de780e1a9c5a4b3d384c083b1988dc91d3d"),
 "train-report.json":(5601,"0254485bb2c9ceee06227405689edd0476a7200deda6e8711789c4f06ae7c86e"),
 "dataset-manifest.json":(2586,"1e9c0ff933c5096cf2d633e81ec0c90498006fab45397a28a4c073fbf181a23a"),
 "pilot-summary.json":(1872,"481e4d324a0e57abe345be96661b54ec51ac953b57a08d334dfd77dfb00d7034"),
}

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def manifest():
    return {"schema":"sagallm-durable-traced-pilot-originals-v1",
            "run_id":46104477318,"execution_revision":"a089929068b5b64ac69bfb3a7d916b1864b8a299",
            "head_revision":"af836553f4580d144299557a25e219a5543eddec",
            "evidence_artifact_id":15247550137,"evidence_zip_sha256":ZIP_SHA,
            "members":{n:{"bytes":s,"sha256":h} for n,(s,h) in sorted(MEMBERS.items())},
            "checkpoint_artifact_id":15247155692,
            "checkpoint_zip_sha256":"f027874de13707df48567bae374ad8278d968160cb8580299278ae06119c6773",
            "checkpoint_bytes_stored_in_git":False,"checkpoint_artifact_expiry":"2026-12-28T03:47:42Z",
            "metadata_only":True,"rights_review":"REVIEW_REQUIRED","independent_evaluation":"NOT_RUN",
            "model_selection_permitted":False,"historical_exposure_verified":False}

def manifest_raw():return (json.dumps(manifest(),sort_keys=True,indent=2)+"\n").encode()

def read_exact(path,size,digest):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size==size,"Unsafe or changed original")
    with path.open("rb") as f:raw=f.read(size+1)
    require(len(raw)==size and sha(raw)==digest,"Original digest mismatch")
    return raw

def verify(root):
    require(root.is_dir() and not root.is_symlink() and {p.name for p in root.iterdir()}
            ==set(MEMBERS)|{"archive-manifest.json"},"Unexpected archive directory members")
    m=manifest_raw()
    require(read_exact(root/"archive-manifest.json",len(m),sha(m))==m,"Archive manifest drift")
    originals={n:read_exact(root/n,size,h) for n,(size,h) in MEMBERS.items()}
    summary=json.loads(originals["pilot-summary.json"]);audit=json.loads(originals["offline-audit.json"])
    report=json.loads(originals["train-report.json"])
    require(summary["report_sha256"]==MEMBERS["train-report.json"][1]
            and summary["trace_sha256"]==MEMBERS["optimizer-exposure.jsonl"][1]
            and summary["data_manifest_sha256"]==MEMBERS["dataset-manifest.json"][1]
            and report["optimizer_steps_completed"]==audit["steps_rechecked"]==12
            and audit["row_exposures_rechecked"]==48 and summary["model_selection_permitted"] is False
            and summary["independent_evaluation"]=="NOT_RUN","Original evidence cross-reference mismatch")
    return {"status":"DURABLE_TRACED_PILOT_ORIGINALS_VERIFIED","original_count":len(MEMBERS),
            "original_bytes":sum(v[0] for v in MEMBERS.values()),"manifest_sha256":sha(m),
            "artifact_access_required_for_local_verification":False,"checkpoint_bytes_stored_in_git":False}

def stage(root,archive):
    require(not root.exists() and not root.is_symlink(),"Refuse overwrite")
    raw=read_exact(archive,16203,ZIP_SHA)
    with zipfile.ZipFile(archive) as z:
        require(len(z.infolist())==len(MEMBERS) and set(z.namelist())==set(MEMBERS),"Unexpected ZIP members")
        originals={}
        for name,(size,h) in MEMBERS.items():
            m=z.getinfo(name)
            require(m.file_size==size and not m.is_dir() and not m.flag_bits&1,"Unsafe original member")
            with z.open(m) as f:body=f.read(size+1)
            require(len(body)==size and sha(body)==h,"Original member pin mismatch")
            originals[name]=body
    root.mkdir(parents=True)
    for n,b in {**originals,"archive-manifest.json":manifest_raw()}.items():
        with (root/n).open("xb") as f:f.write(b)
    return verify(root)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--stage-zip",type=Path)
    a=p.parse_args()
    print(json.dumps(stage(a.root,a.stage_zip) if a.stage_zip else verify(a.root),sort_keys=True))
