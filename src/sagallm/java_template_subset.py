"""Strictly bounded, opt-in subset of the original synthetic Java corpus.

The exact membership was recorded by M2.6-b, artifact 15257180292.
Never accept arbitrary replacement data under this experiment's label.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from collections import Counter
from .java_diverse_corpus import (_expected_rows,_quality,_json,SPLITS,SCHEMA,SEED,
                                 verify_java_diverse_corpus)

PARENT_MANIFEST="1e9c0ff933c5096cf2d633e81ec0c90498006fab45397a28a4c073fbf181a23a"
MEMBERSHIP="8ddb2ff98d0b4f1d6a83557bc168151c7d32ccdd535f8dd50a32da96cebf8064"
MODE="template_capped_v1"
COUNT=1448
SCHEMA_SUBSET="sagallm-java-template-capped-v1"

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return _json(x).encode("utf8")
def file_bytes(rows):return b"".join(canonical(r)+b"\n" for r in rows)
def membership(rows):return sha(canonical(sorted(sha(canonical(r)) for r in rows)))

def canonical_parent():
    rows=_expected_rows()
    files={s:{"filename":s+".jsonl","sha256":sha(file_bytes(rs)),"count":len(rs),
              "families":list(SPLITS[s][0])+list(SPLITS[s][2])} for s,rs in rows.items()}
    manifest={"schema":SCHEMA,"seed":SEED,"files":files,"quality":_quality(rows),
              "creation":"original code generation only; template-derived bilingual tasks",
              "license_attestation":"NOT_LEGALLY_REVIEWED","third_party_sources":[],
              "model_weights_source":"NONE","data_split":"disjoint_by_whole_task_family"}
    require(sha(canonical(manifest)+b"\n")==PARENT_MANIFEST,"Canonical parent generator drift")
    return rows,manifest

def derive_manifest(selected,parent):
    require(len(selected)==COUNT and membership(selected)==MEMBERSHIP,"Pinned membership mismatch")
    require(len({r["id"] for r in selected})==COUNT and len({r["family"] for r in selected})==90,
            "Subset identity/family mismatch")
    langs=Counter(int(r["id"].rsplit("-",1)[1])%2 for r in selected)
    require(langs=={0:724,1:724},"Subset language balance mismatch")
    files={s:dict(v) for s,v in parent["files"].items()}
    files["train"].update(count=COUNT,sha256=sha(file_bytes(selected)))
    return {"schema":SCHEMA_SUBSET,"parent_manifest_sha256":PARENT_MANIFEST,
            "membership_sha256":MEMBERSHIP,"files":files,"plan_artifact_id":15257180292,
            "selection_rule":"template-cap-eight-per-family-signature-language",
            "canonical_parent_order_preserved":True,"license_attestation":"NOT_LEGALLY_REVIEWED",
            "independent_evaluation":"NOT_RUN","model_selection_permitted":False}

def read_bounded(p,limit):
    require(p.is_file() and not p.is_symlink() and 0<p.stat().st_size<=limit,"Unsafe dataset file")
    with p.open("rb") as f:raw=f.read(limit+1)
    require(0<len(raw)<=limit,"Dataset size limit")
    return raw

def verify_subset_train(root):
    """Rebuild source in memory; only read manifest and TRAIN on disk."""
    root=Path(root)
    require(root.is_dir() and not root.is_symlink(),"Unsafe dataset directory")
    parent_rows,parent=canonical_parent()
    raw=read_bounded(root/"train.jsonl",16*1024*1024)
    selected=[json.loads(line) for line in raw.splitlines()]
    expected_ids={r["id"] for r in selected}
    expected=[r for r in parent_rows["train"] if r["id"] in expected_ids]
    require(selected==expected and raw==file_bytes(expected),"Subset source bytes/order mismatch")
    m=derive_manifest(selected,parent)
    require(read_bounded(root/"manifest.json",65536)==canonical(m)+b"\n","Subset manifest drift")
    return m

def verify_template_subset(root):
    root=Path(root);m=verify_subset_train(root)
    for s in ("validation","test"):
        raw=read_bounded(root/(s+".jsonl"),16*1024*1024)
        require(sha(raw)==m["files"][s]["sha256"],"Heldout bytes drift")
    return m

def build_template_subset(source,output,row_hashes):
    source=Path(source);output=Path(output)
    verify_java_diverse_corpus(source)
    require(sha(read_bounded(source/"manifest.json",65536))==PARENT_MANIFEST,"Parent pin mismatch")
    require(not output.exists() and not output.is_symlink(),"Refuse overwrite")
    require(len(row_hashes)==COUNT and sha(canonical(sorted(row_hashes)))==MEMBERSHIP,"Plan membership drift")
    rows,parent=canonical_parent()
    selected=[r for r in rows["train"] if sha(canonical(r)) in row_hashes]
    m=derive_manifest(selected,parent)
    output.mkdir()
    try:
        (output/"train.jsonl").write_bytes(file_bytes(selected))
        for s in ("validation","test"):
            (output/(s+".jsonl")).write_bytes(read_bounded(source/(s+".jsonl"),16*1024*1024))
        (output/"manifest.json").write_bytes(canonical(m)+b"\n")
        return verify_template_subset(output)
    except BaseException:
        import shutil
        shutil.rmtree(output)
        raise
