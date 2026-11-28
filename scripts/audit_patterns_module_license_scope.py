"""Reclassify broad license header signals using exact pinned source bytes.

This creates a human review queue; it never clears source rights.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
from scripts.audit_public_corpus_file_rights import tree, verified_file, signals, SPECS, require, sha

ZIP_SHA="69064805a2406c82610a9f593a85f48f30af251f1a09a47a148f3f79b1a4453c"
INDEX_SHA="c423923211a3dee9ab32ec5e288c876634d809558f2b79d9f5068cb03d44cf8f"
SUMMARY_SHA="3b5f1bd56a10a24b6599811add4ba246539ad625c13a7b164be6cfedc1f803ec"
NOTICE_SHA="84be12a8ad45e3a40f2bf028aa72cc574e70b68feb8b76c7bc673cc3135300f1"
NOTE="This project is licensed under the MIT license. Module model-view-viewmodel is using ZK framework licensed under LGPL (see lgpl-3.0.txt)."
MODULE="model-view-viewmodel/"
PINS={"lgpl-3.0.txt":"95107e3c5982fdc8213bbc97da7786b2b002ff9f",
      "model-view-viewmodel/pom.xml":"e77d53ef7396a6738a735b3d7037857ace314151"}
MODULE_FILES={
 "Book.java":"e7f8f035803d81eaaa06077dd0d2e79bf06cddca",
 "BookService.java":"34bb25c07b8a1c0b301976c9d313554b85dcc4b7",
 "BookServiceImpl.java":"7d7257fcc8166e50eebe38c85d009a4a9757fd1c",
 "BookViewModel.java":"7eb81d1f022439cb133785f5fc393ca495e3b989",
}

def read_index(archive):
    require(archive.is_file() and not archive.is_symlink() and archive.stat().st_size==227177
            and sha(archive.read_bytes())==ZIP_SHA,"Index ZIP pin mismatch")
    pins={"public-source-file-rights-index.jsonl":(1851075,INDEX_SHA),
          "rights-index-summary.json":(1955,SUMMARY_SHA),"public-notice-candidates.json":(1335,NOTICE_SHA)}
    with zipfile.ZipFile(archive) as z:
        require(len(z.infolist())==3 and set(z.namelist())==set(pins),"Unexpected index members")
        raw={}
        for n,(size,digest) in pins.items():
            require(z.getinfo(n).file_size==size,"Index member size mismatch")
            raw[n]=z.read(n);require(sha(raw[n])==digest,"Index member pin mismatch")
    rows=[json.loads(line) for line in raw["public-source-file-rights-index.jsonl"].splitlines()]
    summary=json.loads(raw["rights-index-summary.json"])
    require(len(rows)==2926 and summary["possible_non_root_signal_count"]==1353
            and summary["source_index_sha256"]==INDEX_SHA and all(r["rights_review"]=="REVIEW_REQUIRED" for r in rows),
            "Original index scope mismatch")
    return rows

def classify(text,path):
    header=text[:8192]
    original=signals(text,"MIT")
    # Remove only the exact boilerplate sentence, not every LGPL match.
    # Keep module-specific signals even when the note is boilerplate.
    note_count=header.count(NOTE)
    remainder=signals(text.replace(NOTE,"",1),"MIT") if note_count==1 else original
    if path.startswith(MODULE):
        category="MODULE_ZK_DEPENDENCY_REVIEW_REQUIRED"
    elif original["possible_non_root_license_signal"] and note_count==1 and not remainder["possible_non_root_license_signal"] and "MIT" in remainder["header_markers"]:
        category="MIT_HEADER_WITH_OTHER_MODULE_DEPENDENCY_NOTE"
    elif original["possible_non_root_license_signal"]:
        category="OTHER_HEADER_SIGNAL_REVIEW_REQUIRED"
    else:
        category="NO_NON_ROOT_SIGNAL_DETECTED"
    return category,note_count,remainder

def run(archive,root,out):
    rows=read_index(archive)
    rev=SPECS["patterns"][0];t=tree(root,rev)
    notices={}
    for path,digest in PINS.items():
        require(t[path][1]==digest,"Dependency evidence blob mismatch")
        raw=verified_file(root,path,t[path],512*1024)
        notices[path]={"git_blob":digest,"sha256":sha(raw),"bytes":len(raw)}
        if path.endswith(".txt"):
            require(b"GNU LESSER GENERAL PUBLIC LICENSE" in raw and NOTE.encode() in raw,"Missing LGPL notice")
        else:
            require(b"<!DOCTYPE" not in raw.upper() and b"<!ENTITY" not in raw.upper(),"Unexpected XML declarations")
            xml=ET.fromstring(raw)
            deps=[]
            for dep in xml.findall(".//{*}dependency"):
                group=dep.findtext("{*}groupId")
                artifact=dep.findtext("{*}artifactId")
                if group and ("zkoss" in group.lower() or "zkoss" in (artifact or "").lower()):
                    deps.append({"group_id":group,"artifact_id":artifact,"version_declared":dep.findtext("{*}version")})
            require(deps,"No ZK dependency declaration")
            notices[path]["zk_dependency_declarations"]=deps
    result=[];modules={}
    for row in rows:
        if row["repo"]!="patterns":continue
        p=row["path"];entry=t[p]
        require(entry[1]==row["source_git_blob"],"Original source blob mismatch")
        raw=verified_file(root,p,entry,96000)
        text=raw.decode("utf8").replace("\r\n","\n").replace("\r","\n")
        require(sha(text.encode())==row["normalized_source_sha256"],"Original source text mismatch")
        orig=signals(text,"MIT")
        require(all(row[k]==v for k,v in orig.items()),"Previous heuristic not reproduced")
        category,count,remainder=classify(text,p)
        if p.startswith(MODULE):
            modules[Path(p).name]=entry[1]
        result.append({"path":p,"split":row["split"],"source_git_blob":entry[1],
                       "normalized_source_sha256":row["normalized_source_sha256"],"category":category,
                       "exact_boilerplate_count":count,"residual_header_markers":remainder["header_markers"],
                       "residual_spdx":remainder["spdx_header_values"],"rights_review":"REVIEW_REQUIRED"})
    require(len(result)==1378 and modules==MODULE_FILES,"Module source coverage mismatch")
    counts=Counter(r["category"] for r in result)
    by_split={k:dict(Counter(r["split"] for r in result if r["category"]==k)) for k in counts}
    out.mkdir()
    (out/"patterns-review-queue.jsonl").write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in result))
    report={"schema":"sagallm-patterns-module-license-scope-v1","status":"PINNED_MODULE_SCOPE_REVIEW_QUEUE_COMPLETE",
            "prior_artifact_id":15248000145,"prior_index_sha256":INDEX_SHA,"upstream_revision":rev,
            "patterns_files":len(result),"prior_flagged_files":1353,"categories":dict(counts),"category_split_counts":by_split,
            "module_file_count":len(modules),"dependency_notice_pins":notices,
            "legal_clearance_established":False,"all_file_rights_reviewed":False,
            "rights_review":"REVIEW_REQUIRED","independent_evaluation":"NOT_RUN","model_selection_permitted":False}
    (out/"module-scope-summary.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps(report,sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("archive","root","out"):p.add_argument("--"+n,type=Path,required=True)
    run(**vars(p.parse_args()))
