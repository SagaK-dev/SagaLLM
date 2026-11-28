"""File-level review queue for the exact historical public Java corpus.

Root license pins, notice paths and header signals are evidence, not a legal
determination. Do not print or upload Java source or notice text.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

ROOT_ZIP="e64f49de36921c07c4db543451e543e93b128567d3311dfea0bb4d1243917769"
MANIFEST="9e11217e9bd47d836a680a39e0820cd02f159a62e8e9f2c4cd864dd64ae7bf11"
SPECS={
 "algorithms":("f4d6708297f86629a9e415070935b903ea8b1858","LICENSE","MIT","f6bcf04e7773b3bfca7a30836672dc7322e38944"),
 "gson":("d9b5079eaa4567219771bba6d04289cf45c7589b","LICENSE","Apache-2.0","d645695673349e3947e8e5ae42332d0ac3164cd7"),
 "guava":("36e3995499d6c3b74b1207d362fa9e19915d2589","LICENSE","Apache-2.0","d645695673349e3947e8e5ae42332d0ac3164cd7"),
 "patterns":("4cabb204f240240bea319605a5ffa46ded728001","LICENSE.md","MIT","bcae7daadee982a10f5f9497f493eb4d70cd517f"),
}
NOTICE=re.compile(r"^(?:licen[cs]e|copying|notice|copyright|third[-_ ]?party[-_ ]?notices?)(?:[._-].*)?$",re.I)
SPDX=re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9.+() -]{1,120})")
SPLITS={"train":2468,"validation":254,"test":204}

def require(ok,msg):
    if not ok:raise ValueError(msg)

def sha(b):return hashlib.sha256(b).hexdigest()

def blob(b):return hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()

def git(root,*args):
    return subprocess.check_output(["git","-C",str(root),*args],timeout=60)

def tree(root,rev):
    require(git(root,"rev-parse","HEAD").decode().strip()==rev,"Pinned checkout mismatch")
    out={}
    for item in git(root,"ls-tree","-r","-z",rev).split(b"\0"):
        if not item:continue
        meta,name=item.split(b"\t",1)
        mode,kind,digest=meta.decode().split()
        name=name.decode("utf8")
        require(name not in out and not PurePosixPath(name).is_absolute()
                and ".." not in PurePosixPath(name).parts,"Unsafe tree path")
        if kind=="blob":out[name]=(mode,digest)
    return out

def verified_file(root,path,entry,limit):
    p=root/path
    require(entry[0] in ("100644","100755") and p.is_file() and not p.is_symlink()
            and p.resolve().is_relative_to(root.resolve()) and 0<p.stat().st_size<=limit,
            "Unsafe tracked source or notice")
    raw=p.read_bytes()
    require(len(raw)<=limit and blob(raw)==entry[1],"Git blob bytes mismatch")
    return raw

def signals(text,license_id):
    head=text[:8192].splitlines()[:80]
    header="\n".join(head)
    # Stop at package/import to reduce incidental body matches.
    header=re.split(r"(?m)^\s*(?:package|import)\s+",header,maxsplit=1)[0]
    low=header.lower()
    ids=sorted({m.group(1).strip() for m in SPDX.finditer(header)})
    markers=[]
    for name,pattern in (("APACHE",r"apache license"),("MIT",r"mit license|permission is hereby granted"),
                         ("GPL_FAMILY",r"gnu (?:lesser |affero )?general public license|\b(?:a|l)?gpl[- ]"),
                         ("BSD",r"bsd license|redistribution and use in source"),
                         ("MPL",r"mozilla public license"),("EPL",r"eclipse public license"),
                         ("COPYRIGHT",r"\bcopyright\b")):
        if re.search(pattern,low):markers.append(name)
    mismatch=any(i!=license_id for i in ids) or any(m in markers for m in ("GPL_FAMILY","BSD","MPL","EPL")) or (
        "APACHE" in markers and license_id!="Apache-2.0") or ("MIT" in markers and license_id!="MIT")
    return {"spdx_header_values":ids,"header_markers":markers,
            "possible_non_root_license_signal":mismatch,
            "header_license_signal_present":bool(ids or set(markers)-{"COPYRIGHT"})}

def run(corpus,upstream,archive,out):
    require(archive.is_file() and not archive.is_symlink() and archive.stat().st_size==37938755
            and sha(archive.read_bytes())==ROOT_ZIP,"Original root archive mismatch")
    with zipfile.ZipFile(archive) as z:
        raw=z.read("java-corpus/manifest.json")
        require(sha(raw)==MANIFEST and (corpus/"manifest.json").read_bytes()==raw,"Reconstructed manifest mismatch")
        m=json.loads(raw)
        require(m["splits"]==SPLITS and len(m["sources"])==4,"Original split coverage")
        sources={x["name"]:x for x in m["sources"]}
        trees={};notices={};notice_rows=[]
        for slug,(rev,license_path,lic,lic_blob) in SPECS.items():
            r=upstream/slug
            t=tree(r,rev);trees[slug]=t
            spec=sources[slug]
            require(spec["revision"]==rev and spec["license_id"]==lic and spec["license_blob"]==lic_blob,
                    "Manifest source pin mismatch")
            license_raw=verified_file(r,license_path,t[license_path],512*1024)
            require(blob(license_raw)==lic_blob and sha(license_raw)==spec["license_sha256"]
                    and license_raw==z.read("java-corpus/"+spec["license_file"])
                    ==(corpus/spec["license_file"]).read_bytes(),"Root license byte mismatch")
            candidates=[]
            for p,entry in sorted(t.items()):
                if not NOTICE.fullmatch(PurePosixPath(p).name) or p.endswith(".java"):
                    continue
                if entry[0] not in ("100644","100755"):
                    notice_rows.append({"repo":slug,"path":p,"git_blob":entry[1],"status":"NONREGULAR_REVIEW_REQUIRED"})
                    continue
                nr=verified_file(r,p,entry,512*1024)
                candidates.append(p)
                notice_rows.append({"repo":slug,"path":p,"git_blob":entry[1],"sha256":sha(nr),
                                    "bytes":len(nr),"status":"NOTICE_CANDIDATE_REVIEW_REQUIRED"})
            notices[slug]=candidates
    rows=[];seen=set()
    for split,n in SPLITS.items():
        p=corpus/(split+".jsonl")
        require(p.is_file() and p.stat().st_size<=96_000_000,"Unsafe split")
        raw=p.read_bytes()
        require(sha(raw)==m["file_hashes"][split] and len(raw.splitlines())==n,"Split pin/count mismatch")
        for line in raw.splitlines():
            rec=json.loads(line);slug=rec["repo"];path=rec["path"]
            require(set(rec)=={"repo","path","group","sha256","text"} and (slug,path) not in seen,
                    "Historical record shape/identity mismatch")
            seen.add((slug,path))
            src=verified_file(upstream/slug,path,trees[slug][path],96000)
            normalized=src.decode("utf8").replace("\r\n","\n").replace("\r","\n")
            require(normalized==rec["text"] and sha(normalized.encode())==rec["sha256"],"Source text drift")
            parent=PurePosixPath(path).parent
            applicable=[p for p in notices[slug] if PurePosixPath(p).parent==parent
                        or PurePosixPath(p).parent in parent.parents]
            sig=signals(normalized,SPECS[slug][2])
            rows.append({"repo":slug,"revision":SPECS[slug][0],"path":path,"split":split,
                         "source_git_blob":trees[slug][path][1],"normalized_source_sha256":rec["sha256"],
                         "root_license_id":SPECS[slug][2],"root_license_git_blob":SPECS[slug][3],
                         "ancestor_notice_candidates":applicable,**sig,"rights_review":"REVIEW_REQUIRED"})
    require(len(rows)==sum(SPLITS.values()),"File coverage mismatch")
    out.mkdir()
    data=b"".join(json.dumps(r,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()+b"\n" for r in rows)
    (out/"public-source-file-rights-index.jsonl").write_bytes(data)
    (out/"public-notice-candidates.json").write_text(json.dumps(notice_rows,sort_keys=True,indent=2)+"\n")
    counts={}
    for slug in SPECS:
        subset=[r for r in rows if r["repo"]==slug]
        counts[slug]={"files":len(subset),"by_split":dict(Counter(r["split"] for r in subset)),
                      "notice_candidates":sum(r["repo"]==slug for r in notice_rows),
                      "header_license_signal_count":sum(r["header_license_signal_present"] for r in subset),
                      "possible_non_root_signal_count":sum(r["possible_non_root_license_signal"] for r in subset)}
    result={"schema":"sagallm-public-file-rights-review-index-v1",
            "status":"PINNED_PUBLIC_FILE_PROVENANCE_INDEX_COMPLETE",
            "source_revision_count":4,"source_file_count":len(rows),"split_counts":SPLITS,
            "root_license_byte_matches":4,"notice_candidate_count":len(notice_rows),
            "source_index_sha256":sha(data),"per_repository":counts,
            "possible_non_root_signal_count":sum(r["possible_non_root_license_signal"] for r in rows),
            "header_scan_is_heuristic":True,"all_file_rights_reviewed":False,
            "original_manifest_sha256":MANIFEST,"original_artifact_id":14738360825,
            "rights_review":"REVIEW_REQUIRED","full_exposure_inventory":"NOT_ESTABLISHED",
            "actual_exposure_verified":False,"training_lineage_verified":False,
            "independent_evaluation":"NOT_RUN","model_selection_permitted":False}
    (out/"rights-index-summary.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("corpus","upstream","archive","out"):
        p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args();run(**vars(a))
