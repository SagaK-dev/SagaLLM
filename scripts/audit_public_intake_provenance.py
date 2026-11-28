"""Map pinned public intake cases to upstream canonical data and recompute I/O oracles.

Read metadata only. No generated Java, inference or independent benchmark claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.audit_ranking_auxiliary import require, parse, pin, sha
from scripts.audit_legacy_run_discovery import PUBLIC_SUITE
from scripts.audit_unarchived_artifact_triage import read_evidence, gate

UPSTREAM="4812c213a2647e2034ad5ac78155ce9265cdd171"
LICENSE_BLOB="6813d6298df9a46da6485050e8a5fcd61ab79879"
UNIFIED_ZIP="542526f1e8ff7a995e41ffb0a1b2dd1f71672bc7ed9430c0675e9266a01d4c2c"
SPECS=(
 ("ex_hamming","hamming","93b438b75f2953b154b407eb73d3f63dadbe9ef9",4),
 ("ex_rna_transcription","rna-transcription","d05146ed00c9af5eaeedf7a1fb2292e305420ed9",4),
 ("ex_luhn","luhn","78174b9128a1666ce43190eaafb6b00cd3c33479",5),
 ("ex_roman_numerals","roman-numerals","1edb968d321bc4b9473e446a5d3f1daa173a4d56",4),
 ("ex_allergies","allergies","8fcc359a472dcd70773663e0b039a1d75796d123",5),
 ("ex_atbash_cipher","atbash-cipher","984ea2d6be61592eaa54e3cd434f47646db8bb1d",5),
)


def blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def bounded(path, expected):
    require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=200000,
            "Unsafe upstream path")
    raw=path.read_bytes()
    require(blob(raw)==expected,"Upstream Git blob pin mismatch")
    return raw


def flatten(cases, depth=0):
    require(depth<=12 and isinstance(cases,list),"Invalid canonical case tree")
    out=[]
    for c in cases:
        require(isinstance(c,dict),"Invalid canonical case")
        if "cases" in c:
            out.extend(flatten(c["cases"],depth+1))
        else:
            out.append(c)
        require(len(out)<=1000,"Canonical case limit")
    return out


def adapt(slug,c):
    a=c["input"]; e=c["expected"]
    if type(e) not in (str,int,bool):
        return None
    if slug=="hamming" and c["property"]=="distance":
        inp=a["strand1"]+"\n"+a["strand2"]
    elif slug=="rna-transcription" and c["property"]=="toRna":
        inp=a["dna"]
    elif slug=="luhn" and c["property"]=="valid":
        inp=a["value"]
    elif slug=="roman-numerals" and c["property"]=="roman":
        inp=str(a["number"])
    elif slug=="allergies" and c["property"]=="allergicTo":
        inp=str(a["score"])+" "+a["item"]
    elif slug=="atbash-cipher" and c["property"]=="encode":
        inp=a["phrase"]
    else:
        return None
    if not isinstance(inp,str):
        return None
    return {"stdin":inp+"\n","stdout":(str(e).lower() if type(e) is bool else str(e))+"\n"}


def oracle(slug,stdin):
    require(isinstance(stdin,str) and stdin.endswith("\n"),"Invalid stdin framing")
    line=stdin[:-1]
    if slug=="hamming":
        a,b=line.split("\n")
        require(len(a)==len(b),"Unequal DNA strands")
        out=str(sum(x!=y for x,y in zip(a,b)))
    elif slug=="rna-transcription":
        require(set(line)<=set("ACGT"),"Unexpected DNA alphabet")
        out=line.translate(str.maketrans("ACGT","UGCA"))
    elif slug=="luhn":
        digits=line.replace(" ","")
        valid=len(digits)>=2 and all("0"<=c<="9" for c in digits)
        if valid:
            total=0
            for i,c in enumerate(reversed(digits)):
                n=int(c)*(2 if i%2 else 1)
                total+=n-9 if n>9 else n
            valid=total%10==0
        out=str(valid).lower()
    elif slug=="roman-numerals":
        n=int(line);require(1<=n<=3999,"Out-of-range Roman numeral")
        parts=[]
        for v,s in ((1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),(100,"C"),(90,"XC"),
                    (50,"L"),(40,"XL"),(10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")):
            q,n=divmod(n,v);parts.append(s*q)
        out="".join(parts)
    elif slug=="allergies":
        score,item=line.split()
        bits={k:1<<i for i,k in enumerate(("eggs","peanuts","shellfish","strawberries",
                "tomatoes","chocolate","pollen","cats"))}
        out=str(bool(int(score)&bits[item])).lower()
    elif slug=="atbash-cipher":
        encoded=[]
        for c in line:
            if "A"<=c<="Z": c=c.lower()
            if "a"<=c<="z": encoded.append(chr(ord("z")-(ord(c)-ord("a"))))
            elif "0"<=c<="9": encoded.append(c)
        out=" ".join("".join(encoded[i:i+5]) for i in range(0,len(encoded),5))
    else:
        raise ValueError("Unknown task")
    return out+"\n"


def audit(suite,upstream,notice,unified):
    gate(unified)
    require(unified["status"]=="SCOPED_MODEL_STATE_AND_DEVELOPMENT_INPUTS_INTEGRATED"
            and unified["public_development_scope"]["source_sha256"]==PUBLIC_SUITE,
            "Unified source binding mismatch")
    d=parse(pin(suite,PUBLIC_SUITE))
    require(d["provenance_url"]==f"https://github.com/exercism/problem-specifications/tree/{UPSTREAM}/exercises"
            and d["license_id"]=="NOT_REVIEWED" and d["reviewers"]==[]
            and d["review_status"]=="PENDING_HUMAN_REVIEW","Review gate changed")
    lic=bounded(upstream/"LICENSE",LICENSE_BLOB)
    require(b"The MIT License (MIT)" in lic and b"Copyright (c) 2014, 2019, 2021 Exercism" in lic,
            "Unexpected license notice")
    require(bounded(notice,LICENSE_BLOB)==lic,"Attribution notice mismatch")
    require([t["id"] for t in d["tasks"]]==[x[0] for x in SPECS],"Task ID mismatch")
    rows=[]
    for t,(task,slug,expected,count) in zip(d["tasks"],SPECS):
        raw=bounded(upstream/(slug+".json"),expected)
        u=parse(raw)
        require(u["exercise"]==slug and len(t["cases"])==count,"Canonical exercise/count mismatch")
        leaves=flatten(u["cases"])
        adapted=[(c.get("uuid"),adapt(slug,c)) for c in leaves]
        for i,c in enumerate(t["cases"]):
            found=[uid for uid,io in adapted if io==c]
            require(len(found)==1 and isinstance(found[0],str),"Unmatched or ambiguous canonical case")
            require(oracle(slug,c["stdin"])==c["stdout"],"Reference oracle disagreement")
            rows.append({"task_id":task,"case_index":i,"upstream_exercise":slug,
                         "upstream_case_uuid":found[0],"upstream_git_blob":expected,
                         "case_sha256":sha(json.dumps(c,sort_keys=True,separators=(",",":")).encode()),
                         "oracle_recomputed_match":True})
    require(len(rows)==len({r["upstream_case_uuid"] for r in rows})==27,"Case coverage mismatch")
    return {"schema":"sagallm-public-intake-provenance-v1",
            "status":"PINNED_PUBLIC_CASE_PROVENANCE_AND_ORACLES_MATCH",
            "upstream_revision":UPSTREAM,"upstream_license_git_blob":LICENSE_BLOB,
            "observed_upstream_license":"MIT","license_notice_sha256":sha(lic),
            "attribution_notice_exact_match":True,"suite_sha256":PUBLIC_SUITE,
            "canonical_exercises":6,"matched_cases":27,"oracle_matches":27,"case_bindings":rows,
            "prompt_authorship_reviewed":False,"license_scope_reviewed_by_human":False,
            "training_overlap_reviewed":False,"upstream_license_observation_not_legal_clearance":True,
            "actual_exposure_verified":False,"training_lineage_verified":False,
            "full_exposure_inventory":"NOT_ESTABLISHED","rights_review":"REVIEW_REQUIRED",
            "independent_evaluation":"NOT_RUN","model_selection_permitted":False,
            "new_training":False,"new_inference":False,"new_java_execution":False}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("suite","upstream","notice","unified-zip"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    d,_=read_evidence(a.unified_zip,UNIFIED_ZIP,["scoped-unified-inventory.json","unified-model-state-scope.json"])
    print(json.dumps(audit(a.suite,a.upstream,a.notice,d["unified-model-state-scope.json"]),sort_keys=True,indent=2))
