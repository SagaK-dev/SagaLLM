"""Preflight verified algorithm references against pinned SFT input and tokenizer.
No model loading/training or heldout scoring; no new corpus publication.
"""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path
from collections import Counter

from sagallm.java_algorithm_curriculum import rows,cases,oracle,canonical,sha
from sagallm.java_instruction_sft import source_task,encode_row,sha_file,compact_prompt
from sagallm.java_ai import _model_user_message
from sagallm.tokenization import load_tokenizer
from sagallm.java_diverse_corpus import build_java_diverse_corpus,verify_java_diverse_corpus
from scripts.audit_canonical_sft_template_quality import signature
from scripts.run_pinned_traced_context_pilot import DATA,materialize_parent

REPORT_ZIP="32c05a5bebd649bd4043274c6fb3dafda0b5fd336879fe563d2d0998c6a907fd"
REPORT_SHA="2622545876aec9d04e962521179de05a9656fc407d06e9d8b07cb41b21cd253d"

def require(ok,msg):
    if not ok:raise ValueError(msg)

def verify_report(path):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size==6349 and sha_file(path)==REPORT_ZIP,"Reference artifact drift")
    with zipfile.ZipFile(path) as z:
        require(z.namelist()==["algorithm-curriculum-report.json"],"Archive members drift")
        info=z.infolist()[0]
        require(info.file_size==41960 and not info.is_dir() and not info.flag_bits&1,"Unexpected report member")
        raw=z.read(info)
    require(sha(raw)==REPORT_SHA,"Reference report digest mismatch")
    report=json.loads(raw)
    require(report["status"]=="FIXED_REFERENCE_CURRICULUM_VERIFIED" and report["reference_compiles"]==8
            and report["oracle_comparisons"]==96 and report["training_performed"] is False
            and report["heldout_role_assigned"] is False,"Reference scope drift")
    records=list(rows())
    expected=[{"id":r["id"],"family":r["family"],"variant":r["variant"],"language":r["language"],
               "task_sha256":sha(r["task"].encode()),"source_sha256":sha(r["source"].encode()),
               "row_sha256":sha(canonical(r))} for r in records]
    require(report["record_index"]==expected,"Reference row digest drift")
    seen=set()
    for r in report["reference_results"]:
        key=(r["family"],r["variant"])
        require(key not in seen,"Duplicate implementation");seen.add(key)
        match=[x for x in records if (x["family"],x["variant"])==key]
        require(len(match)==2 and r["source_sha256"]==sha(match[0]["source"].encode()) and r["compile"]=="PASS","Source binding mismatch")
        expected_checks=[]
        for i,stdin in enumerate(cases(key[0])):
            h=sha(oracle(key[0],stdin).encode())
            expected_checks.append({"case":i,"stdin_sha256":sha(stdin.encode()),"expected_stdout_sha256":h,"actual_stdout_sha256":h})
        require(r["checks"]==expected_checks,"Oracle evidence drift")
    require(seen=={(r["family"],r["variant"]) for r in records},"Implementation coverage")
    return records

def canonical_index(dataset):
    index = {}
    for split in ("train", "validation", "test"):
        values = {"task": set(), "source": set(), "signature": set()}
        with (dataset / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                task, source = source_task(json.loads(line))
                values["task"].add(sha(task.encode()))
                values["source"].add(sha(source.encode()))
                values["signature"].add(signature(source))
        index[split] = values
    return index


def sft_candidate(record, prefix):
    return {
        "id": prefix + record["id"],
        "family": record["family"],
        "messages": [
            {"role": "user", "content": _model_user_message(record["task"], "public class Main {}", "Main")},
            {"role": "assistant", "content": record["source"]},
        ],
    }


def run(reference_zip,parent_zip):
    records=verify_report(reference_zip)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);materialize_parent(parent_zip,root)
        tokenizer=load_tokenizer(root/"tokenizer.json")
        dataset=root/"canonical";build_java_diverse_corpus(dataset)
        verify_java_diverse_corpus(dataset)
        require(sha_file(dataset/"manifest.json")==DATA,"Canonical pin drift")
        old = canonical_index(dataset)
        results=[]
        for r in records:
            candidate = sft_candidate(r, "algorithm-v1-")
            task,source=source_task(candidate)
            require(task==r["task"] and source==r["source"],"SFT conversion drift")
            require(tokenizer.decode(tokenizer.encode(compact_prompt(task)))==compact_prompt(task)
                    and tokenizer.decode(tokenizer.encode(source))==source,"Tokenizer roundtrip mismatch")
            info={"id":candidate["id"],"family":r["family"],"variant":r["variant"],"language":r["language"],
                  "candidate_row_sha256":sha(canonical(candidate)),"task_sha256":sha(task.encode()),"source_sha256":sha(source.encode()),
                  "lexical_signature":signature(source),"prompt_tokens":len(tokenizer.encode(compact_prompt(task))),
                  "response_tokens_without_eos":len(tokenizer.encode(source)),
                  "admitted_contexts":[n for n in (192,256,384,512,768) if encode_row(candidate,tokenizer,n) is not None],
                  "canonical_matches":{s:{k:v in idx[k] for k,v in (("task",sha(task.encode())),("source",sha(source.encode())),("signature",signature(source)))} for s,idx in old.items()}}
            results.append(info)
        return {"schema":"sagallm-algorithm-admission-audit-v1","status":"VERIFIED_REFERENCE_SFT_PREFLIGHT_COMPLETE",
                "reference_artifact_id":15257146273,"reference_report_sha256":REPORT_SHA,"canonical_manifest_sha256":DATA,
                "tokenizer_sha256":sha_file(root/"tokenizer.json"),"candidate_records":len(results),
                "unique_reference_sources":len({r["source_sha256"] for r in results}),
                "admission_by_context":{str(n):sum(n in r["admitted_contexts"] for r in results) for n in (192,256,384,512,768)},
                "language_counts":dict(Counter(r["language"] for r in results)),
                "canonical_match_counts":{s:{k:sum(r["canonical_matches"][s][k] for r in results) for k in ("task","source","signature")} for s in old},
                "candidate_membership_sha256":sha(canonical(sorted(r["candidate_row_sha256"] for r in results))),
                "rows":results,"semantic_overlap_verified":False,"test_tokenization_or_scoring":False,
                "training_performed":False,"dataset_published":False,"weights_loaded":False,
                "independent_evaluation":"NOT_RUN","rights_review":"REVIEW_REQUIRED","model_selection_permitted":False}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("reference-zip","parent-zip","output"):p.add_argument("--"+k,type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"Refuse overwrite")
    report=run(a.reference_zip,a.parent_zip)
    with a.output.open("x") as f:json.dump(report,f,sort_keys=True,indent=2);f.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},sort_keys=True))
