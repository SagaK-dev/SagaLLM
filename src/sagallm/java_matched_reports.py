"""Compare pinned independent-evaluation reports offline, without Java or model load.

A structural comparison is not authentication, independence, or promotion.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import math

from .java_independent_benchmark import SCHEMA, SHA_PATTERN, ID_PATTERN, _unique_object
from .java_behavior import (STAGED_EXECUTION_PROTOCOL, STAGED_STATUSES,
                            STAGED_UNAVAILABLE, compile_phase_counts)

MAX_REPORT_BYTES=4*1024*1024
STATES={"PASS","FAIL","TIMEOUT","OUTPUT_LIMIT","REJECTED","NOT_RUN","RUNTIME_ERROR","CANDIDATE_MISSING"}
UNAVAILABLE={"NOT_RUN","RUNTIME_ERROR","CANDIDATE_MISSING"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(x):
    return isinstance(x,str) and SHA_PATTERN.fullmatch(x) is not None


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,
                                     separators=(",",":"),allow_nan=False).encode()).hexdigest()


def load_report(path: Path, expected: str):
    require(pin(expected) and path.is_file() and not path.is_symlink(),"Pinned report required")
    with path.open("rb") as fh:
        raw=fh.read(MAX_REPORT_BYTES+1)
    require(0<len(raw)<=MAX_REPORT_BYTES and hashlib.sha256(raw).hexdigest()==expected,"Report SHA or size mismatch")
    obj=json.loads(raw,object_pairs_hook=_unique_object)
    require(isinstance(obj,dict),"Expected report object")
    return obj


def normalize(report):
    require(report.get("protocol")==SCHEMA and pin(report.get("suite_sha256")),"Invalid protocol/suite")
    require(report.get("benchmark_independence_proven") is False
            and report.get("review_metadata_verified_externally") is False
            and report.get("model_selection") in ("NOT_PERMITTED","NOT_PERMITTED_NO_INDEPENDENCE_PROOF"),
            "Unsupported independence or promotion claim")
    status=report.get("behavioral_status")
    require(status in ("NOT_RUN","PARTIAL_OR_UNAVAILABLE","EVALUATED"),"Invalid behavior status")
    if status=="NOT_RUN":
        require(report.get("java_executed") is False and report.get("behavior_passed_tasks") is None
                and "results" not in report,"Inconsistent NOT_RUN")
        return None
    require(report.get("review_gate")=="ELIGIBLE_SELF_ATTESTED"
            and report.get("vm_review_gate")=="ELIGIBLE_SELF_ATTESTED"
            and report.get("vm_review_metadata_verified_externally") is False
            and report.get("model_freeze_gate")=="PREFLIGHT_ONLY"
            and report.get("source_persisted") is False and report.get("raw_stdout_stderr_persisted") is False,
            "Required machine gates missing")
    require(pin(report.get("vm_review_sha256")) and pin(report.get("candidate_freeze_sha256")),"Missing gate pin")
    f=report.get("model_freeze")
    require(isinstance(f,dict) and canonical_sha(f)==report.get("model_freeze_report_canonical_sha256"),"Freeze report hash mismatch")
    require(f.get("suite_sha256")==report["suite_sha256"] and f.get("oci_image")==report.get("oci_image")
            and f.get("machine_status")=="PREFLIGHT_ONLY" and f.get("declared_lineage_consistent") is True
            and f.get("model_selection_permitted") is False and f.get("independence_proven") is False
            and f.get("training_lineage_verified") is False and f.get("generation_provenance_verified") is False
            and f.get("freeze_precedes_inference_verified") is False,"Invalid freeze claims")
    require(all(pin(f.get(k)) for k in ("freeze_sha256","checkpoint_sha256","tokenizer_sha256","evaluator_sha256")),"Missing model pins")
    g=f.get("generation")
    require(isinstance(g,dict) and set(g)=={"seed","max_new_tokens","decoding","prompt_sha256","generation_code_sha_declared"}
            and type(g["seed"]) is int and 0<=g["seed"]<2**32
            and type(g["max_new_tokens"]) is int and 1<=g["max_new_tokens"]<=2048
            and g["decoding"]=="greedy" and pin(g["prompt_sha256"])
            and isinstance(g["generation_code_sha_declared"],str)
            and len(g["generation_code_sha_declared"])==40
            and all(c in "0123456789abcdef" for c in g["generation_code_sha_declared"]),"Invalid generation settings")
    execution_protocol=report.get("execution_protocol")
    require(execution_protocol in (None, STAGED_EXECUTION_PROTOCOL),"Unknown execution protocol")
    staged=execution_protocol==STAGED_EXECUTION_PROTOCOL
    states=STAGED_STATUSES|{"CANDIDATE_MISSING"} if staged else STATES
    unavailable_states=STAGED_UNAVAILABLE|{"CANDIDATE_MISSING"} if staged else UNAVAILABLE
    rows=report.get("results")
    require(isinstance(rows,list) and 5<=len(rows)<=12
            and report.get("task_count")==len(rows),"Invalid task coverage")
    tasks={}
    attempted=False
    for row in rows:
        tid=row.get("id")
        n=row.get("cases_total")
        require(isinstance(tid,str) and ID_PATTERN.fullmatch(tid) and tid not in tasks
                and type(n) is int and 2<=n<=6,"Invalid task or case count")
        if row.get("status")=="CANDIDATE_MISSING":
            require(set(row)=={"id","status","cases_passed","cases_total"} and row["cases_passed"]==0,"Malformed missing candidate")
            cases=["CANDIDATE_MISSING"]*n
        else:
            require(set(row)=={"id","status","candidate_sha256","case_statuses","cases_passed","cases_total"}
                    and pin(row.get("candidate_sha256")),"Missing candidate pin")
            cases=row["case_statuses"]
            require(isinstance(cases,list) and len(cases)==n
                    and all(isinstance(x,str) and x in states-{"CANDIDATE_MISSING"} for x in cases),"Invalid case statuses")
            expected=("PASS" if all(x=="PASS" for x in cases) else
                      "UNAVAILABLE" if any(x in unavailable_states for x in cases) else "FAIL")
            require(row["status"]==expected and row["cases_passed"]==cases.count("PASS"),"Task count/status mismatch")
            attempted=True
        tasks[tid]=cases
    require(sum(map(len,tasks.values()))==report.get("io_case_count"),"Total case count mismatch")
    unavailable=any(x in unavailable_states for cases in tasks.values() for x in cases)
    require(status==("PARTIAL_OR_UNAVAILABLE" if unavailable else "EVALUATED")
            and report.get("behavior_passed_tasks")== (None if unavailable else sum(all(x=="PASS" for x in c) for c in tasks.values()))
            and report.get("oci_execution_attempted") is attempted
            and report.get("java_executed")==("POSSIBLE_IN_OCI_ONLY" if attempted else False),"Aggregate mismatch")
    if staged:
        require(report.get("compile_case_counts")==compile_phase_counts(x for c in tasks.values() for x in c),"Compile phase totals mismatch")
    else:
        require("compile_case_counts" not in report,"Legacy report cannot establish compile phases")
    return {"tasks":tasks,"freeze":f,"status":status,"execution_protocol":execution_protocol,
            "states":states,"unavailable":unavailable_states}


def compare(left,right,left_sha,right_sha):
    l,r=normalize(left),normalize(right)
    result={"schema":"sagallm-java-matched-report-v1","left_report_sha256":left_sha,"right_report_sha256":right_sha,
            "benchmark_independence_proven":False,"generation_provenance_verified":False,
            "training_lineage_verified":False,"model_selection_permitted":False,"production_promotion":False,
            "compile_status":"NOT_DISTINGUISHABLE_IN_INPUT_REPORTS","new_java_execution":"NOT_RUN"}
    if l is None or r is None:
        return {**result,"status":"NOT_COMPARABLE","reason":"INPUT_BEHAVIOR_NOT_RUN","delta":None}
    require(l["execution_protocol"]==r["execution_protocol"],"Unmatched execution protocol")
    lf,rf=l["freeze"],r["freeze"]
    require(left["suite_sha256"]==right["suite_sha256"] and left["oci_image"]==right["oci_image"]
            and left["vm_review_sha256"]==right["vm_review_sha256"],"Unmatched suite, image or VM review")
    require(all(lf[k]==rf[k] for k in ("tokenizer_sha256","evaluator_sha256","generation")),"Unmatched evaluator/tokenizer/generation")
    require(list(l["tasks"])==list(r["tasks"]) and all(len(c)==len(r["tasks"][k]) for k,c in l["tasks"].items()),"Unmatched task or case coverage")
    require(lf["checkpoint_sha256"]!=rf["checkpoint_sha256"],"Expected distinct model checkpoints")
    pairs=[(a,b) for k,c in l["tasks"].items() for a,b in zip(c,r["tasks"][k])]
    shared=[(a,b) for a,b in pairs if a not in l["unavailable"] and b not in l["unavailable"]]
    def counts(index):
        return {s:sum(p[index]==s for p in pairs) for s in sorted(l["states"])}
    cells={"both_pass":sum(a==b=="PASS" for a,b in shared),
           "left_only_pass":sum(a=="PASS" and b!="PASS" for a,b in shared),
           "right_only_pass":sum(b=="PASS" and a!="PASS" for a,b in shared),
           "neither_pass":sum(a!="PASS" and b!="PASS" for a,b in shared)}
    fully_matched=len(shared)==len(pairs)
    if l["execution_protocol"]==STAGED_EXECUTION_PROTOCOL:
        result.update(compile_status="PER_CASE_STAGE_COUNTS",execution_protocol=STAGED_EXECUTION_PROTOCOL,
                      left_compile_case_counts=left["compile_case_counts"],right_compile_case_counts=right["compile_case_counts"])
    return {**result,"status":"MATCHED_DESCRIPTIVE_COMPLETE" if fully_matched else "MATCHED_PARTIAL_UNAVAILABLE",
            "suite_sha256":left["suite_sha256"],"left_checkpoint_sha256":lf["checkpoint_sha256"],
            "right_checkpoint_sha256":rf["checkpoint_sha256"],"total_cases":len(pairs),
            "jointly_available_cases":len(shared),"unavailable_pairs":len(pairs)-len(shared),
            "left_status_counts":counts(0),"right_status_counts":counts(1),"paired_available_counts":cells,
            "delta":cells["right_only_pass"]-cells["left_only_pass"] if fully_matched else None,
            "left_fully_correct_tasks":left["behavior_passed_tasks"] if fully_matched else None,
            "right_fully_correct_tasks":right["behavior_passed_tasks"] if fully_matched else None,
            "limitations":[("No compile/runtime separation in historical FAIL status" if not l["execution_protocol"]
                            else "Outer timeout/output limit or launcher failure does not establish a compile stage"),
                           "Case matching relies on exact suite hash and declared order",
                           "Reported freeze metadata and human reviews are not independently authenticated",
                           "Different model training exposures may confound comparison"]}


def compare_files(left,right,left_sha,right_sha):
    return compare(load_report(left,left_sha),load_report(right,right_sha),left_sha,right_sha)
