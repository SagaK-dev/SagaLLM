import copy
import json
import pytest
from sagallm.java_matched_reports import compare, canonical_sha, load_report, normalize
from sagallm.java_independent_benchmark import SCHEMA


def fake(checkpoint, case=("PASS","FAIL")):
    f={"suite_sha256":"a"*64,"oci_image":"test@sha256:"+"b"*64,
       "machine_status":"PREFLIGHT_ONLY","declared_lineage_consistent":True,
       "model_selection_permitted":False,"independence_proven":False,
       "training_lineage_verified":False,"generation_provenance_verified":False,
       "freeze_precedes_inference_verified":False,"freeze_sha256":"c"*64,
       "checkpoint_sha256":checkpoint*64,"tokenizer_sha256":"d"*64,"evaluator_sha256":"e"*64,
       "generation":{"seed":1,"max_new_tokens":170,"decoding":"greedy","prompt_sha256":"f"*64,
                     "generation_code_sha_declared":"a"*40}}
    status="UNAVAILABLE" if any(s in ("NOT_RUN","RUNTIME_ERROR") for s in case) else "PASS" if all(s=="PASS" for s in case) else "FAIL"
    rows=[{"id":f"task_{i}","status":status,"candidate_sha256":"b"*64,
           "case_statuses":list(case),"cases_passed":case.count("PASS"),"cases_total":len(case)} for i in range(5)]
    partial=status=="UNAVAILABLE"
    return {"protocol":SCHEMA,"suite_sha256":"a"*64,"benchmark_independence_proven":False,
            "review_metadata_verified_externally":False,"model_selection":"NOT_PERMITTED_NO_INDEPENDENCE_PROOF",
            "behavioral_status":"PARTIAL_OR_UNAVAILABLE" if partial else "EVALUATED",
            "review_gate":"ELIGIBLE_SELF_ATTESTED","vm_review_gate":"ELIGIBLE_SELF_ATTESTED",
            "vm_review_metadata_verified_externally":False,"model_freeze_gate":"PREFLIGHT_ONLY",
            "source_persisted":False,"raw_stdout_stderr_persisted":False,"vm_review_sha256":"1"*64,
            "candidate_freeze_sha256":"2"*64,"oci_image":f["oci_image"],"model_freeze":f,
            "model_freeze_report_canonical_sha256":canonical_sha(f),"task_count":5,"io_case_count":10,
            "results":rows,"behavior_passed_tasks":None if partial else 5 if status=="PASS" else 0,
            "oci_execution_attempted":True,"java_executed":"POSSIBLE_IN_OCI_ONLY"}


def test_matched_complete():
    out=compare(fake("a"),fake("b",("PASS","PASS")),"1"*64,"2"*64)
    assert out["status"]=="MATCHED_DESCRIPTIVE_COMPLETE"
    assert out["delta"]==5 and out["paired_available_counts"]["right_only_pass"]==5
    assert not out["production_promotion"] and not out["model_selection_permitted"]


def test_unavailable_not_failure():
    out=compare(fake("a"),fake("b",("PASS","RUNTIME_ERROR")),"1"*64,"2"*64)
    assert out["status"]=="MATCHED_PARTIAL_UNAVAILABLE"
    assert out["jointly_available_cases"]==5 and out["delta"] is None
    assert out["left_fully_correct_tasks"] is None and out["right_status_counts"]["RUNTIME_ERROR"]==5


def test_missing_candidate():
    r=fake("b")
    r["results"][0]={"id":"task_0","status":"CANDIDATE_MISSING","cases_passed":0,"cases_total":2}
    r.update(behavioral_status="PARTIAL_OR_UNAVAILABLE",behavior_passed_tasks=None)
    out=compare(fake("a"),r,"1"*64,"2"*64)
    assert out["unavailable_pairs"]==2 and out["delta"] is None


def test_not_run():
    r={"protocol":SCHEMA,"suite_sha256":"a"*64,"benchmark_independence_proven":False,
       "review_metadata_verified_externally":False,"model_selection":"NOT_PERMITTED",
       "behavioral_status":"NOT_RUN","java_executed":False,"behavior_passed_tasks":None}
    assert compare(r,fake("a"),"1"*64,"2"*64)["status"]=="NOT_COMPARABLE"


@pytest.mark.parametrize("key",["tokenizer_sha256","evaluator_sha256","generation"])
def test_unmatched(key):
    r=fake("b")
    if key=="generation":r["model_freeze"][key]["seed"]=2
    else:r["model_freeze"][key]="3"*64
    r["model_freeze_report_canonical_sha256"]=canonical_sha(r["model_freeze"])
    with pytest.raises(ValueError,match="Unmatched"):
        compare(fake("a"),r,"1"*64,"2"*64)


@pytest.mark.parametrize("kind",["count","order","freeze","review","suite","same_model","promote","case"])
def test_fail_closed(kind):
    r=fake("b")
    if kind=="count":r["results"][0]["cases_passed"]=2
    elif kind=="order":r["results"].reverse()
    elif kind=="freeze":r["model_freeze_report_canonical_sha256"]="0"*64
    elif kind=="review":r["vm_review_sha256"]="3"*64
    elif kind=="suite":r["suite_sha256"]="9"*64
    elif kind=="same_model":r=fake("a")
    elif kind=="promote":r["benchmark_independence_proven"]=True
    else:r["results"][0]["case_statuses"][0]="INVENTED"
    with pytest.raises(ValueError):
        compare(fake("a"),r,"1"*64,"2"*64)


def test_exact_report_bytes(tmp_path):
    from hashlib import sha256
    p=tmp_path/"report.json"
    raw=json.dumps(fake("a")).encode()
    p.write_bytes(raw)
    assert load_report(p,sha256(raw).hexdigest())["task_count"]==5
    p.write_bytes(raw+b" ")
    with pytest.raises(ValueError,match="SHA"):
        load_report(p,sha256(raw).hexdigest())
