"""Staged OCI result contract: mocked launcher and fixture reports only."""
import pytest
from sagallm import java_behavior as b
from sagallm.java_matched_reports import compare, normalize
from test_java_matched_reports import fake
from test_java_independent_benchmark import _attested, _save, _mock_vm_review
from sagallm import java_independent_benchmark as bench
import hashlib


def test_static_wrapper_and_security_flags(tmp_path):
    args=b.build_oci_command("docker",image="example@sha256:"+"a"*64,
        mount_dir=tmp_path/"source",cid_file=tmp_path/"cid",
        stdin_file_present=True,staged=True)
    cmd=args[-1]
    assert "javac -proc:none --release 17" in cmd
    assert ">/dev/null" in cmd and "exit 40" in cmd and "exit 43" in cmd
    assert "exit 41" in cmd and " < /input/stdin.txt" in cmd
    for flag in ("--network=none","--read-only","--pull=never","--cap-drop=ALL",
                 "--user=65534:65534","--security-opt=no-new-privileges",
                 "--pids-limit=64","--memory=384m","--log-driver=none"):
        assert flag in args


@pytest.mark.parametrize("code,output,status,match",[
    (0,b"ok\n","PASS",True),(0,b"wrong","OUTPUT_MISMATCH",False),
    (40,b"ok\n","COMPILE_FAIL",False),(43,b"","COMPILE_UNAVAILABLE",None),
    (41,b"ok\n","JAVA_NONZERO_EXIT",False),(42,b"","RUNTIME_ERROR",None),
    (125,b"","RUNTIME_ERROR",None),(126,b"","RUNTIME_ERROR",None),
    (127,b"","RUNTIME_ERROR",None),(137,b"","RUNTIME_ERROR",None),
])
def test_exit_mapping(monkeypatch,code,output,status,match):
    monkeypatch.setattr(b,"oci_image_ready",lambda *a:True)
    monkeypatch.setattr(b,"_bounded_capture",lambda *a:(code,output))
    r=b.run_java_in_oci("public class Main {}", "ok\n",allow_execution=True,staged=True)
    assert r["status"]==status and r["behavior_match"] is match
    assert "stdout" not in r and "stderr" not in r


@pytest.mark.parametrize("err,status",[(TimeoutError,"TIMEOUT"),(OverflowError,"OUTPUT_LIMIT"),(OSError,"RUNTIME_ERROR")])
def test_outer_limit_never_infers_compile(monkeypatch,err,status):
    monkeypatch.setattr(b,"oci_image_ready",lambda *a:True)
    def fail(*a):raise err()
    monkeypatch.setattr(b,"_bounded_capture",fail)
    monkeypatch.setattr(b,"_cleanup_stale_container",lambda *a:None)
    r=b.run_java_in_oci("public class Main {}", "",allow_execution=True,staged=True)
    assert r["status"]==status and b.compile_phase_counts([status])["unknown"]==1


def test_no_opt_in_no_engine(monkeypatch):
    def forbidden(*a,**k):raise AssertionError("Engine accessed")
    monkeypatch.setattr(b,"oci_image_ready",forbidden)
    assert b.run_java_in_oci("public class Main {}","",staged=True)["status"]=="NOT_RUN"


def staged_report(checkpoint,cases):
    r=fake(checkpoint,("PASS","PASS"))
    for row in r["results"]:
        row["case_statuses"]=list(cases)
        row["cases_passed"]=cases.count("PASS")
        row["status"]="UNAVAILABLE" if any(x in b.STAGED_UNAVAILABLE for x in cases) else "PASS" if all(x=="PASS" for x in cases) else "FAIL"
    partial=any(x in b.STAGED_UNAVAILABLE for x in cases)
    r["behavioral_status"]="PARTIAL_OR_UNAVAILABLE" if partial else "EVALUATED"
    r["behavior_passed_tasks"]=None if partial else (5 if all(x=="PASS" for x in cases) else 0)
    r["execution_protocol"]=b.STAGED_EXECUTION_PROTOCOL
    r["compile_case_counts"]=b.compile_phase_counts(cases*5)
    return r


def test_matched_staged_counts_and_delta():
    l=staged_report("a",("COMPILE_FAIL","JAVA_NONZERO_EXIT"))
    r=staged_report("b",("PASS","OUTPUT_MISMATCH"))
    out=compare(l,r,"1"*64,"2"*64)
    assert out["delta"]==5 and out["compile_status"]=="PER_CASE_STAGE_COUNTS"
    assert out["left_compile_case_counts"]["failed"]==5
    assert out["right_compile_case_counts"]["passed"]==10


def test_incomplete_compile_and_legacy_mix():
    l=staged_report("a",("COMPILE_UNAVAILABLE","TIMEOUT"))
    r=staged_report("b",("PASS","PASS"))
    out=compare(l,r,"1"*64,"2"*64)
    assert out["delta"] is None and out["unavailable_pairs"]==5
    with pytest.raises(ValueError,match="Unmatched execution protocol"):
        compare(fake("a"),r,"1"*64,"2"*64)
    l["compile_case_counts"]["passed"]=5
    with pytest.raises(ValueError,match="Compile phase totals"):
        normalize(l)


def test_staged_no_legacy_fail():
    r=staged_report("b",("PASS","PASS"))
    r["results"][0]["case_statuses"][0]="FAIL"
    with pytest.raises(ValueError,match="Invalid case"):
        normalize(r)


def test_actual_evaluation_path_mixed_cases(tmp_path,monkeypatch):
    _mock_vm_review(monkeypatch)
    data=_attested()
    suite=_save(tmp_path,data)
    root=tmp_path/"generated";root.mkdir()
    for task in data["tasks"]:
        (root/(task["id"]+".java")).write_text("public class Main {}")
    monkeypatch.setattr(bench,"oci_image_ready",lambda *a:True)
    statuses=iter(["COMPILE_FAIL","JAVA_NONZERO_EXIT","OUTPUT_MISMATCH"]*5)
    def run(*a,**k):
        assert k["staged"] is True
        return {"status":next(statuses)}
    monkeypatch.setattr(bench,"run_java_in_oci",run)
    report=bench.evaluate_suite(suite,root,allow_execution=True,disposable_vm_acknowledged=True,
        expected_suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
        candidate_freeze_path=tmp_path/"mock.json",expected_candidate_freeze_sha256="a"*64)
    assert report["execution_protocol"]==b.STAGED_EXECUTION_PROTOCOL
    assert report["compile_case_counts"]=={"passed":10,"failed":5,"unavailable":0,"not_run":0,"unknown":0}
    assert report["behavior_passed_tasks"]==0
