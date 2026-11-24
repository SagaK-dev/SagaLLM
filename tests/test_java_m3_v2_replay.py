from types import SimpleNamespace
import pytest
from scripts import audit_java_m3_v2 as a
from scripts import evaluate_java_m3_v2 as e
from sagallm.java_behavior import build_oci_command, run_java_in_oci


def test_deterministic_collision_bound():
    rows=[{"id":str(i),"source_sha256":"a"*64} for i in range(6)]
    tasks=[{"id":str(i),"cases":[{"stdin":str(j),"stdout":str(i)} for j in range(3)]}
           for i in range(6)]
    r=a.deterministic_ceiling(rows,tasks)
    assert r["conditional_correct_case_upper_bound"]==3
    assert r["observed_passes"] is None
    with pytest.raises(ValueError):
        a.deterministic_ceiling(rows,tasks[:-1])


def args(**kw):
    return SimpleNamespace(**({"allow_execution":False,"acknowledge_disposable_vm":False,
        "review":None,"review_sha256":None,"image":None}|kw))


def test_gate_default_and_partial():
    assert e.gate(args()) is None
    for v in (args(allow_execution=True),args(acknowledge_disposable_vm=True),
              args(image="sha256:"+"a"*64)):
        with pytest.raises(PermissionError):
            e.gate(v)


def test_ci_refusal(monkeypatch,tmp_path):
    monkeypatch.setenv("GITHUB_ACTIONS","true")
    with pytest.raises(PermissionError):
        e.gate(args(allow_execution=True,acknowledge_disposable_vm=True,
                    review=tmp_path/"review",review_sha256="a"*64,image="sha256:"+"b"*64))


def test_compile_only_never_invokes_java(tmp_path,monkeypatch):
    cmd=build_oci_command("docker",image="sha256:"+"a"*64,
                          mount_dir=tmp_path/"s",cid_file=tmp_path/"c",compile_only=True)
    assert "javac -proc:none" in cmd[-1] and "&& java " not in cmd[-1]
    assert "--network=none" in cmd and "--read-only" in cmd
    assert run_java_in_oci("class Main {}","",compile_only=True)["status"]=="NOT_RUN"
    monkeypatch.setattr("sagallm.java_behavior.oci_image_ready",lambda *args:True)
    monkeypatch.setattr("sagallm.java_behavior._bounded_capture",lambda cmd,t:(0,b""))
    r=run_java_in_oci("class Main {}","",compile_only=True,allow_execution=True)
    assert r["status"]=="PASS" and r["behavior_match"] is None


def test_compile_fail_skips_runtime():
    calls=[]
    def runner(*args,**kw):
        calls.append(kw)
        return {"status":"FAIL"}
    task={"id":"x","cases":[{"stdin":"a","stdout":"b"}]*3}
    result=e.grade("public class Main {}",task,"image",runner)
    assert len(calls)==1 and calls[0]["compile_only"]
    assert result["cases"]==["COMPILE_FAILED"]*3


def test_zero_exit_mismatch():
    def runner(*args,**kw):
        return {"status":"PASS"} if kw.get("compile_only") else {"status":"FAIL","returncode":0}
    task={"id":"x","cases":[{"stdin":"a","stdout":"b"}]*3}
    result=e.grade("public class Main {}",task,"image",runner)
    assert result["compile"]=="PASS" and result["cases"]==["OUTPUT_MISMATCH_ZERO_EXIT"]*3


def test_archive_pin_rejects_tamper(tmp_path):
    p=tmp_path/"bad.zip"
    p.write_bytes(b"bad")
    with pytest.raises(ValueError,match="SHA"):
        a.audit_archive(p)
