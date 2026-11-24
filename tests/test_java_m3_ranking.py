from pathlib import Path
import copy
import pytest
import torch
from scripts import train_java_m3_ranking as r

ROOT=Path(__file__).resolve().parents[1]


def test_unique_substitution_and_identical():
    assert r.substitution_positions([1,2,3],[1,9,3])==(1,)
    assert r.substitution_positions([1,2,3],[1,2,3])==()


def test_ambiguous_and_indel_excluded():
    assert r.substitution_positions([1,1],[1])==()
    assert r.substitution_positions([1,2,3],[1,3])==()
    assert r.substitution_positions([1,2],[1,9,2])==()


def test_alignment_bounds():
    with pytest.raises(ValueError):
        r.substitution_positions([], [1])
    with pytest.raises(ValueError):
        r.substitution_positions([1]*385,[2])


def test_hinge_gradient_sign_and_inactive():
    a=torch.tensor([0.3,0.1],requires_grad=True)
    b=torch.tensor([0.2,0.3],requires_grad=True)
    r.hinge(a,b).sum().backward()
    assert a.grad.tolist()==[1.0,0.0]
    assert b.grad.tolist()==[-1.0,0.0]


def test_shifted_aux_masks_exclude_eos():
    a={"prefix_ids":[1,5],"y":torch.tensor([-100,7,8,9,2])}
    b={"prefix_ids":[1,5,6],"y":torch.tensor([-100,-100,7,10,9,2])}
    rows,pos=r.make_aux(a,b,2)
    assert pos==(1,)
    assert rows[0]["y"].tolist()==[-100,-100,8,-100]
    assert rows[1]["y"].tolist()==[-100,-100,-100,8,-100]


def test_frozen_suite_and_oracle():
    rows=r.frozen_suite(ROOT/"examples/java_m3_development_holdback.json")
    assert len(rows)==6 and sum(len(x["cases"]) for x in rows)==18
    bad=copy.deepcopy(rows[0])
    bad["cases"][0]["stdout"]="0\n"
    with pytest.raises(ValueError,match="oracle"):
        r.oracle(bad)


def test_holdback_tamper(tmp_path):
    f=tmp_path/"suite.json"
    f.write_bytes((ROOT/"examples/java_m3_development_holdback.json").read_bytes()+b" ")
    with pytest.raises(ValueError,match="SHA"):
        r.frozen_suite(f)


def test_schedule_exact_exposure():
    broad=[{"id":str(i),"family":str(i%90)} for i in range(400)]
    pairs=[(str(2*i),str(2*i+1)) for i in range(160)]
    a=r.schedule_for(broad,pairs,4201)
    assert a==r.schedule_for(broad,pairs,4201)
    assert a!=r.schedule_for(broad,pairs,4202)
    assert len({v for b,_ in a for v in b})==320
    assert len({v for _,p in a for v in p})==320
    with pytest.raises(ValueError):
        r.schedule_for(broad,[pairs[0]]*160,4201)


def test_operation_spans_cover_every_canonical_train_row():
    from sagallm.tokenization import ByteBPETokenizer
    from sagallm.java_stdio_curriculum import expected_rows
    raw_rows=expected_rows("train")
    tok=ByteBPETokenizer.train([x["messages"][1]["content"] for x in raw_rows]+[r.compact_prompt(r.encode_row(x,ByteBPETokenizer(),1024)["task"]) for x in raw_rows],vocab_size=512)
    rows=[r.encode_row(x,tok,384) for x in raw_rows]
    assert all(x is not None for x in rows)
    aux,_,audit=r.pairs(rows,2,tokenizer=tok,mask_mode="operation-span-v2")
    assert audit["passed"] and audit["admitted_rows"]==320
    assert all(v==64 for v in audit["rows_per_pair_type"].values())
    assert all(v is not None for v in aux.values())


def test_operation_span_boundary_and_ambiguous_fail_closed():
    from scripts.java_m3_operation_masks import span_positions
    class Tok:
        SPECIAL={"<eos>":2}
        def decode(self,ids):
            return "".join({4:"x=",5:"a+b;",6:"\n"}[i] for i in ids)
    assert span_positions("x=a+b;\n","a+b",[4,5,6],Tok())==(1,)
    for source,operation,ids in (
        ("a+ba+b","a+b",[5,5]),("wrong","wrong",[4]),("x=a+b;\n","a+b",[4,2,5,6])
    ):
        with pytest.raises(ValueError):
            span_positions(source,operation,ids,Tok())


def test_operation_annotations_reject_validation_and_tampering():
    from scripts.java_m3_operation_masks import train_positions
    from sagallm.tokenization import ByteBPETokenizer
    from sagallm.java_stdio_curriculum import expected_rows
    tok=ByteBPETokenizer()
    raw=expected_rows("train")[0]
    row=r.encode_row(raw,tok,1024)
    assert train_positions(row,r.target(row,2),tok)
    for patch in ({"source":"modified"},{"task":"wrong"},{"id":"validation-stdin_add_offset-000"}):
        with pytest.raises(ValueError):
            train_positions({**row,**patch},r.target(row,2),tok)
