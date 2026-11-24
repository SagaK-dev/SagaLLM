from pathlib import Path
import pytest
import torch
from scripts import train_java_m3_parameter_ce as p

ROOT=Path(__file__).resolve().parents[1]


def test_suite_pin_and_oracle(tmp_path):
    f=ROOT/"examples/java_m3_parameter_holdback.json"
    rows=p.load_suite(f)
    assert len(rows)==6 and sum(len(r["cases"]) for r in rows)==18
    bad=tmp_path/"bad.json"
    bad.write_bytes(f.read_bytes()+b" ")
    with pytest.raises(ValueError,match="SHA"):
        p.load_suite(bad)


def test_auxiliary_ce_gradient():
    z=torch.zeros(2,2,requires_grad=True)
    p.auxiliary_loss(z).backward()
    assert z.grad.tolist()==[[-0.25,0.25],[0.25,-0.25]]
    assert p.auxiliary_loss(torch.tensor([[5.,0.],[0.,5.]])).item()<0.01
    with pytest.raises(ValueError):
        p.auxiliary_loss(torch.zeros(3,2))


def test_padded_logits_gather_last_real_token():
    class Model:
        def __call__(self,x):
            from types import SimpleNamespace
            return SimpleNamespace(logits=torch.stack([x.float(),-x.float(),x.float()*2],dim=-1))
    pair={"prefixes":[[1,4],[1,2,7]],"common_response_prefix":[9,5],"alternatives":[0,2]}
    assert p.pair_logits(Model(),pair).tolist()==[[5.,10.],[5.,10.]]


def test_coefficient_zero_has_zero_aux_gradient():
    x=torch.tensor(2.,requires_grad=True)
    logits=torch.zeros(2,2,requires_grad=True)
    (x*x+0.*p.auxiliary_loss(logits)).backward()
    assert x.grad.item()==4. and torch.count_nonzero(logits.grad)==0
