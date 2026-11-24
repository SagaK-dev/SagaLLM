import math
import pytest
import torch
from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from scripts import experiment_java_m3_prompt_bias as p


def model():
    torch.manual_seed(5)
    return SagaTransformer(ModelConfig(32,16,2,2,1,32,32)).eval()


def test_mask_boundary_and_causality():
    m=p.attention_mask(6,3,p.BIAS)
    assert m[0,0]==0 and m[1,1]==0
    assert m[2,0]==0 and m[2,1]==pytest.approx(math.log(2))
    assert m[5,2]==pytest.approx(math.log(2)) and m[5,3]==0
    assert torch.isneginf(m[2,3]) and torch.isneginf(m[0,5])
    z=p.attention_mask(6,3,0.)
    assert torch.equal(torch.isneginf(z),torch.isneginf(m))
    assert torch.all(z[torch.isfinite(z)]==0)


def test_zero_matches_native_forward():
    net=model()
    x=torch.tensor([[1,4,5,7,8]])
    with torch.inference_mode():
        assert torch.allclose(p.logits(net,x,3,0.),net(x).logits,atol=p.ATOL,rtol=0)


def test_no_future_leak_and_no_weight_mutation():
    net=model()
    before={k:v.clone() for k,v in net.state_dict().items()}
    with torch.inference_mode():
        x=torch.tensor([[1,4,5,7,8]])
        y=torch.tensor([[1,4,5,7,9]])
        left=p.logits(net,x,3,p.BIAS)
        right=p.logits(net,y,3,p.BIAS)
        assert torch.allclose(left[:,:4],right[:,:4],atol=p.ATOL,rtol=0)
        assert not torch.equal(left,p.logits(net,x,3,0.))
    assert all(torch.equal(v,net.state_dict()[k]) for k,v in before.items())


def test_invalid_bias_and_prompt():
    for l,n,b in ((4,5,0.),(4,1,0.),(4,2,1.),(385,2,0.)):
        with pytest.raises(ValueError):
            p.attention_mask(l,n,b)


def test_screen_requires_five_models():
    def row(c,t):
        return {"control":{"train":{"both_choices_correct":c}},"bias":{"train":{"both_choices_correct":t}}}
    models={str(i):row(0,int(i<4)) for i in range(7)}
    assert p.screen(models)["decision"]=="NO_CONSISTENT_TRAIN_PROBE_SIGNAL"
    models["4"]=row(0,1)
    assert p.screen(models)["decision"]=="TRAIN_PROBE_SIGNAL_ONLY"


def test_prior_hash_tamper(tmp_path):
    path=tmp_path/"bad.zip"
    path.write_bytes(b"bad")
    with pytest.raises(ValueError,match="SHA"):
        p.verify_swap(path)
