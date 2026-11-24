import pytest
import torch
from sagallm.config import ModelConfig
from sagallm.model import SagaTransformer
from scripts import diagnose_java_m3_activation_swap as m


def test_regions_position_alignment():
    p={"prefixes":[[1,2,3],[1,4,3]],"common_response_prefix":[7,8]}
    assert m.regions(p)=={"changed_prompt":(1,),"unchanged_prompt":(0,2),"final_response":(4,),
                          "all_prompt":(0,1,2),"all_positions":(0,1,2,3,4)}
    p["prefixes"][1].append(9)
    with pytest.raises(ValueError):
        m.regions(p)


def test_swap_only_selected_positions_and_no_mutation():
    x=torch.arange(24).view(2,4,3)
    old=x.clone()
    y=m.swap_output(x,(1,3))
    assert torch.equal(y[:,[1,3]],x.flip(0)[:,[1,3]])
    assert torch.equal(y[:,[0,2]],x[:,[0,2]])
    assert torch.equal(x,old) and torch.equal(m.swap_output(x,(1,),True),x)
    assert torch.equal(m.swap_output((x,None),(1,3))[0],y)
    for positions in ((),(1,1),(4,),(-1,)):
        with pytest.raises(ValueError):
            m.swap_output(x,positions)


def test_effect_direction():
    clean=torch.tensor([[3.,1.],[2.,4.]])
    e=m.effect(clean,clean.flip(0),[0,1])
    assert e["clean_interaction"]==4 and e["patched_interaction"]==-4
    assert e["mean_donor_directed_shift"]==4 and e["both_directions_positive"]
    assert m.effect(clean,clean,[0,1])["mean_donor_directed_shift"]==0


def test_real_tiny_model_hooks_and_cleanup():
    torch.manual_seed(5)
    model=SagaTransformer(ModelConfig(32,16,2,2,1,32,32)).eval()
    pair={"ids":["a","b"],"family":"f","prefixes":[[1,4,5],[1,6,5]],
          "common_response_prefix":[7,8],"alternatives":[10,11]}
    x=torch.tensor([[1,4,5,7,8],[1,6,5,7,8]])
    with torch.inference_mode():
        clean=model(x).logits[:,-1,:].float()
        l,r=m.preference(clean,pair["alternatives"])
        previous={"ids":pair["ids"],"divergence_choice":{"left_correct_minus_other":l,"right_correct_minus_other":-r}}
        out=m.measure(model,pair,previous)
        assert len(out["sites"])==7 and out["control_checks"]>0
        assert torch.equal(model(x).logits[:,-1,:],clean)
        assert all(not module._forward_hooks for module in m.sites(model).values())


def test_hook_removed_on_failed_intervention():
    model=SagaTransformer(ModelConfig(32,16,1,2,1,32,32)).eval()
    with torch.inference_mode(),pytest.raises(ValueError):
        m.patched_logits(model,model.embedding,torch.tensor([[1,2],[1,3]]),(99,))
    assert not model.embedding._forward_hooks


def test_prior_hash_tampering(tmp_path):
    f=tmp_path/"prior.zip"
    f.write_bytes(b"bad")
    with pytest.raises(ValueError,match="SHA"):
        m.prior_report(f)
