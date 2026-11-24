import math
import pytest
import torch
from scripts import diagnose_java_m3_prompt_signal as s


def test_depths():
    assert s.depths(99)==[0,24,49,74,99]
    with pytest.raises(ValueError):
        s.depths(3)


def test_distances_zero_and_shift_invariant():
    x=torch.tensor([1.,2.,3.])
    assert s.distances(x,x)=={"centered_logit_rms":0.,"js_nats":0.}
    d=s.distances(x,x+10)
    assert d["centered_logit_rms"]==0. and d["js_nats"]<1e-12


def test_distances_symmetric_and_finite():
    x=torch.tensor([1.,2.,3.]); y=torch.tensor([3.,2.,1.])
    v=s.distances(x,y)
    assert v==s.distances(y,x) and 0<v["js_nats"]<=math.log(2)
    assert v["centered_logit_rms"]>0.
    with pytest.raises(ValueError):
        s.distances(x,torch.tensor([0.,float("nan"),1.]))


def arm(c,u):
    return {"diagnostics":{"forced_choice":{"fresh":{"both_choices_correct":c}},
                           "generation":{"fresh":{"unique_source_hashes":u}}}}


def test_preregistered_screen_requires_both_in_two_seeds():
    r={k:{"control":arm(0,3),"parameter_ce":arm(1,3)} for k in ("4201","4202","4203")}
    assert s.screen(r)["decision"]=="FORCED_CHOICE_SIGNAL_ONLY"
    r["4201"]["parameter_ce"]=arm(1,2)
    r["4202"]["parameter_ce"]=arm(0,6)
    assert s.screen(r)["decision"]=="NO_CONSISTENT_PARAMETER_SIGNAL"
    assert s.screen(r)["passing_seeds"]==1


def test_tampered_archive(tmp_path):
    p=tmp_path/"artifact.zip";p.write_bytes(b"not the archive")
    with pytest.raises(ValueError,match="SHA"):
        s.audit(p)


def test_empty_equal_length_stratum():
    assert s.summarize([])=={"pairs":0,"status":"NOT_AVAILABLE"}
