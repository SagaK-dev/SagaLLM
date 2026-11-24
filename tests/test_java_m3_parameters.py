from types import SimpleNamespace
import math
import pytest
from scripts import diagnose_java_m3_parameters as d


def test_first_divergence():
    assert d.first_difference([1,2,3,4],[1,2,8,9])==(2,3,8)
    for x,y in (([],[1]),([1],[1]),([1],[1,2])):
        with pytest.raises(ValueError):
            d.first_difference(x,y)


def test_score_logits_two_correct_choices():
    r=d.score_logits(3,1,0,2)
    assert r["interaction"]==4 and r["both_choices_correct"]
    assert r["correct_choices"]==2


def test_no_prompt_response_interaction_for_shared_preference():
    r=d.score_logits(3,1,3,1)
    assert r["interaction"]==0 and not r["both_choices_correct"]
    assert r["correct_choices"]==1


def test_ties_and_nonfinite():
    assert d.score_logits(0,0,0,0)["correct_choices"]==0
    with pytest.raises(ValueError):
        d.score_logits(math.inf,0,0,0)


def test_reject_same_prompt_or_wrong_family():
    a={"id":"a","family":"f","task":"1","prefix_ids":[1,3]}
    for b in ({**a,"id":"b","task":"2"},{**a,"id":"b","family":"g"}):
        with pytest.raises(ValueError):
            d.verify_pair(a,b,None)


def test_aggregate():
    rows=[d.score_logits(3,1,0,2),d.score_logits(3,1,3,1)]
    r=d.aggregate(rows)
    assert r["pairs"]==2 and r["both_choices_correct"]==1
    assert r["correct_choices"]==3 and r["positive_interactions"]==1
    assert r["mean_interaction"]==2
