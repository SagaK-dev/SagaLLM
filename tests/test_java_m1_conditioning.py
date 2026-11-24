"""Pure diagnostic invariants; no Java or model execution in unit tests."""
import json
import pytest
from scripts import diagnose_java_m1_conditioning as m


def test_stop_reasons():
    assert m.stop_reason([8, 2], 2, 4) == "EOS"
    assert m.stop_reason([8, 9], 2, 2) == "TOKEN_LIMIT"
    assert m.stop_reason([8], 2, 2) == "UNEXPLAINED_SHORT_STOP"
    for bad in ([], [2, 9], [8, 9, 10]):
        with pytest.raises(ValueError):
            m.stop_reason(bad, 2, 2)


def test_shape_only_fixed_markers():
    s = "public class Main {System.out.println(System.in); /* PRIVATE_MARKER */}"
    result = m.shape(s)
    assert result["main_class_marker"] and result["stdin_marker"] and result["stdout_marker"]
    assert result["raw_brace_counts_equal"]
    assert "PRIVATE_MARKER" not in json.dumps(result)
    assert m.shape("\x00\ufffd{")["nul_present"]
    assert not m.shape("{")["raw_brace_counts_equal"]


def test_hash_and_prefix():
    assert m.token_hash([1, 23]) != m.token_hash([12, 3])
    assert m.common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert m.common_prefix([1], [1, 2]) == 1
    assert m.common_prefix([], [1]) == 0


def test_sha_pin(tmp_path):
    p = tmp_path / "evidence"
    p.write_bytes(b"fixed")
    assert m.read_pinned(p, m.digest(b"fixed"), 20) == b"fixed"
    with pytest.raises(ValueError, match="SHA"):
        m.read_pinned(p, m.digest(b"wrong"), 20)
    with pytest.raises(ValueError):
        m.read_pinned(p, m.digest(b"fixed"), 2)
    q = tmp_path / "link"
    q.symlink_to(p)
    with pytest.raises(ValueError):
        m.read_pinned(q, m.digest(b"fixed"), 20)


def fixture():
    def arm():
        return {"total_cases": 18, "cases_passed": 0, "oci_invocations": 18,
                "cases": [{"id": task, "cases_passed": 0,
                           "case_categories": ["OUTPUT_MISMATCH_ZERO_EXIT"]*3}
                          for task in m.TASKS]}
    return {"experiment": "JAVA_CONTRASTIVE_STDIO_MATCHED_AB_V1",
            "holdback_behavior_task_ids": list(m.TASKS),
            "baseline_behavior": arm(),
            "arms_results": {a: {"behavior": arm()} for a in m.ARMS[1:]}}


def test_historical_counts():
    assert m.historical_counts(fixture())["base_v018"]["case_categories"] == {"OUTPUT_MISMATCH_ZERO_EXIT": 18}


@pytest.mark.parametrize("mutation", ["case_id", "category", "score", "arm", "holdback"])
def test_historical_corruption(mutation):
    r = fixture()
    if mutation == "case_id":
        r["baseline_behavior"]["cases"][0]["id"] = "unknown"
    elif mutation == "category":
        r["baseline_behavior"]["cases"][0]["case_categories"][0] = "RAW_SOURCE"
    elif mutation == "score":
        r["baseline_behavior"]["cases_passed"] = 18
    elif mutation == "arm":
        del r["arms_results"]["control_interleaved"]
    else:
        r["holdback_behavior_task_ids"].reverse()
    with pytest.raises(ValueError):
        m.historical_counts(r)
