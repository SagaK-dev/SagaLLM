"""Mock-only checkpoint bytes, lineage and candidate gates."""
import json

import pytest

from sagallm import java_independent_benchmark as bench
from sagallm import java_model_freeze as freeze
from test_java_exposure_shards import prepared
from test_java_private_intake import save, sha


def setup(tmp_path, monkeypatch):
    (s, p, inv, root), shard_root, checkpoint, tokenizer, result = prepared(tmp_path, monkeypatch)
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"mock base")
    ev1 = tmp_path / "pretrain-evidence.json"
    ev2 = tmp_path / "sft-evidence.json"
    save(ev1, {"mock": True})
    save(ev2, {"mock": True})
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("mock generation template")
    model = save(tmp_path / "freeze.json", {
        "schema": freeze.SCHEMA, "suite_sha256": sha(s), "provenance_sha256": sha(p),
        "inventory_sha256": sha(inv), "shard_manifest_sha256": result["manifest_sha256"],
        "tokenizer": {"path": tokenizer.name, "sha256": sha(tokenizer)},
        "stages": [
            {"kind": "pretraining", "checkpoint_path": parent.name,
             "checkpoint_sha256": sha(parent), "parent_sha256": None,
             "training_code_sha_declared": "b"*40, "evidence_path": ev1.name,
             "evidence_sha256": sha(ev1), "source_indices": [4]},
            {"kind": "sft", "checkpoint_path": checkpoint.name,
             "checkpoint_sha256": sha(checkpoint), "parent_sha256": sha(parent),
             "training_code_sha_declared": "a"*40, "evidence_path": ev2.name,
             "evidence_sha256": sha(ev2), "source_indices": [0]}],
        "generation": {"seed": 17, "max_new_tokens": 256, "decoding": "greedy",
                       "prompt_path": prompt.name, "prompt_sha256": sha(prompt),
                       "generation_code_sha_declared": "c"*40},
        "evaluator_sha256": freeze.evaluator_sha256(), "oci_image": "mock-image"})
    return s, p, inv, shard_root, model


def verify(state, root):
    s, p, inv, shard_root, model = state
    return freeze.verify_model_freeze(s, sha(s), freeze_path=model,
        expected_freeze_sha256=sha(model), provenance_path=p,
        expected_provenance_sha256=sha(p), inventory_path=inv,
        expected_inventory_sha256=sha(inv), shard_root=shard_root,
        artifact_root=root, image="mock-image")


def test_consistent_declared_chain_and_shard_gate(tmp_path, monkeypatch):
    report = verify(setup(tmp_path, monkeypatch), tmp_path)
    assert report["machine_status"] == "PREFLIGHT_ONLY"
    assert report["declared_lineage_consistent"]
    assert not report["training_lineage_verified"]
    assert not report["generation_provenance_verified"]
    assert not report["freeze_precedes_inference_verified"]
    assert report["shard_audit"]["scanned_records"] == 7


@pytest.mark.parametrize("mode", ["parent", "source", "evidence", "code", "image", "evaluator", "tokenizer", "checkpoint", "generation", "prompt"])
def test_reject_repinning_inconsistent_fields(tmp_path, monkeypatch, mode):
    state = setup(tmp_path, monkeypatch)
    model = state[-1]
    obj = json.loads(model.read_text())
    if mode == "parent":
        obj["stages"][1]["parent_sha256"] = "0"*64
    elif mode == "source":
        obj["stages"][0]["source_indices"] = [1]
    elif mode == "evidence":
        (tmp_path / "sft-evidence.json").write_text("changed")
    elif mode == "code":
        obj["stages"][1]["training_code_sha_declared"] = "f"*40
    elif mode == "image":
        obj["oci_image"] = "different"
    elif mode == "evaluator":
        obj["evaluator_sha256"] = "0"*64
    elif mode == "tokenizer":
        obj["tokenizer"]["sha256"] = "0"*64
    elif mode == "checkpoint":
        (tmp_path / "parent.pt").write_bytes(b"changed")
    elif mode == "prompt":
        (tmp_path / "prompt.txt").write_text("changed")
    else:
        obj["generation"]["max_new_tokens"] = 4096
    save(model, obj)
    with pytest.raises(ValueError):
        verify(state, tmp_path)


def test_candidate_freeze_hash_and_missing(tmp_path):
    root = tmp_path / "candidate"
    root.mkdir()
    code = root / "task_a.java"
    code.write_bytes(b"public class Main {}\r\n")
    manifest = save(tmp_path / "candidate.json", {
        "schema": freeze.CANDIDATES, "model_freeze_sha256": "a"*64,
        "candidates": {"task_a": sha(code), "task_b": None}})
    checked = freeze.read_candidate_freeze(manifest, sha(manifest), "a"*64,
                                          ["task_a", "task_b"], root)
    assert checked["task_a"].endswith("\r\n")
    assert checked["task_b"] is None
    code.write_text("modified")
    with pytest.raises(ValueError, match="Candidate SHA"):
        freeze.read_candidate_freeze(manifest, sha(manifest), "a"*64,
                                    ["task_a", "task_b"], root)


@pytest.mark.parametrize("mode,reason", [
    ("absent", "PINNED_MODEL_FREEZE_REQUIRED"),
    ("bad_model", "MODEL_FREEZE_INVALID"),
    ("bad_shard", "MODEL_FREEZE_INVALID"),
    ("no_candidate_pin", "PINNED_CANDIDATE_FREEZE_REQUIRED"),
    ("bad_candidate", "CANDIDATE_FREEZE_INVALID"),
])
def test_actual_runtime_gate_precedes_engine(tmp_path, monkeypatch, mode, reason):
    s, p, inv, shard_root, model = setup(tmp_path, monkeypatch)
    root = tmp_path / "candidates"
    root.mkdir()
    monkeypatch.setattr(bench, "review_gate", lambda *a, **k: ("ELIGIBLE_SELF_ATTESTED", "a"*64))
    def forbidden(*a, **k):
        raise AssertionError("Engine or generated Java must not run")
    for name in ("local_docker_context", "oci_image_ready", "run_java_in_oci"):
        monkeypatch.setattr(bench, name, forbidden)
    kwargs = dict(freeze_path=model, expected_freeze_sha256=sha(model),
                  provenance_path=p, expected_provenance_sha256=sha(p),
                  inventory_path=inv, expected_inventory_sha256=sha(inv),
                  shard_root=shard_root, artifact_root=tmp_path)
    if mode == "absent":
        kwargs.pop("freeze_path")
    elif mode == "bad_model":
        kwargs["expected_freeze_sha256"] = "0"*64
    elif mode == "bad_shard":
        (shard_root / "source-000-00000.jsonl").unlink()
    elif mode == "bad_candidate":
        doc = save(tmp_path / "candidate.json", {"invalid": True})
        kwargs.update(candidate_freeze_path=doc, expected_candidate_freeze_sha256=sha(doc))
    report = bench.evaluate_suite(s, root, allow_execution=True,
        disposable_vm_acknowledged=True, expected_suite_sha256=sha(s),
        image="mock-image", **kwargs)
    assert report["reason"] == reason
    assert report["behavioral_status"] == "NOT_RUN"
    assert report["java_executed"] is False


def test_actual_gate_with_pinned_sources_and_mocked_oci(tmp_path, monkeypatch):
    s, p, inv, shard_root, model = setup(tmp_path, monkeypatch)
    root = tmp_path / "candidates"
    root.mkdir()
    tasks = json.loads(s.read_text())["tasks"]
    pins = {}
    for task in tasks:
        code = root / (task["id"] + ".java")
        code.write_bytes(b"public class Main {}\n")
        pins[task["id"]] = sha(code)
    candidate = save(tmp_path / "candidate.json", {
        "schema": freeze.CANDIDATES, "model_freeze_sha256": sha(model),
        "candidates": pins})
    monkeypatch.setattr(bench, "review_gate", lambda *a, **k: ("ELIGIBLE_SELF_ATTESTED", "a"*64))
    monkeypatch.setattr(bench, "local_docker_context", lambda: True)
    monkeypatch.setattr(bench, "local_image_has_no_volumes", lambda image: True)
    monkeypatch.setattr(bench, "oci_image_ready", lambda *a: True)
    calls = []
    def simulated(source, expected, **kw):
        calls.append(source)
        return {"status": "PASS"}
    monkeypatch.setattr(bench, "run_java_in_oci", simulated)
    report = bench.evaluate_suite(s, root, allow_execution=True,
        disposable_vm_acknowledged=True, expected_suite_sha256=sha(s),
        image="mock-image", freeze_path=model, expected_freeze_sha256=sha(model),
        provenance_path=p, expected_provenance_sha256=sha(p),
        inventory_path=inv, expected_inventory_sha256=sha(inv),
        shard_root=shard_root, artifact_root=tmp_path,
        candidate_freeze_path=candidate, expected_candidate_freeze_sha256=sha(candidate))
    assert len(calls) == 10
    assert report["behavior_passed_tasks"] == 5
    assert report["model_freeze"]["training_lineage_verified"] is False
    assert "prompt_path" not in report["model_freeze"]["generation"]
    assert report["model_selection"] == "NOT_PERMITTED_NO_INDEPENDENCE_PROOF"
