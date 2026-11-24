"""Synthetic-only tests; no human approvals, network, model or Java execution."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sagallm.java_private_intake import (
    GROUPS, SCHEMA, INVENTORY_SCHEMA, audit_private_intake, known_families,
    task_sha256,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def fixture(tmp_path):
    suite = json.loads((Path(__file__).resolve().parents[1] /
                        "examples/java_independent_io_DRAFT.json").read_text())
    # Artificial unique labels and contents used only to exercise the screen.
    tasks = []
    for i, word in enumerate(("quartz", "basalt", "granite", "marble", "slate")):
        tasks.append({"id": f"mock_task_{i}", "family": f"mock_family_{i}",
                      "prompt": f"Mock fixture text about {word} only, not a benchmark.",
                      "cases": [{"stdin": "a", "stdout": "b"},
                                {"stdin": "c", "stdout": "d"}]})
    suite["tasks"] = tasks
    s = save(tmp_path / "suite.json", suite)
    p = save(tmp_path / "provenance.json", {
        "schema": SCHEMA, "suite_sha256": sha(s), "tasks": [
            {"task_id": t["id"], "task_sha256": task_sha256(t),
             "author_id": "mock-only", "authored_on": "2020-01-01",
             "revision": "mock-v1", "source_url": "https://example.invalid/task",
             "license_id": "PENDING", "permission_url": "https://example.invalid/rights",
             "allowed_uses": [], "held_back": False, "rights_review": "PENDING",
             "io_review": "PENDING", "overlap_review": "PENDING"} for t in tasks]})
    root = tmp_path / "exposure"
    root.mkdir()
    rows = []
    for group in GROUPS:
        path = save(root / (group + ".jsonl"), {"text": "unrelated fixture"})
        rows.append({"group": group, "path": path.name, "sha256": sha(path),
                     "format": "text-jsonl"})
    inv = save(tmp_path / "inventory.json", {"schema": INVENTORY_SCHEMA, "files": rows})
    return s, p, inv, root


def run(paths, **overrides):
    s, p, inv, root = paths
    args = dict(expected_suite_sha256=sha(s), provenance_path=p,
                expected_provenance_sha256=sha(p), inventory_path=inv,
                expected_inventory_sha256=sha(inv), exposure_root=root)
    args.update(overrides)
    return audit_private_intake(s, **args)


def change_exposure(paths, group, obj, kind="text-jsonl"):
    _, _, inv, root = paths
    meta = json.loads(inv.read_text())
    row = next(r for r in meta["files"] if r["group"] == group)
    path = save(root / row["path"], obj)
    row.update(sha256=sha(path), format=kind)
    save(inv, meta)


def test_preflight_remains_blocked_without_real_review(tmp_path):
    report = run(fixture(tmp_path))
    assert report["machine_status"] == "BLOCKED"
    assert report["review_gate"] == "REVIEW_REQUIRED"
    assert report["missing_exposure_groups"] == []
    assert len(report["scanned_files"]) == 5
    assert report["overlap_leads"] == []
    assert not report["inventory_completeness_verified"]
    assert report["human_reviews_verified"] == 0
    assert not report["generated_java_executed"]
    assert "mock-only" not in json.dumps(report)
    assert "quartz" not in json.dumps(report)


def test_normalized_substring_and_chat_match(tmp_path):
    paths = fixture(tmp_path)
    prompt = json.loads(paths[0].read_text())["tasks"][0]["prompt"]
    change_exposure(paths, "train", {"messages": [
        {"role": "user", "content": prompt.upper().replace(" ", "  ")}]}, "chat-jsonl")
    change_exposure(paths, "pretraining", {"text": "prefix " + prompt + " suffix"})
    result = run(paths)
    assert {"task_index": 0, "group": "train", "kind": "NORMALIZED_PROMPT_MATCH"} in result["overlap_leads"]
    assert {"task_index": 0, "group": "pretraining", "kind": "PROMPT_SUBSTRING_MATCH"} in result["overlap_leads"]


def test_known_families_cover_historical_and_stdio():
    labels = known_families()
    assert {"hello", "stdin_add_offset"} <= labels["train"]
    assert "stdin_min_offset" in labels["validation"]
    assert "collatz_max" in labels["test"]
    assert "dna_hamming_distance" in labels["development"]


def test_numeric_variant_and_family_match(tmp_path):
    paths = fixture(tmp_path)
    s, p, _, _ = paths
    suite = json.loads(s.read_text())
    suite["tasks"][0]["prompt"] = "Return the calculation using parameter 17."
    suite["tasks"][0]["family"] = "stdin_add_offset"
    save(s, suite)
    meta = json.loads(p.read_text())
    meta["suite_sha256"] = sha(s)
    meta["tasks"][0]["task_sha256"] = task_sha256(suite["tasks"][0])
    save(p, meta)
    change_exposure(paths, "test", {"prompt": "Return the calculation using parameter 42."}, "prompt-jsonl")
    hits = run(paths)["overlap_leads"]
    assert {"task_index": 0, "group": "train", "kind": "KNOWN_FAMILY_OR_ID"} in hits
    assert {"task_index": 0, "group": "test", "kind": "NUMERIC_TEMPLATE_MATCH"} in hits


def test_missing_group_is_explicit(tmp_path):
    paths = fixture(tmp_path)
    inv = paths[2]
    doc = json.loads(inv.read_text())
    doc["files"].pop()
    save(inv, doc)
    result = run(paths)
    assert result["missing_exposure_groups"] == ["pretraining"]
    assert "EXPOSURE_GROUPS_MISSING" in result["blockers"]


@pytest.mark.parametrize("target", ["suite", "provenance", "inventory"])
def test_pins_are_required(tmp_path, target):
    paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA"):
        run(paths, **{"expected_" + target + "_sha256": "0" * 64})


def test_task_mutation_requires_new_provenance(tmp_path):
    paths = fixture(tmp_path)
    s, p, _, _ = paths
    suite = json.loads(s.read_text())
    suite["tasks"][0]["cases"][0]["stdout"] = "changed"
    save(s, suite)
    meta = json.loads(p.read_text())
    meta["suite_sha256"] = sha(s)
    save(p, meta)
    with pytest.raises(ValueError, match="Task provenance hash"):
        run(paths)


def test_training_use_forbidden(tmp_path):
    paths = fixture(tmp_path)
    p = paths[1]
    meta = json.loads(p.read_text())
    meta["tasks"][0]["allowed_uses"] = ["train"]
    save(p, meta)
    with pytest.raises(ValueError, match="training access"):
        run(paths)


def test_exposure_tampering_and_bad_path(tmp_path):
    paths = fixture(tmp_path)
    _, _, inv, root = paths
    save(root / "train.jsonl", {"text": "modified"})
    with pytest.raises(ValueError, match="SHA mismatch"):
        run(paths)
    meta = json.loads(inv.read_text())
    meta["files"][0]["path"] = "../train.jsonl"
    save(inv, meta)
    with pytest.raises(ValueError, match="Unsafe"):
        run(paths)


def test_symlink_and_malformed_jsonl(tmp_path):
    paths = fixture(tmp_path)
    _, _, inv, root = paths
    path = root / "train.jsonl"
    target = root / "test.jsonl"
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlinked"):
        run(paths)
    path.unlink()
    change_exposure(paths, "train", {"messages": [{"role": "unknown", "content": "a"}]}, "chat-jsonl")
    with pytest.raises(ValueError, match="Malformed"):
        run(paths)
