"""Synthetic byte fixtures only; no model load or Java execution."""
import json

import pytest

from sagallm import java_exposure_shards as shards
from test_java_private_intake import approved_mock, change_exposure, save, sha


def prepared(tmp_path, monkeypatch):
    paths = approved_mock(tmp_path)
    s, p, inv, root = paths
    # Force multiple shards; preserve CRLF and a final line without newline.
    meta = json.loads(inv.read_text())
    train = root / "train.jsonl"
    train.write_bytes(b'{"text":"alpha"}\r\n{"text":"beta"}\n{"text":"gamma"}')
    meta["files"][0]["sha256"] = sha(train)
    save(inv, meta)
    monkeypatch.setattr(shards, "MAX_SHARD_ROWS", 2)
    checkpoint = tmp_path / "mock.pt"
    tokenizer = tmp_path / "mock-tokenizer"
    checkpoint.write_bytes(b"not a model")
    tokenizer.write_bytes(b"not a tokenizer")
    output = tmp_path / "shards"
    result = shards.prepare_exposure_shards(
        inv, sha(inv), root, output, checkpoint_path=checkpoint,
        checkpoint_sha256=sha(checkpoint), tokenizer_path=tokenizer,
        tokenizer_sha256=sha(tokenizer), training_code_sha="a"*40)
    return paths, output, checkpoint, tokenizer, result


def audit(state, **overrides):
    (s, p, inv, _), output, checkpoint, tokenizer, result = state
    args = dict(provenance_path=p, expected_provenance_sha256=sha(p),
                inventory_path=inv, expected_inventory_sha256=sha(inv),
                shard_root=output, expected_manifest_sha256=result["manifest_sha256"],
                checkpoint_path=checkpoint, expected_checkpoint_sha256=sha(checkpoint),
                tokenizer_path=tokenizer, expected_tokenizer_sha256=sha(tokenizer))
    args.update(overrides)
    return shards.audit_exposure_shards(s, sha(s), **args)


def test_exact_round_trip_and_binding(tmp_path, monkeypatch):
    state = prepared(tmp_path, monkeypatch)
    paths, output, _, _, result = state
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["sources"][0]["shards"]) == 2
    joined = b"".join((output / x["path"]).read_bytes() for x in manifest["sources"][0]["shards"])
    assert joined == (paths[3] / "train.jsonl").read_bytes()
    report = audit(state)
    assert result["shards"] == report["scanned_shards"] == 6
    assert report["scanned_records"] == 7
    assert report["machine_status"] == "PREFLIGHT_ONLY"
    assert report["declared_inventory_bytes_verified"]
    assert not report["training_lineage_verified"]
    assert not report["inventory_completeness_verified"]
    assert report["behavior"] == "NOT_RUN"


@pytest.mark.parametrize("mode", ["missing", "extra", "tampered", "symlink", "checkpoint", "tokenizer"])
def test_reject_incomplete_or_changed_artifacts(tmp_path, monkeypatch, mode):
    state = prepared(tmp_path, monkeypatch)
    _, output, checkpoint, tokenizer, _ = state
    first = output / "source-000-00000.jsonl"
    if mode == "missing":
        first.unlink()
    elif mode == "extra":
        (output / "extra.jsonl").write_text("{}")
    elif mode == "tampered":
        first.write_text('{"text":"changed"}')
    elif mode == "symlink":
        first.unlink()
        first.symlink_to(output / "source-000-00001.jsonl")
    elif mode == "checkpoint":
        checkpoint.write_bytes(b"changed")
    else:
        tokenizer.write_bytes(b"changed")
    with pytest.raises(ValueError):
        audit(state)


@pytest.mark.parametrize("mode", ["reordered", "row_count", "source_hash", "omit"])
def test_repinned_manifest_cannot_hide_source_inconsistency(tmp_path, monkeypatch, mode):
    state = prepared(tmp_path, monkeypatch)
    _, output, _, _, _ = state
    path = output / "manifest.json"
    meta = json.loads(path.read_text())
    first = meta["sources"][0]
    if mode == "reordered":
        first["shards"].reverse()
    elif mode == "row_count":
        first["shards"][0]["rows"] += 1
    elif mode == "source_hash":
        first["source_sha256"] = "0" * 64
    else:
        entry = first["shards"].pop()
        (output / entry["path"]).unlink()
    save(path, meta)
    with pytest.raises(ValueError):
        audit(state, expected_manifest_sha256=sha(path))


def test_overlap_in_final_shard(tmp_path, monkeypatch):
    paths = approved_mock(tmp_path)
    s, p, inv, root = paths
    prompt = json.loads(s.read_text())["tasks"][0]["prompt"]
    meta = json.loads(inv.read_text())
    entry = meta["files"][-1]
    path = root / entry["path"]
    path.write_text('{"text":"unrelated"}\n' + json.dumps({"text": prompt}))
    entry["sha256"] = sha(path)
    save(inv, meta)
    monkeypatch.setattr(shards, "MAX_SHARD_ROWS", 1)
    checkpoint = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    checkpoint.write_bytes(b"mock checkpoint")
    tokenizer.write_bytes(b"mock tokenizer")
    output = tmp_path / "shards"
    result = shards.prepare_exposure_shards(inv, sha(inv), root, output,
        checkpoint_path=checkpoint, checkpoint_sha256=sha(checkpoint),
        tokenizer_path=tokenizer, tokenizer_sha256=sha(tokenizer),
        training_code_sha="b"*40)
    report = audit((paths, output, checkpoint, tokenizer, result))
    assert report["machine_status"] == "BLOCKED"
    assert {"task_index": 0, "group": "pretraining", "kind": "NORMALIZED_PROMPT_MATCH"} in report["overlap_leads"]


def test_prepare_failure_cleans_stage(tmp_path, monkeypatch):
    paths = approved_mock(tmp_path)
    _, _, inv, root = paths
    checkpoint = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    checkpoint.write_bytes(b"mock")
    tokenizer.write_bytes(b"mock")
    output = tmp_path / "shards"
    monkeypatch.setattr(shards, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(ValueError, match="byte limit"):
        shards.prepare_exposure_shards(inv, sha(inv), root, output,
            checkpoint_path=checkpoint, checkpoint_sha256=sha(checkpoint),
            tokenizer_path=tokenizer, tokenizer_sha256=sha(tokenizer),
            training_code_sha="b"*40)
    assert not output.exists()
    assert not list(tmp_path.glob(".exposure-*"))
