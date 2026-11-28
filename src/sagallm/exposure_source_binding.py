"""Reconcile a prospective CPU SFT journal against exact TRAIN source bytes.

Checks recorded token membership, not authenticity of the training process.
No checkpoint deserialization, model execution, or validation/test split access.
"""
from __future__ import annotations

import json
from pathlib import Path

from sagallm.exposure_trace import (
    MAX_BYTES, MAX_SEQUENCE, batch_record, digest, file_sha, is_sha,
    require, verify_trace, _unique,
)

MAX_SOURCE_BYTES=96*1024*1024
MAX_MANIFEST_BYTES=65536
MAX_TOKENIZER_BYTES=8*1024*1024
MAX_SOURCE_ROWS=100_000
MAX_CHECKPOINT_BYTES=96*1024*1024


def bounded(path,expected,limit):
    p=Path(path)
    require(is_sha(expected) and p.is_file() and not p.is_symlink(),
            "Invalid pinned source path or digest")
    with p.open("rb") as f:
        raw=f.read(limit+1)
    require(0<len(raw)<=limit and digest(raw)==expected,"Pinned source bytes mismatch")
    return raw


def replay_source_binding(trace_path, *, expected_trace_sha256, expected_tip,
                          dataset_dir, tokenizer_path, base_checkpoint, context):
    from sagallm.java_instruction_sft import encode_row, pack
    from sagallm.tokenization import load_tokenizer
    require(type(context) is int and 1<=context<=MAX_SEQUENCE,"Invalid requested context")
    require(is_sha(expected_trace_sha256) and is_sha(expected_tip),"External trace pins required")
    verified=verify_trace(trace_path,expected_sha256=expected_trace_sha256,expected_tip=expected_tip)
    require(verified["status"]=="FINALIZED_LOCAL_TRACE" and verified["pending_step"] is None,
            "Incomplete trace cannot be certified")
    raw=bounded(trace_path,expected_trace_sha256,MAX_BYTES)
    events=[json.loads(line,object_pairs_hook=_unique) for line in raw.splitlines()]
    header=events[0]["payload"]
    root=Path(dataset_dir)
    require(root.is_dir() and not root.is_symlink(),"Unsafe dataset directory")
    manifest=json.loads(bounded(root/"manifest.json",header["data_manifest_sha256"],MAX_MANIFEST_BYTES),
                        object_pairs_hook=_unique)
    train=manifest["files"]["train"]
    require(train["filename"]=="train.jsonl" and train["sha256"]==header["train_split_sha256"]
            and type(train["count"]) is int and 1<=train["count"]<=MAX_SOURCE_ROWS,
            "Manifest TRAIN declaration mismatch")
    corpus=bounded(root/"train.jsonl",header["train_split_sha256"],MAX_SOURCE_BYTES)
    require(corpus.endswith(b"\n"),"TRAIN split missing final newline")
    lines=corpus.splitlines()
    require(len(lines)==train["count"],"TRAIN row count mismatch")
    del corpus
    bounded(tokenizer_path,header["tokenizer_sha256"],MAX_TOKENIZER_BYTES)
    base=Path(base_checkpoint)
    require(base.is_file() and not base.is_symlink()
            and 0<base.stat().st_size<=MAX_CHECKPOINT_BYTES
            and file_sha(base)==header["base_checkpoint_sha256"],"Base checkpoint digest mismatch")
    tokenizer=load_tokenizer(tokenizer_path)
    require(file_sha(tokenizer_path)==header["tokenizer_sha256"],"Tokenizer changed while reading")

    # Only encode the source records referenced by the journal; check uniqueness
    # across the entire TRAIN split to avoid ambiguous ID-to-bytes bindings.
    wanted={r["id_sha256"] for e in events if e["kind"]=="step_intent" for r in e["payload"]["rows"]}
    selected={}; seen=set()
    for line in lines:
        row=json.loads(line,object_pairs_hook=_unique)
        ident=row["id"]
        require(isinstance(ident,str) and 0<len(ident)<=512,"Invalid TRAIN identity")
        key=digest(b"row-id-v1\0"+ident.encode())
        require(key not in seen,"Duplicate or ambiguous TRAIN row ID")
        seen.add(key)
        if key in wanted:
            encoded=encode_row(row,tokenizer,context)
            require(encoded is not None,"Recorded row does not fit requested context")
            selected[key]=encoded
    require(set(selected)==wanted,"Recorded row is absent from pinned TRAIN split")
    rows_checked=0; targets=0; steps_checked=0
    for e in events:
        if e["kind"]!="step_intent":
            continue
        payload=e["payload"]
        rows=[selected[r["id_sha256"]] for r in payload["rows"]]
        x,y=pack(rows)
        rebuilt=batch_record(rows,x,y)
        require(rebuilt=={k:v for k,v in payload.items() if k not in ("step","optimizer_lr")},
                "Recorded batch does not match re-encoded TRAIN inputs")
        rows_checked+=len(rows)
        targets+=rebuilt["loss_targets"]
        steps_checked+=1
    require(steps_checked==verified["completed_optimizer_returns"],"Unmatched completed step")
    return {"schema":"sagallm-exposure-trace-source-binding-v1",
            "status":"RECORDED_TRAIN_INPUTS_MATCH_PINNED_SOURCE",
            "trace_sha256":expected_trace_sha256,"tip_sha256":expected_tip,
            "data_manifest_sha256":header["data_manifest_sha256"],
            "train_split_sha256":header["train_split_sha256"],
            "tokenizer_sha256":header["tokenizer_sha256"],
            "base_checkpoint_sha256":header["base_checkpoint_sha256"],
            "train_source_rows":len(lines),"distinct_recorded_rows":len(selected),
            "recorded_row_exposures_rechecked":rows_checked,"packed_batches_rechecked":steps_checked,
            "loss_targets_rechecked":targets,"requested_reencoding_context":context,
            "validation_or_test_source_read":False,"checkpoint_deserialized":False,
            "actual_training_process_attested":False,"gradient_influence_verified":False,
            "historical_exposure_verified":False,"independent_evaluation":"NOT_RUN",
            "model_selection_permitted":False}
