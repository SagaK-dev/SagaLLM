"""Rebuild ranking masks and schedules from the historical code, without models."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

REVISION = "babcbe130905d4acc7e4617497f0ebcea68569a9"
REPORT_SHA = "90c8b9bb12e381c415e798d88875fc401143a0c703a3b0424cfab6c2283932cd"
REPORT_RUN = 45373241422
PRIOR_RUN = 45955292307
PRIOR_ARTIFACT = 15178332987
PRIOR_ZIP = "a7d15ef29336efeecf2b52f9a4a7c9cd000e94f55bc2c8f03c5e1e375792353f"
TOKENIZER = "186566808cb03e328cac9a6b7b384a8d0aefc54ccb83fbe97782a751726682fe"
LIMIT = 2_000_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf8")


def unique(pairs):
    out = {}
    for k, v in pairs:
        require(k not in out, "Duplicate JSON key")
        out[k] = v
    return out


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique)


def pin(path, digest, limit=LIMIT):
    require(path.is_file() and not path.is_symlink(), "Unsafe evidence path")
    with path.open("rb") as f:
        raw = f.read(limit + 1)
    require(0 < len(raw) <= limit and sha(raw) == digest, "Evidence pin mismatch")
    return raw


def prior(path):
    raw = pin(path, PRIOR_ZIP)
    names = {"scoped-development-inventory.json", "development-inventory-evidence.json"}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = z.infolist()
        require(len(members) == 2 and {m.filename for m in members} == names,
                "Unexpected prior members")
        blobs = {}
        for m in members:
            require(0 < m.file_size <= LIMIT, "Prior member too large")
            with z.open(m) as f:
                body = f.read(LIMIT + 1)
            require(0 < len(body) <= LIMIT, "Prior member too large")
            blobs[m.filename] = body
    inv = parse(blobs["scoped-development-inventory.json"])
    report = parse(blobs["development-inventory-evidence.json"])
    require(inv["schema"] == "sagallm-private-java-exposure-v1"
            and len(inv["files"]) == 11
            and report["inventory_sha256"] == sha(blobs["scoped-development-inventory.json"])
            and report["status"] == "SCOPED_DEVELOPMENT_SOURCE_BYTES_VERIFIED"
            and report["ancestor_entries_referenced"] == 6 and report["new_entries"] == 5
            and report["full_exposure_inventory"] == "NOT_ESTABLISHED"
            and report["actual_exposure_verified"] is False
            and report["inventory_ready_for_independent_evaluation"] is False
            and report["model_selection_permitted"] is False
            and report["independent_evaluation"] == "NOT_RUN", "Prior scope mismatch")
    return inv, {name: sha(body) for name, body in blobs.items()}


def verify_report(report, ranking):
    require(report["status"] == "MATCHED_CPU_TRAINING_AND_INFERENCE_COMPLETE"
            and report["base_sha256"] == ranking.PINNED_V018
            and report["tokenizer_sha256"] == TOKENIZER
            and report["broad_train_sha256"] == ranking.TRAIN_SHA
            and report["stdio_manifest_sha256"] == ranking.STDIO_SHA
            and report["holdback_sha256"] == ranking.SUITE_SHA
            and report["seeds"] == list(ranking.SEEDS)
            and report["steps_per_arm"] == ranking.STEPS == 160
            and report["mask_protocol"] == "operation-span-v2"
            and report["same_schedule_per_seed"] is True and report["same_aux_graph"] is True
            and report["fresh_used_in_training_or_selection"] is False
            and report["original_test_records_read"] is False
            and report["production_promotion"] is False, "Historical report scope mismatch")


def compare_schedule(report, seed, digest, checkpoints, expected_aux):
    arms = report["results"][str(seed)]
    require(set(arms) == {"control", "ranking"}, "Arm coverage mismatch")
    for arm, coefficient in (("control", 0), ("ranking", 1)):
        row = arms[arm]
        require(row["schedule_sha256"] == digest and row["coefficient"] == coefficient
                and row["steps"] == 160 and row["selected_step"] == 160
                and row["broad_rows_seen"] == row["stdio_rows_seen"] == 320
                and row["aux_rows_seen"] == expected_aux, "Schedule or reported scope mismatch")
        checkpoint = row["checkpoint_sha256"]
        require(isinstance(checkpoint, str) and len(checkpoint) == 64
                and all(c in "0123456789abcdef" for c in checkpoint)
                and checkpoint not in checkpoints, "Duplicate or invalid checkpoint")
        checkpoints.add(checkpoint)


def auxiliary_rows(ordered, lookup, aux, tok, ranking):
    """Represent full own/wrong prompt and reference, not a model-generated answer."""
    rows = []
    for pair in ordered:
        require(len(pair) == 2 and pair[0] != pair[1], "Invalid pair")
        for own, other in (pair, pair[::-1]):
            row = lookup[own]
            v = aux[own]
            require(v is not None, "Missing admitted row")
            tensors, positions = v
            target = tok.encode(row["source"])
            require(tok.decode(target) == row["source"], "Reference round-trip mismatch")
            for role, prompt_id, tensor in zip(("own", "wrong"), (own, other), tensors):
                prompt = ranking.compact_prompt(lookup[prompt_id]["task"])
                prefix = [tok.SPECIAL["<bos>"]] + tok.encode(prompt)
                require(prefix == lookup[prompt_id]["prefix_ids"], "Prompt encoding mismatch")
                ids = prefix + target
                require(tensor["x"].tolist() == ids[:-1], "Auxiliary input mismatch")
                expected = [-100] * (len(ids)-1)
                for pos in positions:
                    require(0 <= pos < len(target), "Invalid mask index")
                    expected[len(prefix)-1+pos] = target[pos]
                require(tensor["y"].tolist() == expected, "Auxiliary mask mismatch")
                # Full reference text includes the final token even though x omits it;
                # it is a conservative screening representation of x and y together.
                rows.append({"id": own, "prompt_id": prompt_id, "role": role,
                             "family": row["family"], "text": prompt + row["source"]})
    require(len(rows) == 640 and len({(r["id"],r["role"]) for r in rows}) == 640,
            "Auxiliary representation coverage mismatch")
    return rows


def run(args):
    from scripts import train_java_m3_ranking as ranking
    from sagallm.tokenization import load_tokenizer
    report = parse(pin(args.report, REPORT_SHA, 8*1024*1024))
    verify_report(report, ranking)
    inventory, prior_hashes = prior(args.prior_zip)
    pin(args.tokenizer, TOKENIZER)
    ranking.frozen_suite(args.historical / "examples/java_m3_development_holdback.json")
    ranking.build_curriculum(args.curriculum)
    pin(args.curriculum / "manifest.json", ranking.STDIO_SHA)
    tok = load_tokenizer(args.tokenizer)
    eos = tok.SPECIAL["<eos>"]
    # schedule_for uses only IDs/families; broad encoding is unnecessary.
    broad = ranking.broad_rows()
    stdio = [ranking.encode_row(row, tok, 384) for row in ranking.expected_rows("train")]
    require(len(stdio) == 320 and all(row is not None for row in stdio), "STDIO admission")
    lookup = {r["id"]: r for r in stdio}
    aux, ordered, admission = ranking.pairs(stdio, eos, tokenizer=tok,
                                            mask_mode="operation-span-v2")
    require(admission == report["alignment"] and admission["passed"] is True
            and admission["admitted_rows"] == 320 and len(ordered) == 160,
            "Ranking mask reconstruction mismatch")
    valid = [r for r in ranking.expected_rows("validation")]
    require([r["id"] for r in valid if int(r["id"].rsplit("-",1)[1]) >= 4]
            == report["selection_ids"] and list(ranking.REGRESSION) == report["regression_ids"],
            "Selection scope mismatch")
    require(set(report["results"]) == {str(s) for s in ranking.SEEDS}, "Seed coverage")
    checkpoints, schedules = set(), {}
    for seed in ranking.SEEDS:
        schedule = ranking.schedule_for(broad, ordered, seed)
        digest = sha(ranking.canonical([[[broad[i]["id"] for i in b], list(ids)]
                                       for b, ids in schedule]))
        compare_schedule(report, seed, digest, checkpoints, admission["admitted_rows"])
        schedules[str(seed)] = digest
    rows = auxiliary_rows(ordered, lookup, aux, tok, ranking)
    raw = b"".join(canonical(row) for row in rows)
    require(0 < len(raw) <= LIMIT, "Auxiliary screening file size")
    with args.derived_output.open("xb") as f:
        f.write(raw)
    entry = {"group": "train", "path": "derived/ranking_auxiliary.jsonl",
             "sha256": sha(raw), "format": "text-jsonl"}
    require(not any(e["path"] == entry["path"] for e in inventory["files"]), "Duplicate path")
    inventory["files"].append(entry)
    inv_raw = canonical(inventory)
    with args.inventory_output.open("xb") as f:
        f.write(inv_raw)
    return {"schema": "sagallm-ranking-auxiliary-evidence-v1",
            "status": "HISTORICAL_RANKING_DECLARATIONS_RECONSTRUCTED",
            "historical_revision": REVISION, "report_run_id": REPORT_RUN,
            "report_sha256": REPORT_SHA, "tokenizer_sha256": TOKENIZER,
            "prior_evidence": {"run_id": PRIOR_RUN, "artifact_id": PRIOR_ARTIFACT,
                               "archive_sha256": PRIOR_ZIP, "member_sha256": prior_hashes},
            "alignment": admission, "schedule_sha256_by_seed": schedules,
            "checkpoint_count": len(checkpoints), "checkpoint_sha256": sorted(checkpoints),
            "derived_entry": entry, "derived_rows": len(rows), "derived_bytes": len(raw),
            "inventory_sha256": sha(inv_raw), "inventory_entries": len(inventory["files"]),
            "prior_source_bytes_rechecked_here": False,
            "representation": "full-own-and-wrong-prompt-plus-reference-v1",
            "actual_exposure_verified": False, "training_lineage_verified": False,
            "full_exposure_inventory": "NOT_ESTABLISHED", "shards_prepared": False,
            "inventory_ready_for_independent_evaluation": False,
            "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("report", "prior_zip", "tokenizer", "historical", "curriculum",
                 "derived_output", "inventory_output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    print(json.dumps(run(p.parse_args()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
