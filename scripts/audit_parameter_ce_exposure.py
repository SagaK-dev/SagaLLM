"""Reconstruct historical Parameter CE input/target declarations without models."""
import argparse
import io
import json
from pathlib import Path
import zipfile

from scripts.audit_ranking_auxiliary import canonical, parse, pin, require, sha, LIMIT, TOKENIZER

REVISION = "fdf53625d8e64596748113b53fd6f4c63e2a1c74"
REPORT_RUN = 45377422360
REPORT_SHA = "26f143575842073c9b4119ecefcb3b50935891bba6c4c9fc60b0b0c8353a89b7"
PRIOR_RUN = 45957422504
PRIOR_ARTIFACT = 15179611408
PRIOR_ZIP = "567493712a2b5358938cc7910c639c7d748a546d67709e49d4a13469a2955a0a"


def prior(path):
    raw = pin(path, PRIOR_ZIP)
    names = {"scoped-ranking-inventory.json", "ranking-auxiliary-evidence.json"}
    blobs = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = z.infolist()
        require(len(members) == 2 and {m.filename for m in members} == names,
                "Unexpected prior archive members")
        for m in members:
            require(0 < m.file_size <= LIMIT, "Prior member size")
            with z.open(m) as f:
                body = f.read(LIMIT + 1)
            require(0 < len(body) <= LIMIT, "Prior member size")
            blobs[m.filename] = body
    inv = parse(blobs["scoped-ranking-inventory.json"])
    report = parse(blobs["ranking-auxiliary-evidence.json"])
    require(inv["schema"] == "sagallm-private-java-exposure-v1"
            and len(inv["files"]) == report["inventory_entries"] == 12
            and report["inventory_sha256"] == sha(blobs["scoped-ranking-inventory.json"])
            and report["status"] == "HISTORICAL_RANKING_DECLARATIONS_RECONSTRUCTED"
            and report["derived_rows"] == 640 and report["derived_entry"] in inv["files"]
            and report["actual_exposure_verified"] is False
            and report["full_exposure_inventory"] == "NOT_ESTABLISHED"
            and report["inventory_ready_for_independent_evaluation"] is False
            and report["model_selection_permitted"] is False
            and report["independent_evaluation"] == "NOT_RUN", "Prior scope mismatch")
    return inv, {name: sha(body) for name, body in blobs.items()}


def verify_report(report, ce, ranking):
    require(report["status"] == "PARAMETER_CE_CPU_TRAINING_AND_INFERENCE_COMPLETE"
            and report["protocol"] == ce.PROTOCOL
            and report["base_sha256"] == ce.a.BASE
            and report["tokenizer_sha256"] == TOKENIZER
            and report["broad_train_sha256"] == ranking.TRAIN_SHA
            and report["stdio_manifest_sha256"] == ranking.STDIO_SHA
            and report["pair_manifest_sha256"] == ce.PAIR_SHA
            and report["previous_suite_sha256"] == ranking.SUITE_SHA
            and report["fresh_suite_sha256"] == ce.SUITE_SHA
            and report["seeds"] == list(ranking.SEEDS)
            and report["steps_per_arm"] == 160
            and report["same_aux_graph"] is True
            and report["fresh_used_for_training_or_selection"] is False
            and report["original_test_records_read"] is False
            and report["production_promotion"] is False
            and report["behavior"] == report["compile"] == "NOT_RUN",
            "Historical report scope mismatch")


def compare_schedule(report, seed, digest, checkpoints):
    require(report["schedule_sha256_by_seed"][str(seed)] == digest, "Seed digest mismatch")
    arms = report["results"][str(seed)]
    require(set(arms) == {"control", "parameter_ce"}, "Arm coverage mismatch")
    for arm, coefficient in (("control", 0), ("parameter_ce", 1)):
        row = arms[arm]
        require(row["schedule_sha256"] == digest and row["coefficient"] == coefficient
                and row["steps"] == row["selected_step"] == 160
                and row["broad_rows_seen"] == row["stdio_rows_seen"] == 320
                and row["auxiliary_directions"] == 320, "Reported schedule scope mismatch")
        digest_cp = row["checkpoint_sha256"]
        require(isinstance(digest_cp, str) and len(digest_cp) == 64
                and all(c in "0123456789abcdef" for c in digest_cp)
                and digest_cp not in checkpoints, "Invalid or repeated checkpoint")
        checkpoints.add(digest_cp)


def screening_rows(pairs, lookup, tok, compact_prompt):
    """Two prompts x two alternatives; text is derived, not serialized tensors."""
    rows, seen = [], set()
    for p in pairs:
        require(len(p["ids"]) == len(p["prefixes"]) == len(p["alternatives"]) == 2
                and p["ids"][0] != p["ids"][1], "Pair shape mismatch")
        left, right = [lookup[key] for key in p["ids"]]
        refs = [tok.encode(row["source"]) for row in (left, right)]
        pos = p["position"]
        require(type(pos) is int and 0 <= pos < min(map(len, refs))
                and refs[0][:pos] == refs[1][:pos] == p["common_response_prefix"]
                and [ref[pos] for ref in refs] == p["alternatives"]
                and refs[0][pos] != refs[1][pos], "First-divergence mismatch")
        require(all(tok.decode(ref) == row["source"] and row["family"] == p["family"]
                    for ref, row in zip(refs, (left, right))), "Reference mismatch")
        for direction, row in enumerate((left, right)):
            prompt = compact_prompt(row["task"])
            prefix = [tok.SPECIAL["<bos>"]] + tok.encode(prompt)
            require(prefix == p["prefixes"][direction]
                    and len(prefix) + pos <= 384, "Prompt input mismatch")
            for alternative in (0, 1):
                key = (row["id"], alternative)
                require(key not in seen, "Duplicate screening direction")
                seen.add(key)
                # Decode the reference prefix together with the actual candidate token.
                # These project-authored Java references are ASCII.
                suffix_ids = p["common_response_prefix"] + [p["alternatives"][alternative]]
                suffix = tok.decode(suffix_ids)
                require(suffix.isascii() and (left, right)[alternative]["source"].startswith(suffix),
                        "Candidate token text mismatch")
                rows.append({"id": row["id"], "family": p["family"],
                             "alternative_id": p["ids"][alternative],
                             "correct": direction == alternative, "text": prompt + suffix})
    require(len(rows) == 4*len(pairs), "Screening coverage mismatch")
    return rows


def verify_diagnostic_scope(report, groups):
    """Recount only identities and token lengths, never scores or logits."""
    diagnostics = [report["baseline"]]
    for arms in report["results"].values():
        diagnostics.extend(row["diagnostics"] for row in arms.values())
    require(len(diagnostics) == 7, "Diagnostic model count mismatch")
    for diag in diagnostics:
        data = diag["forced_choice"]
        require(set(data) == set(groups), "Diagnostic group coverage mismatch")
        for name, pairs in groups.items():
            expected = [(p["ids"], p["family"], p["position"],
                         [len(x) for x in p["prefixes"]]) for p in pairs]
            actual = [(p["ids"], p["family"], p["common_reference_tokens"],
                       p["prompt_token_counts"]) for p in data[name]["rows"]]
            require(data[name]["pairs"] == len(expected) and actual == expected,
                    "Diagnostic identity or token length mismatch")


def add_entry(inventory, root, name, group, rows):
    raw = b"".join(canonical(row) for row in rows)
    require(0 < len(raw) <= LIMIT, "Derived file size")
    relative = "derived/" + name + ".jsonl"
    require(not any(e["path"] == relative for e in inventory["files"]), "Duplicate path")
    with (root / relative).open("xb") as f:
        f.write(raw)
    entry = {"group": group, "path": relative, "sha256": sha(raw), "format": "text-jsonl"}
    inventory["files"].append(entry)
    return {"entry": entry, "rows": len(rows), "bytes": len(raw)}


def run(args):
    from scripts import train_java_m3_parameter_ce as ce
    from scripts import train_java_m3_ranking as ranking
    from scripts import diagnose_java_m3_parameters as diagnostic
    from sagallm.tokenization import load_tokenizer

    report = parse(pin(args.report, REPORT_SHA, 8*1024*1024))
    verify_report(report, ce, ranking)
    inventory, prior_hashes = prior(args.prior_zip)
    pin(args.tokenizer, TOKENIZER)
    tok = load_tokenizer(args.tokenizer)
    fresh = ce.load_suite(args.historical / "examples/java_m3_parameter_holdback.json")
    ranking.frozen_suite(args.historical / "examples/java_m3_development_holdback.json")
    require([x["id"] for x in fresh] == report["fresh_ids"], "Fresh identity mismatch")
    ranking.build_curriculum(args.root / "stdio")
    pin(args.root / "stdio/manifest.json", ranking.STDIO_SHA)
    broad = ranking.broad_rows()
    def encoded(split):
        rows = [ce.encode_row(row, tok, 384) for row in ce.expected_rows(split)]
        require(all(row is not None for row in rows), "Context admission mismatch")
        return {row["id"]: row for row in rows}
    train, valid = encoded("train"), encoded("validation")
    require(len(train) == 320 and len(valid) == 36, "STDIO row count mismatch")
    require(list(ce.a.SELECTION) == report["selection_ids"]
            and len(report["selection_ids"]) == 24
            and set(report["selection_ids"]) <= set(valid), "Selection mismatch")
    pairs = ce.train_pairs(train, tok)
    groups, preflight = diagnostic.build_pairs(tok)
    require(len(pairs) == 160 and preflight["manifest_sha256"] == ce.PAIR_SHA
            and preflight["pairs"] == 160 and preflight["directional_rows"] == 320
            and preflight["families"] == 10 and preflight["optimizer_steps"] == 0,
            "Pair manifest mismatch")
    groups["fresh"] = ce.fresh_pair_rows(fresh, tok)
    verify_diagnostic_scope(report, groups)
    require(set(report["results"]) == set(report["schedule_sha256_by_seed"])
            == {str(s) for s in ranking.SEEDS}, "Seed coverage mismatch")
    schedules, checkpoints = {}, set()
    for seed in ranking.SEEDS:
        schedule = ranking.schedule_for(broad, list(pairs), seed)
        digest = sha(ranking.canonical([[[broad[i]["id"] for i in b], list(ids)]
                                       for b, ids in schedule]))
        compare_schedule(report, seed, digest, checkpoints)
        schedules[str(seed)] = digest
    (args.root / "derived").mkdir()
    rows = screening_rows(list(pairs.values()), train, tok, ce.compact_prompt)
    validation = screening_rows(groups["validation"], valid, tok, ce.compact_prompt)
    fresh_rows = screening_rows(groups["fresh"], {x["id"]: x for x in fresh},
                                tok, ce.compact_prompt)
    sources = [add_entry(inventory, args.root, name, group, records)
               for name, group, records in (
                   ("parameter_ce_train", "train", rows),
                   ("parameter_ce_validation", "development", validation),
                   ("parameter_ce_fresh", "development", fresh_rows))]
    raw = canonical(inventory)
    with args.inventory_output.open("xb") as f:
        f.write(raw)
    return {"schema": "sagallm-parameter-ce-exposure-evidence-v1",
            "status": "HISTORICAL_PARAMETER_CE_DECLARATIONS_RECONSTRUCTED",
            "historical_revision": REVISION, "report_run_id": REPORT_RUN,
            "report_sha256": REPORT_SHA, "tokenizer_sha256": TOKENIZER,
            "prior_evidence": {"run_id": PRIOR_RUN, "artifact_id": PRIOR_ARTIFACT,
                               "archive_sha256": PRIOR_ZIP, "member_sha256": prior_hashes},
            "pair_preflight": preflight, "schedule_sha256_by_seed": schedules,
            "checkpoint_count": len(checkpoints), "checkpoint_sha256": sorted(checkpoints),
            "diagnostic_scope_checked_models": 7, "diagnostic_scores_recomputed": False,
            "derived_sources": sources, "inventory_sha256": sha(raw),
            "inventory_entries": len(inventory["files"]),
            "representation": "prompt-common-reference-prefix-plus-alternative-v1",
            "prior_source_bytes_rechecked_here": False,
            "actual_exposure_verified": False, "training_lineage_verified": False,
            "full_exposure_inventory": "NOT_ESTABLISHED", "shards_prepared": False,
            "inventory_ready_for_independent_evaluation": False,
            "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("report", "prior_zip", "tokenizer", "historical", "root", "inventory_output"):
        p.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    print(json.dumps(run(p.parse_args()), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
