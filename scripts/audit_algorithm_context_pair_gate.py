"""Replay the pinned admission audit and quantify bilingual pair coverage.

Metadata-only planning. No model load, training, inference, Java execution,
heldout scoring, dataset export or change to the existing RoPE policy.
"""
import argparse
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

CONTEXTS = (192, 256, 384, 512, 768)
ADMISSION_RUN = 46129012984
ADMISSION_ARTIFACT = 15256607793
ADMISSION_ZIP = "ce28b14a82f1436a6c355cf81e2cf888884001d1114f7beb116ebed6add3d12b"
ADMISSION_SIZE = 3358
ADMISSION_HEAD = "12c78986737f0787475e08dbeeb6fc0c43bbe38b"
MEMBERSHIP = "9369de0c1dc2d8a034cc9a4c8f209654867699d3ff8cddbc13acf1189d2ad2b5"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode()


def read_admission(path):
    require(path.is_file() and not path.is_symlink()
            and path.stat().st_size == ADMISSION_SIZE, "Admission ZIP size/type drift")
    raw = path.read_bytes()
    require(digest(raw) == ADMISSION_ZIP, "Admission ZIP hash drift")
    with zipfile.ZipFile(path) as archive:
        require(archive.namelist() == ["algorithm-admission.json"], "Archive membership drift")
        member = archive.infolist()[0]
        require(not member.is_dir() and not member.flag_bits & 1
                and 0 < member.file_size <= 128 * 1024, "Unexpected admission member")
        report_bytes = archive.read(member)
    return json.loads(report_bytes), digest(report_bytes)


def summarize_rows(rows):
    """Require exact complete bilingual groups, then calculate pair-aware admission."""
    require(len(rows) == 16, "Expected 16 records")
    groups = defaultdict(dict)
    seen_ids, seen_hashes = set(), set()
    for row in rows:
        row_id, row_hash = row["id"], row["candidate_row_sha256"]
        require(row_id not in seen_ids and row_hash not in seen_hashes, "Duplicate row")
        seen_ids.add(row_id)
        seen_hashes.add(row_hash)
        key = (row["family"], row["variant"])
        lang = row["language"]
        require(lang in ("ja", "en") and lang not in groups[key], "Invalid language group")
        prompt, response = row["prompt_tokens"], row["response_tokens_without_eos"]
        require(type(prompt) is int and type(response) is int and prompt > 0 and response > 0,
                "Invalid token counts")
        # encode_row: BOS + prompt + source + EOS, shifted by one position.
        positions = prompt + response + 1
        require(row["admitted_contexts"] == [n for n in CONTEXTS if positions <= n],
                "Admission does not match exact shifted sequence length")
        require(set(row["canonical_matches"]) == {"train", "validation", "test"}
                and all(set(matches) == {"task", "source", "signature"}
                        and all(value is False for value in matches.values())
                        for matches in row["canonical_matches"].values()),
                "Static overlap or missing overlap evidence")
        groups[key][lang] = (row, positions)
    require(len(groups) == 8 and len({family for family, _ in groups}) == 4,
            "Unexpected curriculum grouping")
    for pair in groups.values():
        require(set(pair) == {"ja", "en"}, "Incomplete bilingual pair")
        ja, en = pair["ja"][0], pair["en"][0]
        require(ja["source_sha256"] == en["source_sha256"]
                and ja["task_sha256"] != en["task_sha256"], "Bilingual source/task drift")
    require(len({r["source_sha256"] for r in rows}) == 8, "Reference source count drift")
    families = {family for family, _ in groups}
    require(all(sum(family == f for family, _ in groups) == 2 for f in families),
            "Expected two implementations per family")
    pairs = []
    for (family, variant), pair in sorted(groups.items()):
        pairs.append({"family": family, "variant": variant,
                      "source_sha256": pair["ja"][0]["source_sha256"],
                      "members": {lang: {"id": r["id"], "row_sha256": r["candidate_row_sha256"],
                                         "sequence_positions": positions}
                                  for lang, (r, positions) in sorted(pair.items())},
                      "minimum_complete_pair_context": max(x[1] for x in pair.values())})
    by_context = {}
    for n in CONTEXTS:
        accepted = [r for r in rows if n in r["admitted_contexts"]]
        complete = [p for p in pairs if p["minimum_complete_pair_context"] <= n]
        complete_families = sorted(f for f in families
                                   if sum(p["family"] == f for p in complete) == 2)
        paired_hashes = sorted(m["row_sha256"] for p in complete for m in p["members"].values())
        by_context[str(n)] = {
            "admitted_rows": len(accepted),
            "language_counts": {lang: sum(r["language"] == lang for r in accepted)
                                for lang in ("ja", "en")},
            "complete_bilingual_pairs": len(complete),
            "partial_pairs": len(accepted) - 2 * len(complete),
            "complete_families": complete_families,
            "all_rows_admitted": len(accepted) == len(rows),
            "paired_subset_membership_sha256": digest(canonical(paired_hashes)),
            "excluded_row_ids": sorted(r["id"] for r in rows if n not in r["admitted_contexts"]),
        }
    return {"pairs": pairs, "by_context": by_context,
            "minimum_full_context": max(p["minimum_complete_pair_context"] for p in pairs),
            "membership_sha256": digest(canonical(sorted(seen_hashes)))}


def policy_matrix():
    # Small synthetic config checks only the actual policy function; not checkpoint
    # inspection or a forward/backward feasibility test.
    from sagallm.config import ModelConfig
    from sagallm.java_instruction_sft import extend_rope_context
    result = {}
    for origin in (192, 384):
        cfg = ModelConfig(vocab_size=32, hidden_size=16, num_layers=1,
                          num_heads=2, num_kv_heads=1, intermediate_size=32,
                          max_position_embeddings=origin)
        result[str(origin)] = {}
        for target in CONTEXTS:
            if target < origin:
                status = "NOT_AN_EXTENSION"
            elif target == origin:
                status = "UNCHANGED"
            else:
                try:
                    extended = extend_rope_context(cfg, target)
                except ValueError:
                    status = "BLOCKED_BY_EXISTING_POLICY"
                else:
                    require(extended.max_position_embeddings == target
                            and extended.parameter_count() == cfg.parameter_count(),
                            "Extension changed model shape")
                    status = "POLICY_ALLOWS_EXTENSION"
            result[str(origin)][str(target)] = status
    return result


def run(admission_zip, reference_zip, parent_zip):
    from scripts.audit_algorithm_curriculum_admission import run as replay_admission
    old, report_hash = read_admission(admission_zip)
    fresh = replay_admission(reference_zip, parent_zip)
    require(old == fresh, "Replayed admission report differs from historical report")
    require(old["candidate_membership_sha256"] == MEMBERSHIP
            and old["training_performed"] is False
            and old["weights_loaded"] is False
            and old["test_tokenization_or_scoring"] is False
            and old["rights_review"] == "REVIEW_REQUIRED", "Admission scope drift")
    summary = summarize_rows(fresh["rows"])
    require(summary["membership_sha256"] == MEMBERSHIP, "Membership drift")
    require({n: v["admitted_rows"] for n, v in summary["by_context"].items()}
            == old["admission_by_context"], "Aggregate count drift")
    matrix = policy_matrix()
    require(matrix["192"]["384"] == "POLICY_ALLOWS_EXTENSION"
            and matrix["192"]["512"] == "BLOCKED_BY_EXISTING_POLICY"
            and matrix["384"]["768"] == "BLOCKED_BY_EXISTING_POLICY",
            "Existing policy changed; review required")
    viable = [n for n in CONTEXTS
              if summary["by_context"][str(n)]["all_rows_admitted"]
              and matrix["192"][str(n)] in ("UNCHANGED", "POLICY_ALLOWS_EXTENSION")]
    require(not viable, "Full admission under current policy requires new review")
    return {"schema": "sagallm-algorithm-context-pair-gate-v1",
            "status": "FULL_CURRICULUM_BLOCKED_BY_CONTEXT_POLICY",
            "admission_run_id": ADMISSION_RUN, "admission_artifact_id": ADMISSION_ARTIFACT,
            "admission_head_sha": ADMISSION_HEAD, "admission_zip_sha256": ADMISSION_ZIP,
            "admission_report_sha256": report_hash, "reference_artifact_id": old["reference_artifact_id"],
            "reference_report_sha256": old["reference_report_sha256"],
            "tokenizer_sha256": old["tokenizer_sha256"],
            "canonical_manifest_sha256": old["canonical_manifest_sha256"],
            "admission_replay_equal": True, **summary, "policy_matrix": matrix,
            "policy_probe_config": "SYNTHETIC_CONFIG_ONLY",
            "full_context_under_current_policy": viable,
            "next_options": ["VERIFY_COMPACT_REFERENCE_VARIANTS_AT_384",
                             "SEPARATELY_REVIEW_LONG_CONTEXT_RESOURCE_AND_POLICY"],
            "training_performed": False, "weights_loaded": False,
            "new_java_execution": False, "test_tokenization_or_scoring": False,
            "dataset_published": False, "semantic_overlap_verified": False,
            "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
            "long_context_model_readiness": "NOT_ESTABLISHED",
            "model_selection_permitted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("admission-zip", "reference-zip", "parent-zip", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "Refuse overwrite")
    result = run(args.admission_zip, args.reference_zip, args.parent_zip)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, sort_keys=True, indent=2)
        out.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "pairs"}, sort_keys=True))
