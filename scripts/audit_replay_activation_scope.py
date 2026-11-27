"""Audit archived replay, binding and activation scope; no model or Java execution."""
import argparse
import io
import json
from pathlib import Path
import zipfile

from scripts.audit_ranking_auxiliary import LIMIT, TOKENIZER, pin, parse, sha, require
from scripts.audit_prompt_probe_scope import BASE, PAIR, CE_ZIP, SPECS as CE_SPECS, pair_ids

PRIOR_RUN = 46002937944
PRIOR_ARTIFACT = 15199246826
PRIOR_ZIP = "7f01297b890aef4915ee030e1ba64ad396729e8815075ee12cf96462d7b93c70"
RANK_ZIP = "2d4d6bf653c394ddf75016b26099efcf2fc5eb09ec1edd1086d7baedbedcc97f"
SIGNAL_ZIP = "46b2df5cdb251e7abb3c11796c9cc0bc92402133474e1d56e9f0c0ee4253075f"
ORACLES = "74ea37e0dcf69175894124a7ff982037960b5d5afb0dc0181eebe71321b400a7"
SUITE = "d641a53e6498a096fda15e92871e10db6cc2380fe1bb1be543c8f4a06aeb035a"
SPECS = (
    (45373241422, "m3-ranking-v2", "90c8b9bb12e381c415e798d88875fc401143a0c703a3b0424cfab6c2283932cd"),
    (45374203447, "m3-frozen-replay", "2676251393c7d589aabd50cc651229f4bc6576dec88be048a13dd295d6e2ef71"),
    (45374793097, "m3-parameter-binding", "b222a4d7ea8149b30f73c159459ee774a097902298f612ec2df4888d654fcc68"),
    (45386333879, "m3-activation-swap", "3d8436e4cfee22aee4918a8718a5eb270d94f7a3af832adb724ff4691e47742c"),
    CE_SPECS[0], CE_SPECS[1],
)
SEEDS = ("4201", "4202", "4203")
SPLITS = ("regression", "fresh_development")


def hexsha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def prior(path):
    raw = pin(path, PRIOR_ZIP)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = z.infolist()
        require(len(members) == 1 and members[0].filename == "prompt-probe-scope.json"
                and 0 < members[0].file_size <= LIMIT, "Unexpected prior ZIP")
        with z.open(members[0]) as f:
            body = f.read(LIMIT + 1)
        require(len(body) == members[0].file_size <= LIMIT, "Prior member size")
    d = parse(body)
    require(d["schema"] == "sagallm-prompt-probe-scope-v1"
            and d["status"] == "ARCHIVED_PROMPT_PROBE_IDENTITIES_MATCH"
            and d["reports"] == [{"run_id": r, "slug": s, "report_sha256": h}
                                 for r, s, h in CE_SPECS]
            and d["inventory_entries"] == 15 and d["new_inventory_entries"] == 0
            and hexsha(d["inventory_sha256"])
            and len(d["models"]) == 7
            and d["signal_pair_comparisons"] == 140 and d["bias_pair_comparisons"] == 322
            and d["bias_regression_row_comparisons"] == 84
            and d["actual_exposure_verified"] is False and d["training_lineage_verified"] is False
            and d["full_exposure_inventory"] == "NOT_ESTABLISHED"
            and d["inventory_ready_for_independent_evaluation"] is False
            and d["independent_evaluation"] == "NOT_RUN"
            and d["model_selection_permitted"] is False, "Prior gate mismatch")
    return d, sha(body)


def load_reports(root):
    return [parse(pin(root / f"{run}-{slug}.json", digest, 8 * 1024 * 1024))
            for run, slug, digest in SPECS]


def generation_summary(diag, expected):
    rows = diag["rows"]
    require(len(rows) == len(expected) == 6 and [r["id"] for r in rows] == expected
            and len(set(expected)) == 6, "Generation ID coverage mismatch")
    for r in rows:
        require(r["compile"] == r["behavior"] == "NOT_RUN" and r["stop"] == "EOS"
                and type(r["tokens"]) is int and 1 <= r["tokens"] <= 170
                and hexsha(r["source_sha256"]), "Generation metadata mismatch")
    require(all(rows[i]["family"] == rows[i+1]["family"] for i in (0, 2, 4))
            and len({r["family"] for r in rows}) == 3, "Family mismatch")
    hashes = [r["source_sha256"] for r in rows]
    cross = sum(a["family"] != b["family"] and a["source_sha256"] == b["source_sha256"]
                for i, a in enumerate(rows) for b in rows[i+1:])
    require(diag["unique_source_hashes"] == len(set(hashes))
            and diag["cross_family_identical_pairs"] == cross, "Generation aggregate mismatch")
    return {"unique": len(set(hashes)), "cross_family_collisions": cross,
            "same_source_within_family_pairs": sum(hashes[i] == hashes[i+1] for i in (0, 2, 4)),
            "eos": 6}


def models(report, arm):
    require(set(report["results"]) == set(SEEDS), "Seed coverage mismatch")
    out = {"base": (BASE, report["baseline"], {})}
    for seed in SEEDS:
        arms = report["results"][seed]
        require(set(arms) == {"control", arm}, "Arm coverage mismatch")
        for name, row in arms.items():
            out[seed + "-" + name] = (row["checkpoint_sha256"], row["diagnostics"], row)
    pins = [r[0] for r in out.values()]
    require(len(set(pins)) == 7 and all(hexsha(p) for p in pins), "Checkpoint declaration mismatch")
    return out


def common(report, status):
    require(report["status"] == status and report["base_sha256"] == BASE
            and report["tokenizer_sha256"] == TOKENIZER
            and report["production_promotion"] is False, "Report scope mismatch")


def no_execution(report):
    require(report["compile"] == report["behavior"] == "NOT_RUN"
            and report["independent_benchmark"] == "NOT_ESTABLISHED", "Execution gate mismatch")


def replay_and_binding(rank, replay, binding, ce):
    common(rank, "MATCHED_CPU_TRAINING_AND_INFERENCE_COMPLETE")
    common(replay, "FROZEN_REPLAY_COMPLETE")
    common(binding, "PARAMETER_BINDING_DIAGNOSTIC_COMPLETE")
    for d in (replay, binding):
        no_execution(d)
        require(d["archive_sha256"] == RANK_ZIP, "Ranking archive declaration mismatch")
    require(rank["holdback_sha256"] == replay["fresh_suite_sha256"] == SUITE
            and replay["regression_oracles_sha256"] == ORACLES
            and replay["report_sha256"] == binding["prior_report_sha256"] == SPECS[0][2]
            and replay["run_id"] == SPECS[0][0] and replay["artifact_id"] == 14909169047
            and replay["review_authenticity"] == "NOT_VERIFIED_BY_SOFTWARE"
            and replay["checkpoint_hashes_verified"] == 6 and replay["source_hash_matches"] == 84
            and binding["source_hash_replay_matches"] == 84 and binding["training"] == "NOT_RUN"
            and binding["root_cause"] == "NOT_ESTABLISHED", "Replay or binding pin mismatch")
    require(binding["train_pair_preflight"] == {"directional_rows": 320, "families": 10,
            "manifest_sha256": PAIR, "optimizer_steps": 0, "pairs": 160}, "Pair preflight mismatch")
    records = models(rank, "ranking")
    require(set(replay["summaries"]) == set(replay["deterministic_bounds"]) ==
            set(binding["models"]) == set(records), "Ranking model coverage mismatch")
    result = []
    for key, (cp, diag, arm) in records.items():
        expected = {}
        for split, ids in zip(SPLITS, (rank["regression_ids"], rank["fresh_ids"])):
            expected[split] = generation_summary(diag[split], ids)
            bounds = replay["deterministic_bounds"][key][split]
            require(bounds["assumption"] == "DETERMINISTIC_STDOUT_FOR_IDENTICAL_SOURCE_AND_STDIN"
                    and bounds["cases"] == 18 and bounds["observed_passes"] is None
                    and type(bounds["conditional_correct_case_upper_bound"]) is int
                    and 0 <= bounds["conditional_correct_case_upper_bound"] <= 18
                    and type(bounds["incompatible_expectation_groups"]) is int
                    and 0 <= bounds["incompatible_expectation_groups"] <= 18,
                    "Conditional bound mislabeled or malformed")
        expected["regression_margin"] = diag["regression_aligned_margin"]["mean_wrong_minus_own_nll"]
        if key != "base":
            expected.update(checkpoint_sha256=cp, validation_nll=arm["best_validation_nll"])
        require(expected == replay["summaries"][key], "Recounted replay summary mismatch")
        b = binding["models"][key]
        require(b["prior_generation_collisions"] == {s: expected[s]["same_source_within_family_pairs"]
                                                     for s in SPLITS}, "Binding collision mismatch")
        for group, count, prefix in (("train", 20, "train-"), ("validation", 3, "validation-")):
            require(pair_ids(b[group], count, prefix, lengths=True) ==
                    pair_ids(ce["baseline"]["forced_choice"][group], count, prefix, lengths=True),
                    "Binding pair identity mismatch")
        result.append({"model": key, "checkpoint_sha256_declared_by_ranking": cp,
                       "generation_rows_recounted": 12, "binding_pairs_compared": 23})
    return result


def activation_scope(act, signal, ce, previous):
    common(act, "ACTIVATION_SWAP_DIAGNOSTIC_COMPLETE")
    no_execution(act)
    require(act["archive_sha256"] == CE_ZIP and act["training_report_sha256"] == CE_SPECS[0][2]
            and act["prior_diagnostic_report_sha256"] == CE_SPECS[1][2]
            and act["prior_diagnostic_zip_sha256"] == SIGNAL_ZIP
            and act["pair_manifest_sha256"] == PAIR and act["training"] == "NOT_RUN"
            and act["root_cause"] == "NOT_ESTABLISHED" and act["probe_split"] == "TRAIN_EXPOSED"
            and act["equal_length_pairs"] == 19 and act["matched_model_pairs"] == 133
            and act["control_checks"] == 3192, "Activation scope mismatch")
    records = models(ce, "parameter_ce")
    previous_models = previous["models"]
    require(len(previous_models) == 7 and {r["model"]: r["checkpoint_sha256_declared_by_ce"]
            for r in previous_models} == {k: v[0] for k, v in records.items()}
            and set(act["models"]) == set(signal["models"]) == set(records), "CE model scope mismatch")
    result = []
    for key, (cp, _, _) in records.items():
        src = signal["models"][key]["rows"]
        require(len(src) == 20 and all(type(r["equal_prompt_length"]) is bool
                and r["equal_prompt_length"] == (r["prompt_lengths"][0] == r["prompt_lengths"][1])
                for r in src), "Equal-length declaration mismatch")
        excluded = [r["ids"] for r in src if not r["equal_prompt_length"]]
        require(len(excluded) == 1 and excluded == act["excluded_unequal_length"], "Excluded pair mismatch")
        expected = [[r["ids"], r["family"], r["common_prefix_tokens"], r["prompt_lengths"][0]]
                    for r in src if r["equal_prompt_length"]]
        model = act["models"][key]
        actual = [[r["ids"], r["family"], r["common_response_tokens"], r["prompt_tokens"]]
                  for r in model["rows"]]
        require(len(expected) == model["pairs"] == 19 and actual == expected
                and model["control_checks"] == 456
                and all(r["control_checks"] == 24 for r in model["rows"]), "Activation row scope mismatch")
        result.append({"model": key, "checkpoint_sha256_declared_by_ce": cp,
                       "equal_length_pairs_compared": 19, "control_checks_recounted": 456})
    return result


def run(root, prior_zip):
    previous, member = prior(prior_zip)
    rank, replay, binding, act, ce, signal = load_reports(root)
    ranking = replay_and_binding(rank, replay, binding, ce)
    activation = activation_scope(act, signal, ce, previous)
    return {"schema": "sagallm-replay-activation-scope-v1", "status": "ARCHIVED_REPLAY_BINDING_ACTIVATION_SCOPE_MATCH",
            "reports": [{"run_id": r, "slug": s, "report_sha256": h} for r, s, h in SPECS],
            "prior_evidence": {"run_id": PRIOR_RUN, "artifact_id": PRIOR_ARTIFACT,
                               "archive_sha256": PRIOR_ZIP, "member_sha256": member},
            "inventory_sha256": previous["inventory_sha256"], "inventory_entries": 15,
            "new_inventory_entries": 0, "ranking_models": ranking, "ce_models": activation,
            "replay_generation_rows_recounted": 84, "binding_pairs_compared": 161,
            "activation_pairs_compared": 133, "activation_control_checks_recounted": 3192,
            "conditional_bounds_recomputed": False, "probe_scores_recomputed": False,
            "probe_source_bytes_reconstructed": False, "checkpoint_bytes_verified_here": False,
            "actual_exposure_verified": False, "training_lineage_verified": False,
            "full_exposure_inventory": "NOT_ESTABLISHED", "shards_prepared": False,
            "inventory_ready_for_independent_evaluation": False,
            "rights_review": "REVIEW_REQUIRED", "independent_evaluation": "NOT_RUN",
            "model_selection_permitted": False,
            "remaining_archived_probes": ["m1-conditioning", "m1-conditioning-rerun",
                                         "contrastive-stdio-ab", "m3-alignment-v1-blocked"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("experiments/archive/reports"))
    parser.add_argument("--prior-zip", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.reports, args.prior_zip), sort_keys=True, indent=2))
